"""Real HTTP JSON/SSE controls for model and process attribution."""
from http.server import HTTPServer
import json
from pathlib import Path
import tempfile
import sys
import threading
import unittest
import socket
import struct
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from coder_common import CoderError
from ops.client_proxy import IdentityProxy
from ops.project_clients import api_bridge
from serve_coder import handler_for


class ClientProxyTests(unittest.TestCase):
    def setUp(self):
        class Backend:
            cfg = {'served_model_name': 'synthetic-provider'}
            identity = {'adapter_sha256': 'a' * 64}
            def complete(self, request):
                return {'id': 'control', 'created': 1, 'model': self.cfg['served_model_name'],
                        'choices': [{'message': {'role': 'assistant', 'content': 'Control.'}, 'finish_reason': 'stop'}],
                        'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2}}
        self.backend = Backend()
        self.server = HTTPServer(('127.0.0.1', 0), handler_for(self.backend))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True); self.thread.start()
        self.url = f'http://127.0.0.1:{self.server.server_port}/v1'
        with urlopen(self.url + '/models') as response: self.info = json.load(response)
    def tearDown(self):
        self.server.shutdown(); self.server.server_close(); self.thread.join(timeout=5)
    def post(self, proxy, stream=False):
        body = json.dumps({'stream': stream}).encode()
        with urlopen(Request(proxy.base_url + '/chat/completions', data=body, headers={'Content-Type': 'application/json'})) as response:
            return response.read()
    def proxy(self): return IdentityProxy(self.url, self.info['wsp'], self.info['wsp_instance'])

    def test_actual_json_and_sse_requests_are_attributed(self):
        with self.proxy() as proxy:
            self.assertIn(b'Control.', self.post(proxy))
            self.assertIn(b'[DONE]', self.post(proxy, True))
            proof = proxy.verify()
            self.assertTrue(proof['verified'])
            self.assertEqual(proof['requests'], 2)
            self.assertEqual(proof['stream_requests'], 1)

    def test_changed_model_is_blocked_before_reaching_the_client(self):
        with self.proxy() as proxy:
            self.backend.identity = {'adapter_sha256': 'b' * 64}
            with self.assertRaises(HTTPError): self.post(proxy)
            with self.assertRaises(CoderError): proxy.verify()

    def test_sse_effective_sampling_drift_is_rejected_even_with_the_same_model(self):
        self.backend.identity={'runtime':{'sampling':{'temperature':0.7,'presence_penalty':1.5}}}
        with urlopen(self.url+'/models') as response:self.info=json.load(response)
        original=self.backend.complete
        sample=dict(self.backend.identity['runtime']['sampling'])
        def completion(request):
            value=original(request)
            value['wsp']={'sampling':dict(sample)}
            return value
        self.backend.complete=completion
        with self.proxy() as proxy:
            self.post(proxy,True)
            sample['presence_penalty']=0
            with self.assertRaises(HTTPError):self.post(proxy,True)
            with self.assertRaises(CoderError):proxy.verify()

    def test_restarted_instance_with_identical_model_is_rejected(self):
        with IdentityProxy(self.url, self.info['wsp'], 'different-process') as proxy:
            with self.assertRaises(HTTPError): self.post(proxy, True)
            self.assertEqual(len(proxy.events), 0)

    def test_provider_fallback_and_no_requests_do_not_attest(self):
        with self.proxy() as proxy:
            with self.assertRaises(CoderError): proxy.verify()

    def test_deadline_waits_for_the_last_request_before_attribution(self):
        entered, release = threading.Event(), threading.Event()
        original = self.backend.complete
        def blocked(request):
            entered.set()
            release.wait(timeout=10)
            return original(request)
        self.backend.complete = blocked
        with self.proxy() as proxy:
            worker = threading.Thread(target=self.post, args=(proxy,), daemon=True)
            worker.start()
            try:
                self.assertTrue(entered.wait(timeout=3))
                with self.assertRaises(CoderError): proxy.drain(0.01)
            finally:
                release.set()
                worker.join(timeout=5)
            proxy.drain(2)
            self.assertEqual(proxy.verify()['requests'], 1)

    def test_stopped_actor_disconnect_does_not_discard_a_validated_response(self):
        entered, release = threading.Event(), threading.Event()
        original = self.backend.complete
        def blocked(request):
            entered.set()
            release.wait(timeout=10)
            return original(request)
        self.backend.complete = blocked
        with self.proxy() as proxy:
            client = socket.create_connection(proxy.server.server_address, timeout=3)
            client.sendall(b'POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: 2\r\n\r\n{}')
            try:
                self.assertTrue(entered.wait(timeout=3))
                client.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack('ii', 1, 0))
                client.close()
            finally:
                release.set()
                client.close()
            proxy.drain(3)
            self.assertEqual(proxy.verify()['requests'], 1)
            self.assertEqual(proxy.errors, [])
            self.assertEqual(proxy.delivery_disconnects, 1)

    def test_capability_discovery_404_does_not_invalidate_real_completions(self):
        with self.proxy() as proxy:
            for path in ['/v1/model/info', '/v1/responses']:
                with self.assertRaises(HTTPError) as caught: urlopen(proxy.base_url.removesuffix('/v1') + path)
                self.assertEqual(caught.exception.code, 404)
            self.post(proxy)
            proof = proxy.verify()
            self.assertEqual(proof['requests'], 1)
            self.assertEqual(proof['discovery_requests'], 2)

    def test_discovery_traffic_alone_cannot_attest_a_provider_fallback(self):
        with self.proxy() as proxy:
            with self.assertRaises(HTTPError): urlopen(proxy.base_url + '/responses')
            with self.assertRaises(CoderError): proxy.verify()

    def test_an_intermediate_error_cannot_be_hidden_by_a_later_success(self):
        with self.proxy() as proxy:
            self.backend.identity = {'other': True}
            with self.assertRaises(HTTPError): self.post(proxy)
            self.backend.identity = self.info['wsp']
            self.post(proxy)
            with self.assertRaises(CoderError): proxy.verify()

    def test_remote_or_unexpected_provider_routes_are_rejected(self):
        for url in ['https://127.0.0.1/v1', 'http://example.com/v1', self.url + '/other']:
            with self.subTest(url=url), self.assertRaises(CoderError): IdentityProxy(url, {}, 'instance')

    @unittest.skipUnless(hasattr(__import__('socket'), 'AF_UNIX'), 'project bridge requires Unix sockets')
    def test_project_container_bridge_preserves_verified_json_and_sse(self):
        with tempfile.TemporaryDirectory() as directory, self.proxy() as proxy:
            address = str(Path(directory) / 'api.sock')
            with api_bridge(address, ('127.0.0.1', proxy.server.server_port)):
                with api_bridge(('127.0.0.1', 0), address, unix_listener=False) as bridge:
                    route = f'http://127.0.0.1:{bridge.server_address[1]}/v1/chat/completions'
                    for stream in [False, True]:
                        with urlopen(Request(route, data=json.dumps({'stream': stream}).encode())) as response:
                            self.assertIn(b'Control.', response.read())
            self.assertEqual(proxy.verify()['requests'], 2)
