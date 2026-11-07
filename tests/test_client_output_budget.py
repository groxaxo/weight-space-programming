"""Hermes-shaped omitted limits must use the advertised serving allowance."""
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import sys
import threading
import unittest
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from serve_coder import handler_for, validate_request
from serve_coder_gguf import NativeBackend
import test_gguf_backend as control


class ClientOutputBudgetTests(unittest.TestCase):
    def request(self, **changes):
        return {'model': 'unit-model', 'messages': [{'role': 'user', 'content': 'Fix the module.'}],
                'stream': True, 'stream_options': {'include_usage': True}, **changes}

    def test_hermes_omitted_limit_survives_frontend_and_native_http_above_1024_tokens(self):
        # The old default both forwards n_predict=1024 and rejects this otherwise
        # valid 1,500-token completion. This is transport proof, not a model score.
        response = control.GgufBackendTests().result(tokens=[21] * 1500, tokens_predicted=1500)
        with control.native_control(response) as (native_url, requests):
            backend = NativeBackend({'served_model_name': 'unit-model'}, control.Tokenizer(),
                {'artifact_path': '/tmp/wsp-unit-model.gguf', 'limits': {'context': 262144, 'output': 28192}},
                native_url, 262144, 28192)
            server = ThreadingHTTPServer(('127.0.0.1', 0), handler_for(backend))
            worker = threading.Thread(target=server.serve_forever, daemon=True); worker.start()
            try:
                request = Request(f'http://127.0.0.1:{server.server_port}/v1/chat/completions',
                    data=json.dumps(self.request()).encode(), headers={'Content-Type': 'application/json'})
                with urlopen(request, timeout=5) as result:
                    stream = result.read().decode()
                self.assertIn('data: [DONE]', stream)
                self.assertIn('"completion_tokens": 1500', stream)
                self.assertEqual(requests[0]['n_predict'], 28192)
            finally:
                server.shutdown(); server.server_close(); worker.join(3)

    def test_explicit_budgets_keep_precedence_and_cannot_exceed_the_serving_cap(self):
        for changes, expected in [({}, 28192), ({'max_tokens': 512}, 512),
                                  ({'max_tokens': 512, 'max_completion_tokens': 2048}, 2048)]:
            with self.subTest(changes=changes):
                self.assertEqual(validate_request(self.request(**changes), 'unit-model', 262144, 28192)[2], expected)
        for value in (28193, True, 0):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_request(self.request(max_tokens=value), 'unit-model', 262144, 28192)

    def test_the_omitted_budget_tracks_a_smaller_configured_server_too(self):
        self.assertEqual(validate_request(self.request(), 'unit-model', 4096, 2048)[2], 2048)


if __name__ == '__main__': unittest.main()
