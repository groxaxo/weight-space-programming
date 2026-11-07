"""Real HTTP controls for client deadlines with a still-pending completion."""
from contextlib import ExitStack
from http.server import HTTPServer
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from ops import client_proxy, smoke_clients, project_clients
import evaluate_coder
from serve_coder import handler_for


class ClientFinalizationTests(unittest.TestCase):
    def control(self, project):
        entered, release = threading.Event(), threading.Event()

        class Backend:
            cfg = {'served_model_name': 'synthetic-deadline-control'}
            identity = {'adapter_sha256': 'a' * 64, 'limits': {'context': 64, 'output': 8}}
            def complete(self, request):
                entered.set()
                if not release.wait(5): raise RuntimeError('Control release deadline')
                return {'id': 'control', 'created': 1, 'model': self.cfg['served_model_name'],
                        'choices': [{'message': {'role': 'assistant', 'content': 'Control.'}, 'finish_reason': 'stop'}],
                        'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2}}

        server = HTTPServer(('127.0.0.1', 0), handler_for(Backend()))
        server_thread = threading.Thread(target=server.serve_forever, daemon=True); server_thread.start()
        url = f'http://127.0.0.1:{server.server_port}/v1'
        with urlopen(url + '/models') as response: info = json.load(response)
        proxies, workers, timers, failures = [], [], [], []
        container_alive = [False]
        original_proxy = client_proxy.IdentityProxy

        def proxy_factory(*arguments):
            proxy = original_proxy(*arguments); proxies.append(proxy); return proxy

        def fast_discovery(request, timeout):
            # A single-threaded provider cannot answer discovery while it is
            # finishing the last generation. Bound this reproduced race to
            # milliseconds rather than the real ten-second discovery timeout.
            if isinstance(request, str) and request.endswith('/models'): timeout = 0.03
            return urlopen(request, timeout=timeout)

        def execute(command, work, env, log, timeout=420):
            entered.clear(); release.clear(); proxy = proxies[-1]
            container_alive[0] = project
            def post():
                try:
                    with urlopen(Request(proxy.base_url + '/chat/completions', data=b'{}',
                                         headers={'Content-Type': 'application/json'}), timeout=3) as response:
                        response.read()
                except Exception as error: failures.append(error)
            worker = threading.Thread(target=post, daemon=True); workers.append(worker); worker.start()
            self.assertTrue(entered.wait(2), 'The synthetic client never entered generation')
            timer = threading.Timer(0.2, release.set); timers.append(timer); timer.start()
            log.write_text('')
            if not project:
                with (work / 'calc.py').open('a') as stream: stream.write('\n# synthetic file edit\n')
            # The stopped actor has reached its deadline, but its final issued
            # HTTP request has not finished. Its verdict must remain a timeout.
            return {'exit_code': None, 'timed_out': True, 'seconds': 1000.0}

        def run(command, **kwargs):
            if '--version' in command: return SimpleNamespace(returncode=0, stdout='synthetic-cli\n')
            if command[:3] == ['docker', 'rm', '-f']:
                self.assertEqual(command[3:], ['synthetic-owned-container'])
                container_alive[0] = False
            return SimpleNamespace(returncode=0, stdout='synthetic-owned-container\n' if container_alive[0] else '')

        def independent_check(*arguments):
            self.assertFalse(container_alive[0], 'A timed-out actor could still mutate the graded snapshot')
            return {'passed': False, 'tests_completed': True}

        client = SimpleNamespace(identity=info['wsp'], instance_id=info['wsp_instance'])
        module = project_clients if project else smoke_clients
        try:
            with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
                output = Path(directory) / 'result'
                arguments = ['clients', '--base-url', url, '--image', 'python:control@sha256:' + 'b' * 64,
                             '--output', str(output)]
                if project: arguments += ['--clients', 'hermes']
                stack.enter_context(patch.object(sys, 'argv', arguments))
                stack.enter_context(patch.object(sys, 'stdout', io.StringIO()))
                stack.enter_context(patch.object(client_proxy, 'urlopen', side_effect=fast_discovery))
                stack.enter_context(patch.object(client_proxy, 'IdentityProxy', side_effect=proxy_factory))
                stack.enter_context(patch.object(smoke_clients, 'IdentityProxy', side_effect=proxy_factory))
                stack.enter_context(patch.object(smoke_clients, 'execute', side_effect=execute))
                stack.enter_context(patch.object(smoke_clients, 'Client', return_value=client))
                stack.enter_context(patch.object(evaluate_coder, 'Client', return_value=client))
                stack.enter_context(patch.object(module.shutil, 'which', return_value=sys.executable))
                stack.enter_context(patch.object(module.subprocess, 'run', side_effect=run))
                if project:
                    stack.enter_context(patch.object(module.os, 'uname', return_value=SimpleNamespace(nodename='4790')))
                    stack.enter_context(patch.object(module, 'observe_containers'))
                    stack.enter_context(patch.object(module, 'independent_check', side_effect=independent_check))
                else:
                    stack.enter_context(patch.object(module, 'run_python', return_value={'passed': False, 'tests_completed': True}))
                with self.assertRaises(SystemExit) as stopped: module.main()
                self.assertEqual(stopped.exception.code, 2)
                report = json.loads((output / 'acceptance.json').read_text())
                self.assertFalse(report['passed'])
                self.assertEqual(len(report['results']), 1 if project else 2)
                for row in report['results']:
                    self.assertTrue(row['timed_out'])
                    self.assertFalse(row['passed'])
                    self.assertNotIn('error', row)
                    self.assertTrue(row['model_attestation']['verified'])
                    self.assertEqual(row['model_attestation']['requests'], 1)
                    self.assertEqual(row['seconds'], 1000.0)
                    if project: self.assertTrue(row['owned_containers_stopped_before_snapshot'])
            self.assertEqual(failures, [])
        finally:
            release.set()
            for timer in timers: timer.join(2)
            for worker in workers: worker.join(3)
            for proxy in proxies: proxy.close()
            server.shutdown(); server.server_close(); server_thread.join(3)

    def test_file_client_timeout_drains_and_attests_without_becoming_a_pass(self):
        self.control(False)

    def test_project_client_timeout_drains_and_attests_without_becoming_a_pass(self):
        self.control(True)


if __name__ == '__main__': unittest.main()
