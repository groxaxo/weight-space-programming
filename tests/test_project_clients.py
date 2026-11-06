"""Control the optional project grader and real stream transport without a model."""
import io
import json
from pathlib import Path
import socket
import socketserver
import sys
import tempfile
import threading
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ops.project_clients import api_bridge, ran_tests, configure_hermes, TESTS, INITIAL_VALIDATION, INITIAL_PLANNING

REFERENCE_VALIDATION = '''def normalize(jobs):
    result = []
    names = set()
    for job in jobs:
        if not isinstance(job, dict):
            raise ValueError("job must be a dictionary")
        name = job.get("id")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("invalid or duplicate id")
        names.add(name)
        deps = job.get("deps", ())
        if isinstance(deps, str) or deps is None:
            raise ValueError("invalid dependencies")
        try:
            deps = list(deps)
        except TypeError as exc:
            raise ValueError("invalid dependencies") from exc
        if any(not isinstance(d, str) or not d for d in deps):
            raise ValueError("invalid dependency id")
        result.append((name, tuple(dict.fromkeys(deps))))
    if any(d not in names for _, deps in result for d in deps):
        raise ValueError("unknown dependency")
    return result
'''
REFERENCE_PLANNING = '''from .validation import normalize
import heapq

def plan(jobs):
    records = normalize(jobs)
    order = {name: i for i, (name, _) in enumerate(records)}
    waiting = {name: len(deps) for name, deps in records}
    children = {name: [] for name, _ in records}
    for name, deps in records:
        for dep in deps:
            children[dep].append(name)
    ready = [(order[name], name) for name in waiting if waiting[name] == 0]
    heapq.heapify(ready)
    output = []
    while ready:
        _, name = heapq.heappop(ready)
        output.append(name)
        for child in children[name]:
            waiting[child] -= 1
            if waiting[child] == 0:
                heapq.heappush(ready, (order[child], child))
    if len(output) != len(records):
        raise ValueError("cycle")
    return output
'''


def fixture_outcome(validation, planning):
    names = ['scheduler', 'scheduler.validation', 'scheduler.planning']
    old = {name: sys.modules.get(name) for name in names}
    try:
        package = types.ModuleType('scheduler')
        package.__path__ = []
        sys.modules['scheduler'] = package
        for name, code in zip(names[1:], [validation, planning]):
            module = types.ModuleType(name)
            module.__package__ = 'scheduler'
            sys.modules[name] = module
            exec(code, module.__dict__)
        namespace = {'__name__': 'fixture_contract'}
        exec(TESTS, namespace)
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(namespace['SchedulerTests'])
        return unittest.TextTestRunner(stream=io.StringIO()).run(suite)
    finally:
        for name, original in old.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


class ProjectClientTests(unittest.TestCase):
    def test_hermes_profile_binds_docker_and_only_the_fixture_mount(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory)/'state'
            state.mkdir()
            work = Path(directory)/'workspace'
            image = 'python:3.12-slim@sha256:'+'a'*64
            profile = configure_hermes(state, work, image, 'wsp54.project_trial=control')
            config = json.loads((profile/'config.yaml').read_text())
            terminal = config['terminal']
            self.assertEqual(terminal['backend'], 'docker')
            self.assertEqual(terminal['docker_volumes'], [str(work)+':/workspace'])
            self.assertEqual(terminal['docker_image'], image)
            self.assertFalse(terminal['docker_network'])
            self.assertFalse(terminal['docker_mount_cwd_to_workspace'])
            self.assertFalse(terminal['container_persistent'])
            self.assertEqual(terminal['docker_forward_env'], [])
            self.assertEqual(config['mcp_servers'], {})
            with self.assertRaises(FileExistsError):
                configure_hermes(state, work, image, 'second')

    def test_explicit_fixture_module_is_a_real_test_run(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory)/'client.log'
            for command, expected in [
                ('cd /workspace && python3 -m unittest -v test_scheduler', True),
                ('python3 -m unittest -v test_scheduler.py', True),
                ('cd /workspace && python3 -m unittest -v 2>&1', True),
                ('python3 -m unittest -v unrelated_tests', False),
                ('echo python3 -m unittest -v test_scheduler', False),
                ('python3 -m unittest -v test_scheduler && echo passed', False),
                ('python3 -m unittest -v 2>&1 && echo passed', False),
            ]:
                event = {'type': 'tool_use', 'part': {'tool': 'bash', 'state': {
                    'status': 'completed', 'input': {'command': command}}}}
                log.write_text(json.dumps(event)+'\n')
                with self.subTest(command=command):
                    self.assertEqual(ran_tests(log, 'opencode'), expected)

    def test_project_contract_accepts_reference_and_rejects_initial_bug(self):
        correct = fixture_outcome(REFERENCE_VALIDATION, REFERENCE_PLANNING)
        wrong = fixture_outcome(INITIAL_VALIDATION, INITIAL_PLANNING)
        self.assertEqual(correct.testsRun, 7)
        self.assertTrue(correct.wasSuccessful())
        self.assertFalse(wrong.wasSuccessful())

    def test_two_hop_bridge_preserves_response_after_request_half_close(self):
        class Echo(socketserver.BaseRequestHandler):
            def handle(self):
                payload = bytearray()
                while True:
                    data = self.request.recv(65536)
                    if not data:
                        break
                    payload.extend(data)
                self.request.sendall(b'ack:' + payload)
        with tempfile.TemporaryDirectory(prefix='wsp-') as directory:
            path = str(Path(directory)/'api.sock')
            with socketserver.TCPServer(('127.0.0.1', 0), Echo) as upstream:
                thread = threading.Thread(target=upstream.handle_request, daemon=True)
                thread.start()
                with api_bridge(path, upstream.server_address):
                    with api_bridge(('127.0.0.1', 0), path, unix_listener=False) as inner:
                        with socket.create_connection(inner.server_address, timeout=5) as client:
                            payload = b'fixture-data-' * 20000
                            client.sendall(payload)
                            client.shutdown(socket.SHUT_WR)
                            response = bytearray()
                            while True:
                                data = client.recv(65536)
                                if not data:
                                    break
                                response.extend(data)
                            self.assertEqual(bytes(response), b'ack:' + payload)
                thread.join(timeout=5)
            self.assertFalse(Path(path).exists())

    def test_test_command_requires_real_tool_events_and_exact_command(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory)/'client.log'
            log.write_text(json.dumps({'type': 'text', 'text': 'python3 -m unittest -v passed'})+'\n')
            self.assertFalse(ran_tests(log, 'opencode'))
            self.assertFalse(ran_tests(log, 'hermes'))
            event = {'type': 'tool_use', 'part': {'tool': 'bash', 'state': {'status': 'completed',
                     'input': {'command': 'echo python3 -m unittest -v'}}}}
            log.write_text(json.dumps(event)+'\n')
            self.assertFalse(ran_tests(log, 'opencode'))
            event['part']['state']['input']['command'] = 'cd /workspace && python3 -m unittest -v'
            log.write_text(json.dumps(event)+'\n')
            self.assertTrue(ran_tests(log, 'opencode'))
            log.write_text(json.dumps({'type':'tool_use','name':'terminal','tool_call_id':'one',
                                      'input':{'command':'python3 -m unittest -v'}})+'\n')
            self.assertFalse(ran_tests(log, 'hermes'))
            with log.open('a') as stream:
                stream.write(json.dumps({'type':'tool_result','name':'terminal','tool_call_id':'one',
                                         'is_error':False,'output':'actual result'})+'\n')
            self.assertTrue(ran_tests(log, 'hermes'))
            # A completed failing test process still proves that tests ran.
            for result, observed in [
                ({'exit_code': 1, 'error': None, 'output': 'FAILED (failures=1)'}, True),
                ({'exit_code': -1, 'error': 'BLOCKED by policy'}, False),
            ]:
                event = {'type':'tool_use','name':'terminal','tool_call_id':'failed-test',
                         'input':{'command':'python3 -m unittest -v 2>&1'}}
                response = {'type':'tool_result','name':'terminal','tool_call_id':'failed-test',
                            'is_error':True,'output':json.dumps(result)}
                log.write_text(json.dumps(event)+'\n'+json.dumps(response)+'\n')
                with self.subTest(terminal_result=result):
                    self.assertEqual(ran_tests(log, 'hermes'), observed)


if __name__ == '__main__':
    unittest.main()
