#!/usr/bin/env python3
"""Optional two-module coding/terminal acceptance in network-disabled containers.

OpenCode runs entirely inside the pinned Python container. A task-local Unix
socket relays only the loopback model API into that container. Hermes uses its
installed per-session Docker terminal/file backend with an empty profile.
Neither route mounts a Docker socket or a user's home into the agent sandbox.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import select
import shlex
import shutil
import socket
import socketserver
import subprocess
import sys
import tempfile
import threading
import time
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
INITIAL_VALIDATION = '''def normalize(jobs):
    return [(job["id"], tuple(job.get("deps", ()))) for job in jobs]
'''
INITIAL_PLANNING = '''from .validation import normalize

def plan(jobs):
    return sorted(name for name, dependencies in normalize(jobs))
'''
TESTS = '''import unittest
from scheduler.validation import normalize
from scheduler.planning import plan

class SchedulerTests(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(normalize(iter(())), [])
        self.assertEqual(plan(iter(())), [])

    def test_normalize_generator_and_duplicate_dependencies(self):
        jobs = iter([{"id": "a", "deps": iter(["b", "b"])}, {"id": "b"}])
        self.assertEqual(normalize(jobs), [("a", ("b",)), ("b", ())])

    def test_invalid_records(self):
        bad = [[{"id": "x"}, {"id": "x"}], [{}], [{"id": ""}], [{"id": 3}],
               [3], [{"id": "a", "deps": "a"}], [{"id": "a", "deps": [3]}],
               [{"id": "a", "deps": [""]}],
               [{"id": "a", "deps": None}], [{"id": "a", "deps": ["missing"]}]]
        for jobs in bad:
            with self.subTest(jobs=jobs), self.assertRaises(ValueError):
                normalize(jobs)

    def test_stable_topological_order(self):
        jobs = [{"id": "deploy", "deps": ["build"]}, {"id": "lint"},
                {"id": "build"}, {"id": "docs"}]
        self.assertEqual(plan(jobs), ["lint", "build", "deploy", "docs"])
        self.assertEqual(jobs[0]["deps"], ["build"])

    def test_generator_and_shared_dependencies(self):
        jobs = [{"id": "release", "deps": ["a", "b", "a"]},
                {"id": "a", "deps": ["base"]}, {"id": "b", "deps": ["base"]},
                {"id": "base"}]
        self.assertEqual(plan(iter(jobs)), ["base", "a", "b", "release"])

    def test_cycles(self):
        for jobs in [[{"id": "a", "deps": ["a"]}],
                     [{"id": "a", "deps": ["b"]}, {"id": "b", "deps": ["a"]}]]:
            with self.subTest(jobs=jobs), self.assertRaises(ValueError):
                plan(jobs)

    def test_plan_propagates_validation(self):
        with self.assertRaises(ValueError):
            plan([{"id": "a", "deps": ["absent"]}])

if __name__ == "__main__":
    unittest.main()
'''
PROMPT = '''Fix the scheduler package in /workspace. Read scheduler/validation.py,
scheduler/planning.py and test_scheduler.py. You may edit only the two package
modules. Preserve the public normalize(jobs) and plan(jobs) APIs.

normalize accepts an iterable of job dictionaries. Each id must be a unique,
nonempty string. deps defaults to an empty iterable; reject a bare string,
None, or dependency ids that are not nonempty strings. Deduplicate dependencies
while preserving their order. Reject unknown dependency ids. Invalid records
must raise ValueError. Return a list of (id, tuple_of_dependencies) in input order.

plan returns all ids in topological order. Whenever several jobs are available,
choose the one occurring earliest in the original input, including newly
unblocked jobs. Reject cycles with ValueError. Both APIs must accept generators
and avoid mutating the caller's dictionaries/lists.

Use file tools to edit the modules. Run `python3 -m unittest -v` in /workspace,
inspect the actual results and fix failures. Leave test_scheduler.py unchanged.
Briefly explain your final changes and report only tests you actually ran.
'''


def relay(first, second, stopping):
    """Bidirectional byte relay; half-close preserves an HTTP response after EOF."""
    readable = {first, second}
    last = time.monotonic()
    while readable and not stopping.is_set() and time.monotonic() - last < 600:
        ready, _, _ = select.select(list(readable), [], [], 0.25)
        for source in ready:
            target = second if source is first else first
            data = source.recv(65536)
            if not data:
                readable.remove(source)
                try:
                    target.shutdown(socket.SHUT_WR)
                except OSError:
                    pass
                continue
            target.sendall(data)
            last = time.monotonic()


@contextmanager
def api_bridge(address, upstream, unix_listener=True):
    stopping = threading.Event()

    class Handler(socketserver.BaseRequestHandler):
        def handle(self):
            family = socket.AF_INET if unix_listener else socket.AF_UNIX
            with socket.socket(family, socket.SOCK_STREAM) as target:
                target.settimeout(10)
                target.connect(upstream)
                target.settimeout(None)
                relay(self.request, target, stopping)

    base = socketserver.ThreadingUnixStreamServer if unix_listener else socketserver.ThreadingTCPServer

    class Server(base):
        daemon_threads = True
        allow_reuse_address = True

    server = Server(address, Handler)
    if unix_listener:
        os.chmod(address, 0o600)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        stopping.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        if unix_listener:
            Path(address).unlink(missing_ok=True)


def container_client():
    """Called inside the OpenCode container; imports only the Python stdlib."""
    os.environ.update(HOME='/tmp/client-home', XDG_CONFIG_HOME='/tmp/config',
                      XDG_DATA_HOME='/tmp/data', XDG_STATE_HOME='/tmp/state',
                      XDG_CACHE_HOME='/tmp/cache', OPENCODE_CONFIG='/workspace/opencode.json',
                      OPENCODE_CONFIG_DIR='/tmp/opencode-config', TERM='dumb')
    Path('/tmp/client-home').mkdir(exist_ok=True)
    prompt = Path('/workspace/prompt.txt').read_text()
    model = sys.argv[2]
    interfaces = sorted(os.listdir('/sys/class/net'))
    if interfaces != ['lo'] or Path('/var/run/docker.sock').exists():
        raise SystemExit('Unexpected container network or Docker socket access')
    print(json.dumps({'container_preflight': {'interfaces': interfaces, 'uid': os.getuid(),
          'docker_socket_mounted': False}}), flush=True)
    with api_bridge(('127.0.0.1', 18084), '/model-api/model.sock', unix_listener=False):
        result = subprocess.run(['/usr/local/bin/opencode', 'run', '--pure', '--format', 'json',
                                 '--model', 'wsp-local/' + model, prompt], cwd='/workspace')
    raise SystemExit(result.returncode)


def create_fixture(work):
    (work / 'scheduler').mkdir(parents=True)
    (work / 'scheduler/__init__.py').write_text('')
    (work / 'scheduler/validation.py').write_text(INITIAL_VALIDATION)
    (work / 'scheduler/planning.py').write_text(INITIAL_PLANNING)
    (work / 'test_scheduler.py').write_text(TESTS)
    (work / 'prompt.txt').write_text(PROMPT)
    (work / 'AGENTS.md').write_text('Work in /workspace only. Edit scheduler/validation.py and scheduler/planning.py; run the supplied unittest suite.\n')


def configure_hermes(state, work, image, label):
    """Bind Docker in the actual session config, not only the launch environment.

    Managed Hermes workers freeze the session config. Safe-mode ignores this
    profile and can select the default local backend despite TERMINAL_ENV.
    A fresh, explicit profile is therefore required for the contained trial.
    """
    profile = state/'hermes'
    profile.mkdir()
    config = {'terminal': {'backend': 'docker', 'cwd': '/workspace',
        'docker_image': image, 'docker_network': False,
        'docker_mount_cwd_to_workspace': False, 'docker_volumes': [str(work)+':/workspace'],
        'docker_forward_env': [], 'docker_env': {}, 'docker_run_as_host_user': True,
        'container_persistent': False, 'docker_persist_across_processes': False,
        'docker_orphan_reaper': False, 'docker_extra_args': ['--label='+label, '--pids-limit=64'],
        'container_memory': 768, 'container_cpu': 1, 'timeout': 60, 'lifetime_seconds': 120},
        'mcp_servers': {}, 'agent': {'max_turns': 16}}
    with (profile/'config.yaml').open('x') as stream:
        stream.write(json.dumps(config, indent=2)+'\n')
    return profile


def independent_check(work, image):
    sys.path.insert(0, str(ROOT / 'scripts'))
    from evaluate_coder import run_python
    modules = {name: (work / name).read_text() for name in
               ['scheduler/__init__.py', 'scheduler/validation.py', 'scheduler/planning.py']}
    setup = ('from pathlib import Path\nimport sys\n'
             'root=Path("/tmp/project-check")\nroot.mkdir()\n'
             f'files={modules!r}\n'
             'for name, content in files.items():\n'
             ' p=root/name\n p.parent.mkdir(parents=True,exist_ok=True)\n p.write_text(content)\n'
             'sys.path.insert(0,str(root))\n')
    checks = TESTS.split('if __name__ == "__main__":')[0]
    checks += ('\nsuite=unittest.defaultTestLoader.loadTestsFromTestCase(SchedulerTests)\n'
               'outcome=unittest.TextTestRunner().run(suite)\n'
               'assert outcome.testsRun == 7 and outcome.wasSuccessful()\n')
    return run_python(setup, checks, image)


def is_test_command(command):
    try:
        words = shlex.split(command)
    except (ValueError, TypeError):
        return False
    if words[:3] == ['cd', '/workspace', '&&']:
        words = words[3:]
    # Merging stderr into stdout does not change which test suite executes.
    if words[-1:] == ['2>&1']:
        words = words[:-1]
    return words in [['python3', '-m', 'unittest'], ['python3', '-m', 'unittest', '-v'],
                     ['python3', '-m', 'unittest', 'discover', '-v'],
                     ['python3', '-m', 'unittest', '-v', 'test_scheduler'],
                     ['python3', '-m', 'unittest', '-v', 'test_scheduler.py'],
                     ['python3', '-m', 'unittest', 'test_scheduler', '-v'],
                     ['python3', '-m', 'unittest', 'test_scheduler.py', '-v']]


def ran_tests(log, name):
    """Require engine-generated tool events, rather than an assistant's prose."""
    calls = set()
    for line in log.read_text().splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if name == 'opencode':
            part = row.get('part', {})
            state = part.get('state', {})
            command = state.get('input', {}).get('command', '')
            if (row.get('type') == 'tool_use' and part.get('tool') == 'bash'
                    and state.get('status') == 'completed' and is_test_command(command)):
                return True
        elif row.get('type') == 'tool_use' and row.get('name') == 'terminal':
            if is_test_command(row.get('input', {}).get('command', '')):
                calls.add(row.get('tool_call_id'))
        elif row.get('type') == 'tool_result' and row.get('name') == 'terminal':
            if row.get('tool_call_id') in calls:
                if row.get('is_error') is False:
                    return True
                # Hermes flags nonzero subprocess exits as tool errors. Count
                # their execution, while keeping blocked/failed launches out.
                try:
                    result = json.loads(row.get('output', ''))
                except (ValueError, TypeError):
                    continue
                if (isinstance(result, dict) and type(result.get('exit_code')) is int
                        and 0 <= result['exit_code'] <= 255 and result.get('error') is None):
                    return True
    return False


def observe_containers(label, area, allowed_extra, stopping, observations):
    """Inspect only this trial's uniquely labeled containers while a CLI runs."""
    seen = set()
    while not stopping.is_set():
        try:
            ids = subprocess.run(['docker', 'ps', '-aq', '--filter', 'label='+label],
                                 capture_output=True, text=True, timeout=5, check=True).stdout.split()
            for cid in ids:
                if cid in seen:
                    continue
                process = subprocess.run(['docker', 'inspect', cid], capture_output=True, text=True, timeout=5)
                if process.returncode:
                    continue  # A --rm container can disappear between these read-only calls.
                item = json.loads(process.stdout)[0]
                config = item['HostConfig']
                mounts = [{'type': m['Type'], 'source': m.get('Source', ''),
                           'destination': m['Destination']} for m in item['Mounts']]
                valid = (config['NetworkMode'] == 'none' and not config['Privileged']
                         and 'ALL' in (config['CapDrop'] or []) and 0 < config['PidsLimit'] <= 64
                         and 0 < config['Memory'] <= 768 * 1024**2
                         and item['Config']['User'].split(':')[0] == str(os.getuid()))
                def permitted(mount):
                    if mount['type'] == 'tmpfs':
                        return mount['destination'] == '/tmp'
                    source = Path(mount['source']).resolve()
                    return (mount['type'] == 'bind' and
                            (source.is_relative_to(area) or source in allowed_extra))
                valid = valid and all(permitted(m) for m in mounts)
                observations.append({'id': cid, 'network': config['NetworkMode'], 'privileged': config['Privileged'],
                    'cap_drop': config['CapDrop'], 'pids_limit': config['PidsLimit'], 'memory_bytes': config['Memory'],
                    'user': item['Config']['User'], 'mounts': mounts, 'verified': valid})
                seen.add(cid)
        except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError):
            pass  # Acceptance still requires an observed, verified container below.
        stopping.wait(0.5)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--container-client':
        container_client()
    sys.path.insert(0, str(ROOT / 'scripts'))
    sys.path.insert(0, str(ROOT))
    from coder_common import check, digest, write_json
    from evaluate_coder import Client
    from ops.client_proxy import IdentityProxy
    from ops.smoke_clients import execute
    from coder_sampling import request_sampling, client_limits
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:18084/v1')
    parser.add_argument('--model', default='wsp-qwen35-4b-coder')
    parser.add_argument('--image', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--clients', default='opencode,hermes')
    args = parser.parse_args()
    check(os.uname().nodename == '4790', 'This installed-client harness is for .54')
    args.output = args.output.resolve()
    check(not args.output.exists(), 'Refusing existing project acceptance output')
    check('@sha256:' in args.image, 'A pinned sandbox image is required')
    parsed = urlparse(args.base_url)
    check(parsed.scheme == 'http' and parsed.hostname in ('127.0.0.1', 'localhost'), 'Only a loopback model API is supported')
    names = args.clients.split(',')
    check(names and len(names) == len(set(names)) and set(names) <= {'opencode', 'hermes'}, 'Invalid clients')
    client = Client(args.base_url, args.model)
    args.output.mkdir(parents=True)
    rows = []
    for name in names:
        area = args.output / name
        area.mkdir()
        work = area / 'workspace'
        create_fixture(work)
        state = area / '.client-state'
        state.mkdir()
        tests_hash = digest(work / 'test_scheduler.py')
        protected = {p: digest(work / p) for p in ['test_scheduler.py', 'scheduler/__init__.py', 'AGENTS.md']}
        row = {'client': name, 'passed': False, 'scope': 'two-module project, file and terminal tools, network-disabled execution'}
        env = {'PATH': '/home/op/.local/bin:/usr/local/bin:/usr/bin:/bin', 'LANG': 'C.UTF-8', 'TERM': 'dumb',
               'XDG_CONFIG_HOME': str(state/'config'), 'XDG_DATA_HOME': str(state/'data'),
               'XDG_STATE_HOME': str(state/'state'), 'XDG_CACHE_HOME': str(state/'cache')}
        executable = shutil.which(name) or f'/home/op/.local/bin/{name}'
        container_name = 'wsp54-project-' + name + '-' + str(os.getpid())
        trial_label = 'wsp54.project_trial=' + str(os.getpid()) + '-' + str(time.time_ns())
        bridge = None
        proxy = None
        observations = []
        allowed_extra = [Path(__file__).resolve(), Path(executable).resolve()] if name == 'opencode' else []
        observation_stop = threading.Event()
        observer = threading.Thread(target=observe_containers,
            args=(trial_label, area, allowed_extra, observation_stop, observations), daemon=True)
        observer.start()
        try:
            proxy = IdentityProxy(args.base_url, client.identity, client.instance_id)
            cli_base_url = proxy.base_url
            proxy_address = urlparse(cli_base_url)
            version = subprocess.run([executable, '--version'], capture_output=True, text=True, timeout=30, check=True)
            row['version'] = version.stdout.splitlines()[0]
            if name == 'opencode':
                config = {'$schema': 'https://opencode.ai/config.json', 'model': 'wsp-local/'+args.model,
                    'provider': {'wsp-local': {'npm': '@ai-sdk/openai-compatible', 'name': 'WSP contained project',
                        'options': {'baseURL': 'http://127.0.0.1:18084/v1', 'apiKey': 'local-only'},
                        'models': {args.model: {'name': 'Qwen project trial', 'toolcall': True,
                                               'limit': client_limits(client.identity)}}}},
                    'agent':{'build':{'temperature':request_sampling(client.identity)['temperature']}},
                    'tools': {'task': False, 'webfetch': False, 'websearch': False, 'glob': False, 'grep': False},
                    'permission': {'*': 'deny', 'read': 'allow', 'edit': 'allow', 'write': 'allow', 'bash': 'allow', 'external_directory': 'deny'}}
                write_json(work/'opencode.json', config)
                bridge = Path(tempfile.mkdtemp(prefix='wsp-api-'))
                allowed_extra.append(bridge.resolve())
                command = ['docker', 'run', '--rm', '--name', container_name, '--label', trial_label, '--network=none', '--read-only',
                    '--cap-drop=ALL', '--security-opt=no-new-privileges', '--pids-limit=64', '--memory=768m', '--cpus=1',
                    '--user', f'{os.getuid()}:{os.getgid()}', '--tmpfs', '/tmp:rw,nosuid,nodev,size=268435456,mode=1777',
                    '--mount', f'type=bind,src={work},dst=/workspace',
                    '--mount', f'type=bind,src={bridge},dst=/model-api,readonly',
                    '--mount', f'type=bind,src={Path(executable).resolve()},dst=/usr/local/bin/opencode,readonly',
                    '--mount', f'type=bind,src={Path(__file__).resolve()},dst=/harness/project_clients.py,readonly',
                    '--workdir', '/workspace', args.image, 'python3', '/harness/project_clients.py', '--container-client', args.model]
                row['command_argv'] = command
                with api_bridge(str(bridge/'model.sock'), (proxy_address.hostname, proxy_address.port or 80)):
                    row.update(execute(command, work, env, area/'client.log', timeout=1000))
            else:
                profile = configure_hermes(state, work, args.image, trial_label)
                row['hermes_profile_sha256'] = digest(profile/'config.yaml')
                env.update(HERMES_HOME=str(profile), HERMES_IGNORE_RULES='1', TERMINAL_ENV='docker', TERMINAL_CWD='/workspace',
                    TERMINAL_DOCKER_IMAGE=args.image, TERMINAL_DOCKER_IMAGE_PINNED='1',
                    TERMINAL_CONTAINER_PERSISTENT='false', TERMINAL_DOCKER_PERSIST_ACROSS_PROCESSES='false',
                    TERMINAL_DOCKER_ORPHAN_REAPER='false', TERMINAL_DOCKER_NETWORK='false',
                    TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE='false', TERMINAL_DOCKER_RUN_AS_HOST_USER='true',
                    TERMINAL_DOCKER_FORWARD_ENV='[]', TERMINAL_DOCKER_VOLUMES=json.dumps([str(work)+':/workspace']),
                    TERMINAL_DOCKER_ENV='{}',
                    TERMINAL_DOCKER_EXTRA_ARGS=json.dumps(['--label='+trial_label, '--pids-limit=64']),
                    TERMINAL_CONTAINER_MEMORY='768', TERMINAL_CONTAINER_CPU='1', TERMINAL_TIMEOUT='60',
                    TERMINAL_LIFETIME_SECONDS='120')
                command = [executable, 'chat', '--ignore-rules', '--provider', 'custom', '--base-url', cli_base_url,
                    '--api-key', 'local-only', '--model', args.model, '--oneshot', '--format', 'stream-json',
                    '--toolsets', 'file,terminal', '--max-turns', '16', '--query-file', str(work/'prompt.txt')]
                row['command_argv'] = command
                row.update(execute(command, work, env, area/'client.log', timeout=1000))
            row['model_attestation'] = proxy.verify()
            for path, expected in protected.items():
                check((work/path).is_file() and not (work/path).is_symlink() and digest(work/path) == expected, 'Client changed protected fixture: '+path)
            for path in ['scheduler/validation.py', 'scheduler/planning.py']:
                check((work/path).is_file() and not (work/path).is_symlink(), 'Missing regular source module')
            row['independent_tests'] = independent_check(work, args.image)
            row['tests_sha256'] = tests_hash
            row['source_sha256'] = {p: digest(work/p) for p in ['scheduler/validation.py', 'scheduler/planning.py']}
            row['test_command_observed'] = ran_tests(area/'client.log', name)
            row['passed'] = (row['exit_code'] == 0 and not row['timed_out'] and row['test_command_observed']
                             and row['independent_tests']['passed'] and row['independent_tests']['tests_completed'])
        except Exception as error:
            row['error'] = f'{type(error).__name__}: {error}'
        finally:
            if proxy is not None:
                proxy.close()
            observation_stop.set()
            observer.join(timeout=12)
            row['container_isolation'] = observations
            row['passed'] = row['passed'] and bool(observations) and all(p['verified'] for p in observations)
            try:
                owned = subprocess.run(['docker', 'ps', '-aq', '--filter', 'label='+trial_label],
                                       capture_output=True, text=True, timeout=15, check=True).stdout.split()
                if owned:
                    subprocess.run(['docker', 'rm', '-f', *owned], capture_output=True, timeout=30)
                remaining = subprocess.run(['docker', 'ps', '-aq', '--filter', 'label='+trial_label],
                                           capture_output=True, text=True, timeout=15, check=True).stdout.split()
                row['remaining_test_containers'] = remaining
                row['passed'] = row['passed'] and not remaining
            except Exception as cleanup_error:
                row['cleanup_error'] = str(cleanup_error)
                row['passed'] = False
            if bridge is not None:
                shutil.rmtree(bridge)
        rows.append(row)
        print(json.dumps(row), flush=True)
    report = {'schema_version': 1, 'identity': client.identity, 'sandbox_image': args.image,
              'harness_sha256': digest(Path(__file__)), 'results': rows, 'passed': all(r['passed'] for r in rows)}
    write_json(args.output/'acceptance.json', report)
    raise SystemExit(0 if report['passed'] else 2)


if __name__ == '__main__':
    main()
