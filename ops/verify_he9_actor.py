#!/usr/bin/env python3
"""Synthetic real-CLI control for the fresh HE9 actor; no Qwen or HE9 answers.

The controlled provider asks OpenCode to read/edit a trivial fixture and run an
assertion. This verifies the exact contained client, Unix bridge and attribution
path without pretending to measure model quality.
"""
import argparse
from http.server import HTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from urllib.parse import urlparse
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
from coder_common import check, digest, loads, write_json
from ops.replay_he9 import actor_events, inspect_actor, opencode_config
from ops.client_proxy import IdentityProxy
from ops.project_clients import api_bridge
from serve_coder import handler_for
from coder_sampling import QWEN35_INSTRUCT, effective_sampling


class ControlBackend:
    cfg = {'served_model_name': 'synthetic-he9-control'}
    identity = {'scope': 'Synthetic provider; no Qwen inference or benchmark score.',
                'runtime':{'sampling':QWEN35_INSTRUCT}}

    def __init__(self):
        self.calls = 0
        self.results = []

    def complete(self, request):
        sampling=effective_sampling(request,QWEN35_INSTRUCT)
        self.calls += 1
        self.results += [m for m in request['messages'] if m.get('role') == 'tool']
        calls = [('read', {'filePath': '/work/solution.py'}),
            ('edit', {'filePath': '/work/solution.py', 'oldString': 'return 0', 'newString': 'return 7'}),
            ('bash', {'command': 'python3 -c "from solution import answer; assert answer() == 7"',
                'description': 'Verify the synthetic control fixture'})]
        message = {'role': 'assistant', 'content': ''}
        if self.calls <= len(calls):
            name, arguments = calls[self.calls-1]
            check(name in {t['function']['name'] for t in request['tools']}, 'Expected OpenCode tool missing')
            message['tool_calls'] = [{'id': 'control-'+str(self.calls), 'type': 'function',
                'function': {'name': name, 'arguments': json.dumps(arguments)}}]
        else:
            message['content'] = 'Synthetic fixture edited and checked.'
        return {'id': 'synthetic-'+str(self.calls), 'model': self.cfg['served_model_name'], 'created': 1,
            'wsp':{'sampling':sampling},
            'choices': [{'index': 0, 'message': message, 'finish_reason': 'tool_calls' if 'tool_calls' in message else 'stop'}],
            'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=False)
    runtime = loads(args.runtime.read_text()); image = runtime['actor_image_id']
    check(subprocess.check_output(['docker', 'run', '--rm', '--network', 'none', image, '--version'],
        text=True).strip() == '1.19.4', 'Unexpected OpenCode version')
    checksum = subprocess.check_output(['docker', 'run', '--rm', '--network', 'none', '--entrypoint', 'sha256sum',
        image, '/usr/local/bin/opencode'], text=True).split()[0]
    check(checksum == runtime['opencode_sha256'], 'Actor executable changed')
    work = output/'work'; work.mkdir()
    (work/'solution.py').write_text('def answer():\n    return 0\n')
    write_json(work/'opencode.json', opencode_config('synthetic-he9-control'))
    backend = ControlBackend()
    server = HTTPServer(('127.0.0.1', 0), handler_for(backend))
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    upstream = f'http://127.0.0.1:{server.server_port}/v1'
    from urllib.request import urlopen
    with urlopen(upstream+'/models') as response: discovery = loads(response.read())
    name = 'wsp-he9-synthetic-'+uuid.uuid4().hex[:12]
    try:
        with IdentityProxy(upstream, discovery['wsp'], discovery['wsp_instance']) as proxy:
            parsed = urlparse(proxy.base_url)
            with tempfile.TemporaryDirectory(prefix='he9-control-') as directory:
                bridge = Path(directory)
                with api_bridge(str(bridge/'model.sock'), (parsed.hostname, parsed.port)):
                    mounted = [(work, '/work', True), (bridge, '/model-api', False),
                        (ROOT/'ops/he9_actor.py', '/harness/he9_actor.py', False),
                        (ROOT/'ops/project_clients.py', '/harness/project_clients.py', False)]
                    argv = ['docker', 'create', '--name', name, '--network', 'none', '--read-only',
                        '--user', '1000:1000', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
                        '--pids-limit', '180', '--memory', '3g', '--cpus', '2',
                        '--tmpfs', '/tmp:rw,nosuid,uid=1000,gid=1000,size=256m',
                        '--tmpfs', '/workspace:rw,nosuid,uid=1000,gid=1000,size=768m', '-w', '/work']
                    for path, dest, writable in mounted:
                        argv += ['--mount', f'type=bind,src={path},dst={dest}'+('' if writable else ',readonly')]
                    argv += ['--entrypoint', 'python3', image, '/harness/he9_actor.py', 'synthetic-he9-control']
                    subprocess.run(argv, capture_output=True, check=True, timeout=30)
                    item = loads(subprocess.check_output(['docker', 'inspect', name]))[0]
                    isolation = inspect_actor(item, {(str(p), d, rw) for p, d, rw in mounted}, image)
                    with (output/'opencode.jsonl').open('x') as log:
                        result = subprocess.run(['docker', 'start', '-a', name], stdout=log,
                            stderr=subprocess.STDOUT, timeout=90)
                    check(result.returncode == 0, 'Synthetic OpenCode control failed')
                    proof = proxy.verify()
        events = actor_events(output/'opencode.jsonl')
        check(not events['actor_errors'] and events['preflight']['empty_client_state'] is True, 'Synthetic actor state/error')
        check((work/'solution.py').read_text() == 'def answer():\n    return 7\n', 'Actual file edit did not occur')
        check({t['tool'] for t in events['tools'] if t['status'] == 'completed'} >= {'read', 'edit', 'bash'}, 'Actual tools did not complete')
        check(backend.calls == 4, 'Synthetic provider conversation differed')
        report = {'passed': True, 'scope': backend.identity['scope'], 'actual_opencode_version': '1.19.4',
            'opencode_sha256': checksum, 'actor_image': image, 'model_attestation': proof,
            'isolation': isolation, 'events': events, 'provider_calls': backend.calls}
        write_json(output/'control.json', report)
        print(json.dumps({'passed': True, 'provider_calls': backend.calls, 'scope': report['scope']}), flush=True)
    finally:
        subprocess.run(['docker', 'rm', '-f', name], capture_output=True, timeout=20)
        server.shutdown(); server.server_close(); thread.join(timeout=5)


if __name__ == '__main__': main()
