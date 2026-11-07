#!/usr/bin/env python3
"""Replay one original Experimentos HE9 task in a fresh offline OpenCode actor.

Run on the clean client VM. The host controller restarts its own native serving
process between tasks, so a deadline cannot leak generation into the next task.
Private archive files are required separately; none are included in this repo.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import tempfile
import time
from urllib.parse import urlparse
from urllib.request import urlopen
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
from coder_common import check, digest, fingerprint, loads, write_json
from ops.client_proxy import IdentityProxy
from ops.project_clients import api_bridge
from coder_sampling import QWEN35_INSTRUCT

TASKS = ['HumanEval/'+str(n) for n in (1, 6, 32, 39, 76, 99, 101, 132, 141)]
PRIVATE_SHA256 = {
    'grade_worker.py': 'fedd6addd2e03df62f00185ac0f0beb4123f34bc99e731b753076a51bf5392a5',
    'grader-cases-private.jsonl': 'a9da6444d2142606e0d7ca7f81971f65f595bbb55370e322fc8480c578b18b4d',
    'suite-selected.json': 'e1d7d6e8b79b82ae9649a05db506d2ce40f89401de0067429b5faa6956ea3add',
}
GRADER_IMAGE = 'python:3.11.17-slim@sha256:b7a9e5f6a93d6d4a31f129bbd4bdee69efccb6ded0fb230d78aafaf37af98377'
LIMITS = {'context': 262144, 'input': 253952, 'output': 8192, 'actor_seconds': 1200,
          **QWEN35_INSTRUCT,'sampling_profile':'qwen35-instruct',
          'thinking': False, 'cache_type_k': 'q8_0', 'cache_type_v': 'q8_0'}


def public_tasks(private):
    for name, expected in PRIVATE_SHA256.items():
        check(digest(private/name) == expected, 'Frozen private archive changed: '+name)
    selected = loads((private/'suite-selected.json').read_text())
    # Project only these public fields; no canonical code, test functions or
    # hidden input values may enter actor configuration or training material.
    cases = [loads(line) for line in (private/'grader-cases-private.jsonl').read_text().splitlines()]
    check(len(cases) == 9 and {r['task_id'] for r in cases} == set(TASKS), 'Different HE9 task set')
    output = {}
    for case in cases:
        spec = next(s for s in selected if s['task_id'] == case['task_id'])
        check(hashlib.sha256(case['prompt'].encode()).hexdigest() == spec['problem_sha256'], 'Changed public starter')
        output[case['task_id']] = {k: case[k] for k in ('task_id', 'prompt', 'entry_point')}
    return output


def opencode_config(model):
    return {'$schema': 'https://opencode.ai/config.json', 'model': 'local/'+model,
        'small_model': 'local/'+model, 'enabled_providers': ['local'], 'share': 'disabled',
        'snapshot': False, 'plugin': [], 'mcp': {}, 'instructions': [],
        'permission': {'*': 'allow'}, 'compaction': {'auto': True, 'reserved': 4096},
        'agent': {'title': {'disable': True}, 'summary': {'disable': True},
            'build': {'temperature': LIMITS['temperature'], 'tools': {'*': False, 'bash': True, 'read': True,
                'write': True, 'edit': True, 'glob': True, 'grep': True, 'list': True,
                'apply_patch': True, 'todowrite': True, 'todoread': True}}},
        'provider': {'local': {'npm': '@ai-sdk/openai-compatible', 'name': 'Qwen frozen HE9 replay',
            'options': {'baseURL': 'http://127.0.0.1:18084/v1', 'apiKey': 'local-only',
                'timeout': 1800000, 'headerTimeout': 1800000, 'chunkTimeout': 1800000},
            'models': {model: {'name': model, 'reasoning': LIMITS['thinking'], 'toolcall': True,
                'limit': {k: LIMITS[k] for k in ('context', 'input', 'output')},
                'options': {'reasoningEffort': 'xhigh'} if LIMITS['thinking'] else {}}}}}}


def inspect_actor(item, expected_mounts, image):
    config = item['HostConfig']
    mounts = [{'type': m['Type'], 'source': m.get('Source', ''), 'destination': m['Destination'],
               'rw': m.get('RW', False)} for m in item['Mounts']]
    observed = {(m['source'], m['destination'], m['rw']) for m in mounts if m['type'] == 'bind'}
    valid = (item['Image'] == image and config['NetworkMode'] == 'none'
        and not config['Privileged'] and config['ReadonlyRootfs']
        and 'ALL' in (config['CapDrop'] or []) and config['PidsLimit'] == 180
        and config['Memory'] == 3*1024**3 and item['Config']['User'] == '1000:1000'
        and observed == expected_mounts and all(m['type'] == 'bind' or
            (m['type'] == 'tmpfs' and m['destination'] in ('/tmp', '/workspace')) for m in mounts)
        and set(config.get('Tmpfs') or {}) == {'/tmp', '/workspace'}
        and 'no-new-privileges' in config['SecurityOpt'])
    check(valid, 'Actor isolation differs from the declared protocol')
    return {'id': item['Id'], 'image': item['Image'], 'network': config['NetworkMode'],
        'mounts': mounts, 'user': item['Config']['User'], 'verified': True,
        'read_only_root': True, 'memory_bytes': config['Memory'], 'pids_limit': config['PidsLimit']}


def actor_events(log):
    errors, preflight, tools, completed = [], None, [], False
    for line in log.read_text(errors='replace').splitlines():
        try:
            row = loads(line)
        except ValueError:
            continue
        if row.get('container_preflight'):
            preflight = row['container_preflight']
        if row.get('type') == 'error':
            errors.append(row)
        if row.get('type') == 'tool_use':
            part = row.get('part', {})
            tools.append({'tool': part.get('tool'), 'status': part.get('state', {}).get('status')})
        if row.get('type') == 'step_finish' and row.get('part', {}).get('reason') == 'stop':
            completed = True
    return {'actor_errors': errors, 'preflight': preflight, 'tools': tools, 'final_step_observed': completed}


def grade(task, candidate, private, area):
    check(candidate.is_file() and not candidate.is_symlink(), 'Missing regular candidate')
    name = 'wsp-he9-grade-'+uuid.uuid4().hex[:12]
    argv = ['docker', 'run', '--rm', '--name', name, '--label', 'wsp.he9_run='+str(area.parent), '--network', 'none', '--read-only',
        '--user', '1000:1000', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
        '--pids-limit', '90', '--memory', '1g', '--cpus', '2',
        '--tmpfs', '/tmp:rw,nosuid,uid=1000,gid=1000,size=128m', '-e', 'PYTHONDONTWRITEBYTECODE=1']
    for source, dest in [(candidate, '/candidate.py'), (private/'grade_worker.py', '/private/grade_worker.py'),
                          (private/'grader-cases-private.jsonl', '/private/cases.jsonl')]:
        argv += ['--mount', f'type=bind,src={source},dst={dest},readonly']
    argv += ['--entrypoint', 'python3', GRADER_IMAGE, '/private/grade_worker.py', task,
             '/candidate.py', '/private/cases.jsonl']
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=320)
        (area/'grader.log').write_text(result.stdout+result.stderr)
        check(result.returncode == 0, 'Frozen grader process failed')
        verdict = loads(result.stdout.splitlines()[-1])
        check(verdict['task_id'] == task and verdict['grader'] == 'reconstructed-base-plus-v1', 'Different grader result')
        return verdict
    finally:
        subprocess.run(['docker', 'rm', '-f', name], capture_output=True, timeout=20)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--private', type=Path, required=True)
    parser.add_argument('--setup', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--task', choices=TASKS, required=True)
    parser.add_argument('--base-url', default='http://127.0.0.1:18084/v1')
    parser.add_argument('--model', default='wsp-qwen35-4b-coder')
    args = parser.parse_args()
    os.umask(0o077)
    def interrupted(signum, _frame):
        raise SystemExit(128+signum)
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, interrupted)
    check(os.getuid() == 1000, 'HE9 guest runner must be UID 1000')
    args.private, args.setup, area = args.private.resolve(), args.setup.resolve(), args.output.resolve()
    check(not area.exists(), 'Refusing an existing HE9 task output')
    tasks = public_tasks(args.private)
    setup = loads((args.setup/'runtime.json').read_text())
    reference = loads((args.setup/'reference-selftest.json').read_text())
    check(reference['tests_sha256'] == PRIVATE_SHA256 and len(reference['results']) == 9
        and all(r['base_pass'] and r['plus_pass'] for r in reference['results'])
        and reference['total_plus_cases'] == 5546, 'All nine reference controls must pass first')
    check(setup['grader_image'] == GRADER_IMAGE and setup['opencode_version'] == '1.19.4', 'Different pinned clients')
    image = setup['actor_image_id']
    check(image.startswith('sha256:') and len(image) == 71, 'Actor image must be pinned by content')
    with urlopen(args.base_url+'/models', timeout=15) as response:
        discovery = loads(response.read())
    identity, instance = discovery['wsp'], discovery['wsp_instance']
    check(identity['limits'] == {'context': LIMITS['context'], 'output': LIMITS['output']}, 'Serving limits differ')
    launch = identity['runtime']['launch_options']
    check(all(launch[k] == LIMITS[k] for k in ('context', 'output', 'thinking', 'cache_type_k', 'cache_type_v')),
          'Serving context, thinking or KV cache differs')
    check(identity['runtime'].get('sampling')==QWEN35_INSTRUCT,'Serving sampling profile differs')
    area.mkdir(parents=True)
    proof_area=area/'setup';proof_area.mkdir()
    for name in ('runtime.json','reference-selftest.json','project-source.json'):
        shutil.copyfile(args.setup/name,proof_area/name)
    work = area/'work'; work.mkdir()
    source = work/'solution.py'; source.write_text(tasks[args.task]['prompt'])
    starter_hash = digest(source)
    write_json(work/'opencode.json', opencode_config(args.model))
    config_hash = digest(work/'opencode.json')
    row = {'task_id': args.task, 'started_at_utc': datetime.now(timezone.utc).isoformat(),
        'identity': identity, 'instance_id': instance, 'limits': LIMITS,
        'suite_sha256': PRIVATE_SHA256, 'grader_image': GRADER_IMAGE,
        'actor_image': image, 'opencode_version': setup['opencode_version'],
        'opencode_sha256': setup['opencode_sha256'], 'starter_sha256': starter_hash,
        'harness_sha256': {str(p.relative_to(ROOT)): digest(p) for p in
            [Path(__file__), ROOT/'ops/he9_actor.py', ROOT/'ops/project_clients.py', ROOT/'ops/client_proxy.py']},
        'reference_selftest_sha256': digest(args.setup/'reference-selftest.json'),
        'project_source_sha256': digest(args.setup/'project-source.json')}
    name = 'wsp-he9-actor-'+uuid.uuid4().hex[:12]
    proxy = None
    process = None
    try:
        with ExitStack() as stack:
            proxy = stack.enter_context(IdentityProxy(args.base_url, identity, instance))
            parsed = urlparse(proxy.base_url)
            bridge = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix='he9-api-')))
            stack.enter_context(api_bridge(str(bridge/'model.sock'), (parsed.hostname, parsed.port)))
            mounted = [(work, '/work', True), (bridge, '/model-api', False),
                (ROOT/'ops/he9_actor.py', '/harness/he9_actor.py', False),
                (ROOT/'ops/project_clients.py', '/harness/project_clients.py', False)]
            argv = ['docker', 'create', '--name', name, '--label', 'wsp.he9_run='+str(area.parent), '--network', 'none', '--read-only',
                '--user', '1000:1000', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
                '--pids-limit', '180', '--memory', '3g', '--cpus', '2',
                '--tmpfs', '/tmp:rw,nosuid,uid=1000,gid=1000,size=256m',
                '--tmpfs', '/workspace:rw,nosuid,uid=1000,gid=1000,size=768m', '-w', '/work']
            for path, dest, writable in mounted:
                argv += ['--mount', f'type=bind,src={path},dst={dest}'+('' if writable else ',readonly')]
            argv += ['--entrypoint', 'python3', image, '/harness/he9_actor.py', args.model]
            subprocess.run(argv, capture_output=True, text=True, check=True, timeout=30)
            item = loads(subprocess.check_output(['docker', 'inspect', name]))[0]
            row['isolation'] = inspect_actor(item, {(str(p), d, rw) for p, d, rw in mounted}, image)
            started = time.monotonic()
            with (area/'opencode.jsonl').open('x') as log:
                process = subprocess.Popen(['docker', 'start', '-a', name], stdout=log, stderr=subprocess.STDOUT)
                try:
                    process.wait(timeout=LIMITS['actor_seconds'])
                    row.update(actor_exit=process.returncode, actor_status='finished')
                except subprocess.TimeoutExpired:
                    subprocess.run(['docker', 'rm', '-f', name], capture_output=True, timeout=20)
                    process.wait(timeout=30)
                    row.update(actor_exit=124, actor_status='budget_limit')
            row['actor_seconds'] = round(time.monotonic()-started, 3)
            row.update(actor_events(area/'opencode.jsonl'))
            # Allow the final already-issued request to drain before attribution.
            # The host always terminates this task's native server afterwards.
            # No attempt is ever started on a server left busy by its predecessor.
            proxy.drain()
            row['model_attestation'] = proxy.verify()
        check(digest(work/'opencode.json') == config_hash, 'Actor modified its provider configuration')
        check(source.is_file() and not source.is_symlink() and source.stat().st_size <= 1024**2,
              'Invalid saved candidate file')
        candidate = area/'candidate.py'; candidate.write_bytes(source.read_bytes())
        row.update(candidate_sha256=digest(candidate), modified=digest(candidate) != starter_hash)
        if row['actor_status'] == 'budget_limit' or (row['actor_exit'] == 0 and not row['actor_errors']):
            row.update(grade(args.task, candidate, args.private, area))
            row['grading'] = 'scored_artifact_at_budget_limit' if row['actor_status'] == 'budget_limit' else 'scored'
        else:
            row['grading'] = 'infrastructure_or_api_error_unscored'
    except BaseException as error:
        row.update(grading='infrastructure_or_api_error_unscored', infrastructure_error=f'{type(error).__name__}: {error}')
        raise
    finally:
        if proxy is not None:
            row['proxy_diagnostics'] = {'errors': list(proxy.errors), 'validated_requests': len(proxy.events),
                'delivery_disconnects': proxy.delivery_disconnects}
        subprocess.run(['docker', 'rm', '-f', name], capture_output=True, timeout=20)
        if process is not None and process.poll() is None:
            process.terminate(); process.wait(timeout=20)
        row['remaining_actor_containers'] = subprocess.check_output(
            ['docker', 'ps', '-aq', '--filter', 'name=^/'+name+'$'], text=True).split()
        if (source.is_file() and not source.is_symlink() and source.stat().st_size <= 1024**2
                and not (area/'candidate.py').exists()):
            (area/'candidate.py').write_bytes(source.read_bytes())
            row.update(candidate_sha256=digest(area/'candidate.py'), modified=digest(source) != starter_hash)
        row['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
        write_json(area/'result.json', row)
        print(json.dumps({k: row.get(k) for k in ('task_id', 'grading', 'base_pass', 'plus_pass', 'actor_seconds', 'infrastructure_error')}), flush=True)


if __name__ == '__main__':
    main()
