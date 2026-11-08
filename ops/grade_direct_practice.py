#!/usr/bin/env python3
"""Grade a completed exposed practice replay in a separate retained container.

This is practice diagnosis, never independent final acceptance. All source,
grader, output, container state and bounded logs remain available for review.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from direct_weights import read_json, publish_json
from pipeline import require, sha256
from evaluate_coder import CODE_CASES, extract_code

IMAGE = 'python@sha256:4840d514c583a4d3d8cad3940e68dee6f0f6f3635c12fcd0a3d74c21a5b93a54'


def prepare(generation, request):
    require(request.get('role') == 'practice', 'Only exposed practice grading is supported')
    selected = next((c for c in request['cases'] if c['id'] == generation['id']), None)
    require(selected is not None, 'Generation is not in the frozen request')
    name = generation['id'].rsplit('-r', 1)[0]
    case = next((c for c in CODE_CASES if c[0] == name), None)
    require(case is not None and selected['messages'] == [{'role': 'user', 'content': case[1]}],
            'Practice prompt differs from the exposed frozen regression')
    require(type(generation['hit_token_limit']) is bool, 'Missing completion budget status')
    require(not generation['hit_token_limit'], 'Censored output: not a completed semantic trial')
    return extract_code(generation['text']), case[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generation', type=Path, required=True)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not os.path.lexists(args.output), 'Refusing an existing grading output')
    args.output.mkdir(parents=True, mode=0o755)
    generation, request = read_json(args.generation), read_json(args.request)
    code, tests = prepare(generation, request)
    task = args.output / 'task'; task.mkdir(mode=0o755)
    with (task / 'solution.py').open('x') as handle:
        handle.write(code)
    checker = (
        "import resource\nresource.setrlimit(resource.RLIMIT_CPU,(5,5))\n"
        "resource.setrlimit(resource.RLIMIT_FSIZE,(1048576,1048576))\n"
        "namespace={}\nexec(compile(open('/task/solution.py').read(),'/task/solution.py','exec'),namespace)\n"
        "exec(" + repr(tests) + ",namespace)\nimport os\nos._exit(73)\n")
    with (task / 'check.py').open('x') as handle:
        handle.write(checker)
    (task / 'solution.py').chmod(0o444)
    (task / 'check.py').chmod(0o444)
    name = 'wsp-direct-grade-' + args.output.name + '-' + uuid.uuid4().hex[:12]
    require(re.fullmatch(r'wsp-direct-grade-[a-z0-9-]{1,80}', name) is not None, 'Unsafe grader identity')
    command = ['docker', 'run', '--name', name, '--pull', 'never', '--network', 'none', '--read-only',
        '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges', '--pids-limit', '32',
        '--memory', '256m', '--memory-swap', '256m', '--cpus', '1', '--user', '65534:65534',
        '--tmpfs', '/tmp:rw,noexec,nosuid,size=16m', '--log-driver', 'local',
        '--log-opt', 'max-size=1m', '--log-opt', 'max-file=1', '--log-opt', 'compress=false',
        '--mount', f'type=bind,source={task.resolve()},target=/task,readonly',
        '--workdir', '/task', '-e', 'PYTHONHASHSEED=0', IMAGE,
        'python', '-P', '-s', '-S', '-B', '/task/check.py']
    require(',' not in str(task.resolve()), 'Unsafe bind path')
    timed_out = False
    try:
        result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=20)
        exit_code = result.returncode
    except subprocess.TimeoutExpired:
        timed_out, exit_code = True, None
        subprocess.run(['docker', 'stop', '--time', '1', name], capture_output=True, timeout=15)
    inspect = subprocess.run(['docker', 'inspect', name], capture_output=True, text=True, timeout=15)
    require(inspect.returncode == 0, 'No owned grading container state; grading infrastructure failed')
    state = json.loads(inspect.stdout)
    publish_json({'containers': state}, args.output / 'container-inspect.json')
    logs = subprocess.run(['docker', 'logs', '--tail', '100', name], capture_output=True, timeout=15)
    with (args.output / 'container.log').open('xb') as handle:
        handle.write(logs.stdout + logs.stderr)
    require(len(state) == 1 and state[0]['State']['Status'] == 'exited'
            and not state[0]['State'].get('Error'),
            'Grading container did not execute cleanly; infrastructure failure, not a model verdict')
    report = {'schema_version': 1, 'role': 'practice', 'id': generation['id'],
        'generation_sha256': sha256(args.generation), 'request_sha256': sha256(args.request),
        'solution_sha256': sha256(task / 'solution.py'), 'checker_sha256': sha256(task / 'check.py'),
        'evaluator_source_sha256': sha256(ROOT / 'scripts/evaluate_coder.py'),
        'grading_source_sha256': sha256(Path(__file__)), 'image': IMAGE,
        'passed': exit_code == 73 and not timed_out, 'container_exit_code': exit_code,
        'tests_completed': exit_code == 73 and not timed_out, 'timed_out': timed_out,
        'scope': 'Completed exposed practice assertion checks only; not independent paired evaluation',
        'artifacts_preserved': True, 'command': command}
    publish_json(report, args.output / 'result.json')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
