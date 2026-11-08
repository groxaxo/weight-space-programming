#!/usr/bin/env python3
"""Supervise the matched HE9 replay on .54 and its dedicated virgin client VM.

Only a loopback SSH reverse tunnel exposes the host model to the guest. Each
task receives a newly started native model process, fresh offline OpenCode
container and separate offline hidden grader. All owned processes stop before
the next task, including after a 1,200-second actor deadline.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
from coder_common import check, digest, loads, write_json
from ops.replay_he9 import TASKS, LIMITS, PRIVATE_SHA256, GRADER_IMAGE
from ops.run54_gpu_phase import restore, stop_child


def record_progress(output, plan, rows):
    # coder_common.write_json deliberately never replaces reports. Preserve an
    # immutable checkpoint after every task and publish replay.json only once.
    write_json(output/('progress-'+str(len(rows)).zfill(2)+'.json'),
               {**plan, 'status': 'running', 'results': rows})


def main():
    from method_policy import blocked_legacy_operation
    blocked_legacy_operation('Legacy VM/native GPU replay requires a new direct-edit protocol')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--conversion', type=Path, required=True)
    parser.add_argument('--profile', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--control', type=Path, required=True)
    parser.add_argument('--guest', required=True)
    parser.add_argument('--guest-output', required=True)
    args = parser.parse_args()
    check(os.uname().nodename == '4790', 'This controller is only for .54')
    output = args.output.resolve(); check(not output.exists(), 'Refusing existing HE9 round')
    output.mkdir(parents=True)
    control = args.control.resolve()
    creation = loads((control/'creation.json').read_text())
    virgin = loads((control/'virgin-guest-proof.json').read_text())
    ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', '-o', 'StrictHostKeyChecking=yes',
        '-o', 'UserKnownHostsFile='+str(control/'known_hosts'), '-i', str(control/'id_ed25519')]
    destination = 'runner@'+args.guest
    guest_root = '/home/runner/weight-space-programming'
    python = str(ROOT/'.venv-train/bin/python')
    protocol_paths = [ROOT/'ops'/n for n in ('replay_he9_vm54.py', 'replay_he9.py', 'he9_actor.py',
        'client_proxy.py', 'project_clients.py', 'run54_gguf_phase.py')]
    protocol_paths += [ROOT/'scripts'/n for n in ('serve_coder_gguf.py', 'serve_coder.py', 'coder_common.py', 'coder_sampling.py')]
    frozen = {str(p.relative_to(ROOT)): digest(p) for p in protocol_paths}
    def unchanged():
        check(all(digest(ROOT/n) == h for n, h in frozen.items()), 'HE9 protocol changed during this round')
    def remote(argv, **kwargs):
        return subprocess.run(ssh+[destination, shlex.join(argv)], check=True, **kwargs)
    rows = []
    tunnel = server = None
    current = None
    cleaned = False
    def interrupted(signum, _frame):
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGINT, interrupted)
    cleanup = ('import subprocess; ids=subprocess.check_output(["docker","ps","-aq","--filter",'
        +repr('label=wsp.he9_run='+args.guest_output.rstrip('/'))+'],text=True).split(); '
        'subprocess.run(["docker","rm","-f",*ids],check=True) if ids else None')
    try:
        # Verify the guest executes the identical current harness before any actor.
        source_check = ('from pathlib import Path\nimport hashlib,json\nroot=Path('+repr(guest_root)+')\n'
            'expected=json.loads('+repr(json.dumps(frozen))+')\n'
            'if not all(hashlib.sha256((root/n).read_bytes()).hexdigest()==h for n,h in expected.items()):\n'
            ' raise RuntimeError("Guest harness source differs")\n'
            'if hashlib.sha256(Path("/etc/machine-id").read_bytes()).hexdigest() != '
            +repr(virgin['machine_id_sha256'])+':\n raise RuntimeError("Different client VM")\n')
        remote(['/home/runner/he9-env/bin/python', '-c', source_check], timeout=30)
        plan = {'schema_version': 1, 'suite': 'experimentos-selected-he9', 'tasks': TASKS,
            'suite_sha256': PRIVATE_SHA256, 'limits': LIMITS, 'grader_image': GRADER_IMAGE,
            'host': os.uname().nodename, 'profile': args.profile, 'conversion': str(args.conversion.resolve()),
            'vm': creation, 'virgin_guest_proof': virgin, 'protocol_sha256': frozen,
            'started_at_utc': datetime.now(timezone.utc).isoformat(),
            'selection_policy': 'Strict Base+Plus solved-count gain; no loss of any base-solved task; '
                'no reduction in completed actors; all nine scored with unchanged frozen tests and matched runtime. '
                'OpenCode/Hermes development compatibility is a separate required promotion gate.',
            'promotion': 'No automatic endpoint change or adapter promotion.'}
        write_json(output/'plan.json', plan)
        with (output/'tunnel.log').open('x') as log:
            tunnel = subprocess.Popen(ssh+['-o', 'ExitOnForwardFailure=yes', '-N',
                '-R', '127.0.0.1:18084:127.0.0.1:18084', destination], stdout=log, stderr=subprocess.STDOUT)
            time.sleep(1); check(tunnel.poll() is None, 'Guest model tunnel failed')
        for task in TASKS:
            unchanged(); check(tunnel.poll() is None, 'Guest model tunnel exited')
            stem = task.replace('/', '-')
            current = output/stem
            command = [python, '-u', str(ROOT/'ops/run54_gguf_phase.py'), '--phase', 'serve',
                '--conversion', str(args.conversion.resolve()), '--profile', args.profile,
                '--output', str(current), '--max-context', str(LIMITS['context']),
                '--max-output', str(LIMITS['output']), '--cache-type-k', 'q8_0',
                '--cache-type-v', 'q8_0', '--audit-generations', '--allow-pause-bonsai']
            command.extend(['--sampling-profile',LIMITS['sampling_profile']])
            if LIMITS['thinking']:command.append('--thinking')
            with (output/(stem+'-supervisor.log')).open('x') as log:
                server = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                try:
                    deadline = time.monotonic()+360
                    while not (current/'server-ready.json').exists():
                        check(server.poll() is None, 'Native server failed before HE9 task '+task)
                        check(time.monotonic() < deadline, 'Native server readiness timed out')
                        time.sleep(1)
                    guest_area = args.guest_output.rstrip('/')+'/'+stem
                    argv = ['/home/runner/he9-env/bin/python', '-u', guest_root+'/ops/replay_he9.py',
                        '--private', '/home/runner/wsp-he9-private/original', '--setup', '/home/runner/wsp-he9/setup',
                        '--task', task, '--output', guest_area]
                    with (output/(stem+'-actor.log')).open('x') as actor_log:
                        try:
                            remote(argv, stdout=actor_log, stderr=subprocess.STDOUT, timeout=2100)
                            runner_exit = 0
                        except subprocess.CalledProcessError as error:
                            # The guest publishes an immutable failure result in
                            # finally. Preserve it and attempt every remaining
                            # task. Qualification still rejects any unscored row.
                            runner_exit = error.returncode
                    result = remote(['cat', guest_area+'/result.json'], capture_output=True, text=True, timeout=30)
                    row = loads(result.stdout)
                    check(row['task_id'] == task, 'Different guest task result')
                    row['host_phase'] = str(current)
                    row['guest_output'] = guest_area
                    row['runner_exit'] = runner_exit
                    rows.append(row)
                    record_progress(output, plan, rows)
                    print(json.dumps({k: row.get(k) for k in
                        ('task_id', 'grading', 'base_pass', 'plus_pass', 'actor_seconds')}), flush=True)
                finally:
                    preserving_error = sys.exc_info()[0] is not None
                    stop_child(server); server = None
                    restore(current)
                    if not preserving_error:
                        check((current/'phase-completed.json').exists(), 'Native task did not shut down cleanly')
                    check(not (current/'restore-bonsai-required').exists(), 'Bonsai restoration is pending')
                    current = None
        unchanged()
        stop_child(tunnel); tunnel = None
        remote(['/usr/bin/python3', '-c', cleanup], timeout=40)
        cleaned = True
        write_json(output/'replay.json', {**plan, 'status': 'complete', 'results': rows,
            'finished_at_utc': datetime.now(timezone.utc).isoformat(),
            'base_passed': sum(r.get('base_pass') is True for r in rows),
            'plus_passed': sum(r.get('plus_pass') is True for r in rows)})
    except BaseException as error:
        write_json(output/'failed.json', {'status': 'failed', 'error': f'{type(error).__name__}: {error}', 'completed_tasks': len(rows)})
        raise
    finally:
        if server is not None: stop_child(server)
        if current is not None: restore(current)
        if tunnel is not None: stop_child(tunnel)
        if not cleaned:
            try: remote(['/usr/bin/python3', '-c', cleanup], timeout=40)
            except Exception as error:
                write_json(output/'cleanup-failed.json', {'error': f'{type(error).__name__}: {error}'})


if __name__ == '__main__': main()
