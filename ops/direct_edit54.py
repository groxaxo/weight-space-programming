#!/usr/bin/env python3
"""Admit bounded CPU-only direct-edit research without preempting host jobs.

Containers, outputs and failed runs are retained. No GPU, service control,
network, cache cleanup, image pull, host home or Docker socket enters a worker.
"""
from __future__ import annotations

import argparse
import fcntl
import stat
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import shutil
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from pipeline import ValidationError, require, sha256
from direct_weights import read_json, read_bound_json, load_lock, publish_json, MODEL_ID, REVISION
import direct_evidence as evidence_contract

IMAGE = 'python@sha256:4840d514c583a4d3d8cad3940e68dee6f0f6f3635c12fcd0a3d74c21a5b93a54'
GIB = 1024 ** 3
MAX_WORKER_SECONDS = 1800
# One inode per host/user, shared by all worktrees. Never unlink this lock file.
DEFAULT_LEASE_PATH = Path('/run/user') / str(os.getuid()) / 'wsp-direct-worker.lock'


class WorkerLease:
    """Serialize admission through confirmed exit; retain intent after broker death."""
    def __init__(self, output):
        self.output, self.handle, self.active = output, None, None
        self.exit_confirmed = False

    @staticmethod
    def exited(name):
        require(type(name) is str and re.fullmatch(r'wsp-direct-[a-z0-9-]{1,80}', name),
                'Invalid retained reservation container')
        result = subprocess.run(['docker', 'inspect', name], capture_output=True, text=True, timeout=15)
        require(result.returncode == 0, 'Reservation state unknown; inspect retained worker before admission')
        values = json.loads(result.stdout)
        require(type(values) is list and len(values) == 1, 'Ambiguous retained container state')
        state = values[0].get('State', {})
        require(state.get('Status') in ('exited', 'dead') and state.get('Running') is False
                and state.get('Restarting') is False, 'Not admitted: reserved worker has not confirmed exit')
        return values

    def record(self, value):
        self.handle.seek(0)
        self.handle.write(json.dumps(value, sort_keys=True).encode())
        self.handle.truncate()
        self.handle.flush()
        os.fsync(self.handle.fileno())

    def __enter__(self):
        fd = os.open(DEFAULT_LEASE_PATH, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
        self.handle = os.fdopen(fd, 'r+b')
        try:
            info = os.fstat(fd)
            require(stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid() and info.st_nlink == 1
                    and not info.st_mode & 0o077, 'Unsafe shared worker lease file')
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ValidationError('Not admitted: another broker owns the host worker reservation') from error
            raw = self.handle.read(4097)
            require(len(raw) <= 4096, 'Oversized shared reservation state')
            previous = json.loads(raw) if raw else None
            if previous is not None:
                require(type(previous) is dict and previous.get('state') in ('active', 'released'),
                        'Unknown reservation state; manual inspection required')
                if previous['state'] == 'active':
                    # PID liveness/age never proves that a Docker worker has exited.
                    self.exited(previous.get('container'))
            publish_json({'path': str(DEFAULT_LEASE_PATH), 'previous': previous,
                          'owner_pid': os.getpid(), 'exclusive': True}, self.output / 'reservation-acquired.json')
            self.record({'state': 'released', 'owner_pid': os.getpid()})
            return self
        except BaseException:
            self.handle.close()
            self.handle = None
            raise

    def launching(self, name, memory_bytes):
        self.active = {'state': 'active', 'container': name, 'owner_pid': os.getpid(),
                       'memory_bytes': memory_bytes, 'run': str(self.output.resolve())}
        # Intent is durable before docker run: a killed broker cannot lose ownership evidence.
        self.record(self.active)
        publish_json(self.active, self.output / 'reservation-worker.json')

    def confirm_exit(self):
        values = self.exited(self.active['container'])
        self.exit_confirmed = True
        return values

    def __exit__(self, *_):
        try:
            if self.active is None or self.exit_confirmed:
                state = {'state': 'released', 'owner_pid': os.getpid(),
                         'container': self.active['container'] if self.active else None,
                         'worker_exit_confirmed': self.exit_confirmed}
                self.record(state)
                publish_json(state, self.output / 'reservation-released.json')
            # Otherwise leave active intent behind. Next broker fails closed on unknown/live state.
        finally:
            self.handle.close()



def resources(path):
    memory = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, value = line.split(':', 1)
        memory[key] = int(value.split()[0]) * 1024
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu',
                          '--format=csv,noheader'], capture_output=True, text=True, timeout=10)
    return {'captured_at_utc': datetime.now(timezone.utc).isoformat(),
            'mem_available_bytes': memory['MemAvailable'],
            'disk_free_bytes': shutil.disk_usage(path).free,
            'gpu_observation_only': gpu.stdout.strip(), 'gpu_query_error': gpu.stderr.strip(),
            'cpu_load': list(os.getloadavg())}


def admit(state, memory_bytes):
    # Reserve 4 GiB for the host in addition to the worker's entire hard limit.
    require(state['mem_available_bytes'] >= memory_bytes + 4 * GIB,
            'Not admitted: worker limit plus 4 GiB host memory reserve unavailable')
    require(state['disk_free_bytes'] >= 2 * GIB,
            'Not admitted: less than 2 GiB free; no automatic cleanup permitted')


def validate_host_request(request, mode, lock_hash):
    versions = (1, 2) if mode == 'probe' else (1,)
    require(type(request.get('schema_version')) is int and request['schema_version'] in versions,
            'Unsupported request schema for worker mode')
    require(request.get('role') == 'practice', 'This broker does not admit final/holdout evaluation')
    require(request.get('lock_sha256') == lock_hash, 'Request does not bind the exact model lock')
    require('patch' not in request, 'Candidate application requires a separately frozen causal protocol')
    if mode in ('probe', 'causal'):
        require(type(request.get('budget', {}).get('max_wall_seconds')) is int
                and request['budget']['max_wall_seconds'] <= 1700,
                'Diagnostic must leave postcheck/stop time inside the 1800-second worker maximum')
    if mode == 'tool-observe':
        from direct_tools import validate_request
        validate_request(request, {'_sha256': lock_hash})
    if mode == 'causal':
        from direct_causal import validate_request
        validate_request(request, {'_sha256': lock_hash},
                         patient_sha256=request.get('patient_sha256'),
                         source_request_sha256=request.get('source_request_sha256'))


def freeze_donors(source, destination, request):
    """Copy only bounded, hash-bound numeric/JSON evidence into a new mount."""
    from direct_causal import DONOR_NAMES
    expected = DONOR_NAMES | ({'stage1-result.json'} if request['stage'] == 2 else set())
    require(set(request['donor_files']) == expected, 'Unexpected causal donor files')
    require(source.is_dir(), 'Causal donor directory is missing')
    destination.mkdir(mode=0o700)
    for name in sorted(expected):
        incoming = source / name
        require(incoming.is_file() and not incoming.is_symlink()
                and incoming.stat().st_size <= 2 * 1024 * 1024,
                'Causal donor must be a bounded regular file: ' + name)
        require(sha256(incoming) == request['donor_files'][name], 'Donor hash differs: ' + name)
        target = destination / name
        with incoming.open('rb') as reader, target.open('xb') as writer:
            shutil.copyfileobj(reader, writer)
        require(sha256(target) == request['donor_files'][name] == sha256(incoming),
                'Donor changed during capture: ' + name)


def mounted_files(lock, baseline):
    result = []
    require(baseline.is_dir(), 'Baseline directory missing')
    for name, spec in lock['model']['files'].items():
        require(Path(name).name == name and name not in ('', '.', '..'), 'Unsafe model filename')
        target = (baseline / name).resolve(strict=True)
        require(target.is_file() and target.stat().st_size == spec['size'],
                'Missing or wrong-size model input: ' + name)
        result.append((str(target), '/model/' + name))
    return result


def container_command(name, code, deps, model_files, output, request, memory_bytes, mode,
                      patient=None, source_request=None, donors=None, episode=None):
    require(re.fullmatch(r'wsp-direct-[a-z0-9-]{1,80}', name) is not None, 'Unsafe container name')
    require(mode in ('verify', 'observe', 'probe', 'causal', 'tool-observe'), 'Unsupported worker mode')
    command = ['docker', 'run', '--name', name, '--pull', 'never', '--network', 'none',
               '--read-only', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
               '--pids-limit', '64', '--memory', str(memory_bytes), '--memory-swap', str(memory_bytes),
               '--cpus', '1', '--cpu-shares', '2', '--blkio-weight', '10',
               '--user', f'{os.getuid()}:{os.getgid()}', '--workdir', '/code',
               '--tmpfs', '/tmp:rw,noexec,nosuid,size=67108864', '--ulimit', 'fsize=67108864:67108864']
    for key, value in {'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONPATH': '/deps:/code/scripts',
                       'CUDA_VISIBLE_DEVICES': '', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                       'HF_HOME': '/tmp/hf', 'HOME': '/tmp', 'TOKENIZERS_PARALLELISM': 'false',
                       'USER': 'wsp-direct', 'LOGNAME': 'wsp-direct',
                       'TORCHINDUCTOR_CACHE_DIR': '/tmp/torchinductor', 'TRITON_CACHE_DIR': '/tmp/triton',
                       'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
                       'OPENBLAS_NUM_THREADS': '1'}.items():
        command.extend(['-e', key + '=' + value])
    mounts = [(str(code), '/code'), (str(deps), '/deps'), (str(request), '/request.json'), *model_files]
    if mode in ('probe', 'causal'):
        require(patient is not None and source_request is not None, 'Probe needs frozen patient/source request')
        mounts.extend([(str(patient), '/patient.json'), (str(source_request), '/source-request.json')])
    if mode == 'causal':
        require(donors is not None, 'Causal worker needs frozen donor artifacts')
        mounts.append((str(donors), '/donors'))
    if mode == 'tool-observe' and episode is not None:
        mounts.append((str(episode), '/episode'))
    for source, target in mounts:
        require(',' not in source, 'Comma in bind path is unsupported')
        command.extend(['--mount', f'type=bind,source={source},target={target},readonly'])
    require(',' not in str(output), 'Comma in output path is unsupported')
    command.extend(['--mount', f'type=bind,source={output},target=/out', IMAGE, 'python', '-u'])
    if mode == 'tool-observe':
        command.extend(['scripts/direct_tools.py', '--request', '/request.json', '--output', '/out/worker.json'])
        if episode is not None:
            command.extend(['--episode', '/episode'])
    elif mode == 'causal':
        command.extend(['scripts/direct_causal.py', '--request', '/request.json', '--patient', '/patient.json',
                        '--source-request', '/source-request.json', '--donors', '/donors',
                        '--output', '/out/causal', '--base', '/model'])
    elif mode == 'probe':
        command.extend(['scripts/direct_probe.py', '--request', '/request.json', '--patient', '/patient.json',
                        '--source-request', '/source-request.json', '--output', '/out/probe', '--base', '/model'])
    else:
        command.extend(['scripts/direct_forward.py', '--mode', mode,
                        '--request', '/request.json', '--output', '/out/worker.json'])
    return command


def write_new(path, value):
    publish_json(value, path)


def freeze_source(destination):
    hashes = {}
    destination.mkdir(mode=0o700)
    for folder in ('scripts', 'ops', 'configs'):
        for source in sorted((ROOT / folder).glob('*')):
            if not source.is_file() or source.suffix not in ('.py', '.json'):
                continue
            relative = source.relative_to(ROOT)
            expected = sha256(source)
            target = destination / relative
            target.parent.mkdir(exist_ok=True)
            with source.open('rb') as incoming, target.open('xb') as outgoing:
                shutil.copyfileobj(incoming, outgoing)
            require(sha256(target) == expected == sha256(source), 'Source changed during capture')
            hashes[str(relative)] = expected
    return hashes


def worker_evidence(output, mode, request, request_hash, lock_hash, *, source_hashes=None):
    path = output / (mode + '/result.json' if mode in ('probe', 'causal') else 'worker.json')
    require(path.is_file(), 'Worker exited without its bound result artifact')
    report = read_json(path)
    require(report.get('request_sha256') == request_hash
            and report.get('protocol_sha256') == request['protocol_sha256']
            and report.get('parent') == {'model_id': MODEL_ID, 'revision': REVISION, 'lock_sha256': lock_hash},
            'Worker result does not bind this request, protocol and official parent')
    require(report.get('training') is False, 'Worker cannot claim training')
    if mode in ('probe', 'causal'):
        require(report.get('role') == 'practice' and report.get('status') == 'practice_only'
                and report.get('weights_edited') is False and report.get('quality_evaluation') == 'NOT_RUN',
                'Decision diagnosis cannot become a model edit or independent quality claim')
        require(all(report.get(k) == request[k] for k in ('patient_sha256', 'source_request_sha256')),
                'Probe result patient identities differ')
        require(report.get('parameters_unchanged') is True
                and report.get('parameter_postcheck_complete') is True,
                'Probe parameter preservation is unverified')
        complete = report.get('complete') is True
        if mode == 'causal':
            require(report.get('stage') == request['stage']
                    and report.get('question') == request['question']
                    and report.get('donor_files') == request['donor_files'],
                    'Causal result changed its frozen stage or donor identities')
    else:
        require(report.get('mode') == mode and report.get('patch_applied') is False
                and report.get('quality_improvement_claimed') is False,
                'Practice worker result changed its mode or quality scope')
        complete = True
        if mode == 'observe':
            require([row['id'] for row in report.get('results', [])] == [row['id'] for row in request['cases']],
                    'Observation did not finish the declared practice replays')
    lock = load_lock()
    require(lock['_sha256'] == lock_hash, 'Host result uses a different official parent lock')
    source_hashes = evidence_contract.SOURCE_AT_IMPORT if source_hashes is None else source_hashes
    evidence_contract.verify_common(path.parent, mode, report, lock=lock, source_hashes=source_hashes)
    if mode == 'tool-observe':
        from direct_tools import observer_request, verify_tool_evidence
        evidence_contract.verify_observation(path.parent, report, observer_request(request))
        verify_tool_evidence(path.parent, report, request, lock)
    elif mode == 'observe':
        evidence_contract.verify_observation(path.parent, report, request)
    elif mode in ('probe', 'causal'):
        evidence_contract.verify_arms(path.parent, mode, report, request)
    complete = report['complete']
    return {'path': str(path), 'sha256': sha256(path), 'worker_complete': complete,
            'quality_improvement_claimed': False, 'independent_evaluation': 'NOT_RUN'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('status', 'verify', 'observe', 'probe', 'causal', 'tool-observe'))
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--deps', type=Path, help='Existing Python 3.12 site-packages, mounted read-only')
    parser.add_argument('--request', type=Path, help='Frozen numerical/practice request, never executable Python')
    parser.add_argument('--patient', type=Path, help='Completed exposed generation, required for a decision probe')
    parser.add_argument('--source-request', type=Path, help='Original frozen practice request for the patient')
    parser.add_argument('--donors', type=Path, help='Bounded completed practice evidence for causal controls')
    parser.add_argument('--episode', type=Path, help='Prior actual initial worker-output; frozen read-only for continuation')
    parser.add_argument('--lock', type=Path, default=ROOT / 'configs/qwen35-4b-direct.lock.json')
    parser.add_argument('--output', type=Path, required=True, help='New private run directory')
    args = parser.parse_args()
    require(os.uname().nodename == '4790', 'This admission profile is only for the .54 host 4790')
    require(not os.path.lexists(args.output), 'Refusing existing output path')
    args.output.mkdir(parents=True, mode=0o700)
    if args.mode == 'status':
        state = resources(args.output)
        write_new(args.output / 'resources-before.json', state)
        print(json.dumps(state))
        return
    require(args.baseline is not None and args.deps is not None and args.request is not None,
            'baseline, deps and request are required')
    request, frozen_request_hash, request_bytes = read_bound_json(args.request)
    memory_bytes = (10 * GIB if args.mode in ('observe', 'probe', 'causal', 'tool-observe') else GIB)
    try:
        with WorkerLease(args.output) as lease:
            state = resources(args.output)
            write_new(args.output / 'resources-before.json', state)
            validate_host_request(request, args.mode, sha256(args.lock))
            admit(state, memory_bytes)
            model_files = mounted_files(read_json(args.lock), args.baseline)
            name = 'wsp-direct-' + args.output.name.lower() + '-' + uuid.uuid4().hex[:12]
            frozen_source = args.output / 'source'
            source_hashes = freeze_source(frozen_source)
            # This broker deliberately supports only the checked-in 4B lock.
            require(sha256(frozen_source / 'configs/qwen35-4b-direct.lock.json') == sha256(args.lock),
                    'Custom lock differs from the frozen official 4B trust anchor')
            frozen_request = args.output / 'request.json'
            with frozen_request.open('xb') as outgoing:
                outgoing.write(request_bytes)
            require(sha256(frozen_request) == frozen_request_hash, 'Frozen request differs from admitted bytes')
            frozen_patient = frozen_source_request = None
            if args.mode in ('probe', 'causal'):
                require(args.patient is not None and args.source_request is not None,
                        'Probe requires patient and source-request files')
                for source, filename, key in ((args.patient, 'patient.json', 'patient_sha256'),
                                              (args.source_request, 'source-request.json', 'source_request_sha256')):
                    require(sha256(source) == request.get(key), 'Probe input differs from frozen request: ' + key)
                    target = args.output / filename
                    with source.open('rb') as incoming, target.open('xb') as outgoing:
                        shutil.copyfileobj(incoming, outgoing)
                    require(sha256(target) == request[key] == sha256(source), 'Probe input changed during capture')
                frozen_patient = (args.output / 'patient.json').resolve()
                frozen_source_request = (args.output / 'source-request.json').resolve()
            frozen_donors = None
            if args.mode == 'causal':
                require(args.donors is not None, 'Causal worker requires donor artifacts')
                frozen_donors = args.output / 'donors'
                freeze_donors(args.donors, frozen_donors, request)
            frozen_episode = None
            if args.mode == 'tool-observe' and request['phase'] == 'continuation':
                from direct_tools import copy_capsule
                require(args.episode is not None, 'Continuation requires the actual initial episode')
                frozen_episode = args.output / 'episode'
                copy_capsule(args.episode, frozen_episode, request)
            elif args.mode == 'tool-observe':
                require(args.episode is None, 'Initial generation cannot consume a substituted episode')
            worker_output = args.output / 'worker-output'
            worker_output.mkdir(mode=0o700)
            command = container_command(name, frozen_source.resolve(), args.deps.resolve(strict=True), model_files,
                                        worker_output.resolve(), frozen_request.resolve(), memory_bytes, args.mode,
                                        patient=frozen_patient, source_request=frozen_source_request,
                                        donors=frozen_donors.resolve() if frozen_donors is not None else None,
                                        episode=frozen_episode.resolve() if frozen_episode is not None else None)
            write_new(args.output / 'invocation.json', {'schema_version': 1, 'mode': args.mode,
                      'container': name, 'command': command, 'request_sha256': frozen_request_hash,
                      'source_sha256': source_hashes, 'model_file_mounts': model_files,
                      'training': False, 'gpu': False, 'service_control': False})
            timeout = MAX_WORKER_SECONDS if args.mode in ('observe', 'probe', 'causal', 'tool-observe') else 900
            with (args.output / 'console.log').open('x') as log:
                lease.launching(name, memory_bytes)
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                start = time.monotonic()
                def interrupted(signum, _frame):
                    raise SystemExit(128 + signum)
                signal.signal(signal.SIGTERM, interrupted)
                signal.signal(signal.SIGINT, interrupted)
                try:
                    while child.poll() is None:
                        time.sleep(5)
                        current = resources(args.output)
                        require(current['mem_available_bytes'] >= 3 * GIB,
                                'Host memory reserve fell below 3 GiB; stopping only own worker')
                        require(current['disk_free_bytes'] >= GIB,
                                'Host disk reserve fell below 1 GiB; stopping only own worker')
                        require(time.monotonic() - start < timeout,
                                'Own worker exceeded its bounded runtime')
                except BaseException:
                    # Stop only this invocation's precisely named container; retain it.
                    subprocess.run(['docker', 'stop', '--time', '10', name], capture_output=True, timeout=30)
                    child.wait(timeout=30)
                    lease.confirm_exit()
                    raise
            write_new(args.output / 'container-inspect.json', {'containers': lease.confirm_exit()})
            write_new(args.output / 'execution.json', {'exit_code': child.returncode,
                      'resources_after': resources(args.output), 'outputs_retained': True})
            require(child.returncode == 0, 'Worker failed; inspect the retained console and state')
            evidence = worker_evidence(worker_output, args.mode, request, frozen_request_hash, sha256(args.lock),
                                       source_hashes=source_hashes)
            write_new(args.output / 'worker-evidence.json', evidence)
    except BaseException as error:
        write_new(args.output / 'failure.json', {'error': type(error).__name__ + ': ' + str(error),
                                               'preserve': True, 'cleanup': False})
        raise


if __name__ == '__main__':
    main()
