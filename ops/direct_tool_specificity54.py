#!/usr/bin/env python3
"""Admit the fixed five exposed semantic controls; no weights or final cases.

Reuse the established lease, resource admission and immutable Docker profile.
Historical input capture is a whitelist of 32 bounded data files, never a home,
service socket, old executable source tree or report-nominated absolute path.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from ops import direct_edit54 as host
import direct_tool_specificity as s
from direct_weights import load_lock, publish_json, read_bound_json, read_json, verify_baseline
from pipeline import require, sha256

DRIVER = 'ops/direct_tool_specificity54.py'
QUALIFICATION = 'docs/experiments/direct-tool-finalization-cycle2-20261108/QUALIFICATION-LEDGER.json'


def capture_file(source, target, expected, maximum=4 * 1024 * 1024):
    require(source.is_file() and not source.is_symlink() and source.stat().st_size <= maximum,
            'Input must be a bounded regular file: ' + str(source))
    require(sha256(source) == expected, 'Input differs before capture')
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open('rb') as incoming, target.open('xb') as outgoing:
        shutil.copyfileobj(incoming, outgoing)
    require(sha256(target) == expected == sha256(source), 'Input changed during capture')


def capture_stage1(source, target, stage1):
    target.mkdir(mode=0o700)
    require(len(stage1['manifest']['files']) == 32, 'All four historical records required')
    for relative, expected in stage1['manifest']['files'].items():
        path = Path(relative)
        require(not path.is_absolute() and '..' not in path.parts and len(path.parts) == 3
                and path.parts[1] == 'worker-output', 'Unsafe historical data path')
        capture_file(source / path, target / path, expected)
    require(s.load_stage1(target, load_lock())['manifest_sha256'] == stage1['manifest_sha256'],
            'Frozen historical bundle differs')


def freeze_source(destination):
    hashes = host.freeze_source(destination)
    capture_file(ROOT / QUALIFICATION, destination / QUALIFICATION, s.QUALIFICATION_SHA256)
    hashes[QUALIFICATION] = s.QUALIFICATION_SHA256
    require(hashes[DRIVER] == sha256(ROOT / DRIVER)
            and hashes['scripts/direct_tool_specificity.py'] == s.SOURCE_SHA256,
            'Specificity source changed since process start')
    return hashes


def command(name, code, deps, model_files, output, request, memory, stage1, protocol,
            protocol_sha256, conditions=None, *, compile_only=False):
    # Preserve all established hardening flags. Only replace the fixed entrypoint
    # and append these narrow read-only mounts before the immutable image.
    base = host.container_command(name, code, deps, model_files, output, request, memory, 'observe')
    index = base.index(host.IMAGE)
    extra = []
    for source, target in ((stage1, '/stage1'), (protocol, '/specificity-protocol.json'),
                           *(([(conditions, '/conditions.json')]) if conditions is not None else [])):
        require(',' not in str(source), 'Comma in input bind path')
        extra += ['--mount', f'type=bind,source={source},target={target},readonly']
    tail = (['ops/direct_tool_specificity54.py', 'inside-compile'] if compile_only else
            ['scripts/direct_tool_specificity.py', '--request', '/request.json',
             '--conditions', '/conditions.json', '--output', '/out/worker.json'])
    return base[:index] + extra + [host.IMAGE, 'python', '-u'] + tail + [
        '--stage1', '/stage1', '--protocol', '/specificity-protocol.json',
        '--protocol-sha256', protocol_sha256]


def inside_compile(args):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only compiler required')
    lock = load_lock()
    s.load_stage1(args.stage1, lock)  # Revalidate before importing/loading tokenizer.
    s.load_protocol(args.protocol, args.protocol_sha256, lock)
    verified = verify_baseline(lock, Path('/model'))
    publish_json(verified, Path('/out/baseline-verification.json'))
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('/model', local_files_only=True, trust_remote_code=False)
    conditions = s.compile_conditions(tokenizer, args.stage1, args.protocol, args.protocol_sha256, lock)
    publish_json(conditions, Path('/out/conditions.json'))
    publish_json({'status': 'specificity_compiler_passed', 'generation_invocations': 0,
                  'model_loaded': False, 'training': False, 'weights_edited': False,
                  'protocol_sha256': args.protocol_sha256,
                  'conditions_sha256': sha256(Path('/out/conditions.json')),
                  'stage1_manifest_sha256': conditions['stage1_manifest_sha256'],
                  'driver_sha256': sha256(ROOT / DRIVER), 'specificity_sha256': s.SOURCE_SHA256,
                  'source_sha256': s.evidence.SOURCE_AT_IMPORT}, Path('/out/compiler.json'))


def run_one(args, area, stage1, protocol, protocol_sha256, request, *, compile_only=False):
    require(not os.path.lexists(area), 'Refusing existing worker evidence')
    area.mkdir(parents=True, mode=0o700)
    started = time.monotonic()
    memory = (2 if compile_only else 10) * host.GIB
    timeout = 900 if compile_only else 1800
    try:
        with host.WorkerLease(area) as lease:
            state = host.resources(area); publish_json(state, area / 'resources-before.json')
            host.admit(state, memory)
            code = area / 'source'; hashes = freeze_source(code)
            frozen_stage1 = area / 'stage1'; capture_stage1(args.stage1, frozen_stage1, stage1)
            frozen_protocol = area / 'protocol.json'
            capture_file(args.protocol, frozen_protocol, protocol_sha256)
            frozen_conditions = None
            if not compile_only:
                frozen_conditions = area / 'conditions.json'
                capture_file(args.conditions, frozen_conditions, request['conditions_sha256'])
            request_path = area / 'request.json'; publish_json(request, request_path)
            output = area / 'worker-output'; output.mkdir(mode=0o700)
            name = 'wsp-direct-specificity-' + ('compile' if compile_only else request['control'].lower()) + '-' + uuid.uuid4().hex[:12]
            argv = command(name, code.resolve(), args.deps.resolve(strict=True),
                           host.mounted_files(load_lock(), args.baseline), output.resolve(), request_path.resolve(),
                           memory, frozen_stage1.resolve(), frozen_protocol.resolve(), protocol_sha256,
                           frozen_conditions.resolve() if frozen_conditions else None, compile_only=compile_only)
            publish_json({'container': name, 'command': argv, 'source_sha256': hashes,
                          'protocol_sha256': protocol_sha256, 'request_sha256': sha256(request_path),
                          'stage1_manifest_sha256': stage1['manifest_sha256'], 'training': False,
                          'gpu': False, 'service_control': False,
                          'generation_invocations_max': 0 if compile_only else 1}, area / 'invocation.json')
            lease.launching(name, memory)
            with (area / 'console.log').open('x') as log:
                child = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT)
                def interrupted(signum, _frame): raise InterruptedError('Own worker interrupted: ' + str(signum))
                old = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGINT)}
                try:
                    while child.poll() is None:
                        time.sleep(5)
                        current = host.resources(area)
                        require(current['mem_available_bytes'] >= 3 * host.GIB
                                and current['disk_free_bytes'] >= host.GIB, 'Host reserve lost')
                        require(time.monotonic() - started < timeout, 'Own bounded worker timeout')
                except BaseException:
                    subprocess.run(['docker', 'stop', '--time', '10', name], capture_output=True, timeout=30)
                    child.wait(timeout=30); lease.confirm_exit(); raise
                finally:
                    for sig, handler in old.items(): signal.signal(sig, handler)
            publish_json({'containers': lease.confirm_exit()}, area / 'container-inspect.json')
            require(child.returncode == 0, 'Worker failed; retain console and partial artifacts')
            if compile_only:
                result = read_json(output / 'compiler.json')
                require(result['status'] == 'specificity_compiler_passed' and result['generation_invocations'] == 0
                        and result['conditions_sha256'] == sha256(output / 'conditions.json')
                        and result['protocol_sha256'] == protocol_sha256
                        and result['driver_sha256'] == hashes[DRIVER]
                        and result['specificity_sha256'] == s.SOURCE_SHA256
                        and result['source_sha256'] == s.evidence.SOURCE_AT_IMPORT, 'Compiler receipt differs')
            else:
                report = read_json(output / 'worker.json')
                analysis = s.verify_specificity_evidence(output, report, request, protocol, protocol_sha256,
                    read_json(args.conditions), s.load_stage1(frozen_stage1, load_lock()), load_lock())
                result = {'control': request['control'], 'worker': report, 'analysis': analysis}
            resources_after = host.resources(area)
        # Include the final resource query and confirmed lease release. The
        # cycle caller additionally measures through this receipt's own fsync.
        publish_json({'exit_code': child.returncode, 'complete': True,
                      'charged_worker_seconds': math.ceil(time.monotonic() - started + 1),
                      'resources_after': resources_after, 'preserved': True}, area / 'execution.json')
        return result
    except BaseException as error:
        publish_json({'complete': False, 'charged_worker_seconds': math.ceil(time.monotonic() - started + 1),
                      'error': type(error).__name__ + ': ' + str(error), 'preserved': True,
                      'weights_edited': False}, area / 'failure.json')
        raise


def validate_budget(ledger, protocol_sha256, conditions_sha256):
    require(ledger.get('kind') == 'specificity-source-qualification'
            and ledger.get('all_costs_reconciled') is True
            and ledger.get('generation_invocations_used') == 8
            and type(ledger['generation_invocations_used']) is int
            and ledger.get('protocol_sha256') == protocol_sha256
            and ledger.get('conditions_sha256') == conditions_sha256
            and ledger.get('source_sha256') == s.evidence.SOURCE_AT_IMPORT
            and ledger.get('specificity_sha256') == s.SOURCE_SHA256
            and ledger.get('driver_sha256') == sha256(ROOT / DRIVER), 'Unqualified or uncharged source')
    seconds = ledger.get('total_worker_seconds')
    require(type(seconds) in (int, float) and math.isfinite(seconds) and seconds >= 3081.713629
            and seconds + 5 * 1800 <= 21600, 'Whole five-control tranche must fit the cumulative budget')
    return seconds


def unfinished_controls(completed, attempted):
    require(type(completed) is int and type(attempted) is int and 0 <= completed <= attempted <= 5,
            'Invalid completed/attempted control counts')
    return {'attempted_unverified_controls': list(s.CONTROLS[completed:attempted]),
            'not_run_controls': list(s.CONTROLS[attempted:])}


def run_cycle(args, stage1, protocol):
    conditions, conditions_hash, _ = read_bound_json(args.conditions)
    ledger, ledger_hash, ledger_raw = read_bound_json(args.ledger)
    seconds = validate_budget(ledger, args.protocol_sha256, conditions_hash)
    require(not os.path.lexists(args.output), 'Refusing existing cycle evidence')
    args.output.mkdir(mode=0o700, parents=True)
    with (args.output / 'qualification-ledger.json').open('xb') as handle: handle.write(ledger_raw)
    requests = []
    for control in s.CONTROLS:
        request = {'schema_version': 1, 'role': 'practice', 'question': s.QUESTION, 'control': control,
                   'lock_sha256': load_lock()['_sha256'], 'protocol_sha256': args.protocol_sha256,
                   'conditions_sha256': conditions_hash, 'stage1_manifest_sha256': stage1['manifest_sha256'],
                   'generation': dict(s.PROFILE)}
        s.validate_request(request, protocol, args.protocol_sha256, load_lock())
        s.validate_conditions(conditions, request, protocol, stage1)
        publish_json(request, args.output / (control + '-request.json')); requests.append(request)
    frozen = {DRIVER: sha256(ROOT / DRIVER), 'scripts/direct_tool_specificity.py': s.SOURCE_SHA256,
              **s.evidence.SOURCE_AT_IMPORT}
    claim_path = host.DEFAULT_LEASE_PATH.parent / ('wsp-direct-specificity-' + stage1['manifest_sha256'] + '.json')
    claim = {'stage1_manifest_sha256': stage1['manifest_sha256'], 'protocol_sha256': args.protocol_sha256,
             'conditions_sha256': conditions_hash, 'ledger_sha256': ledger_hash,
             'output': str(args.output.resolve()), 'source_sha256': frozen,
             'time_utc': datetime.now(timezone.utc).isoformat(), 'cleanup': False, 'generations_max': 5}
    fd = os.open(claim_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'w') as handle:
        json.dump(claim, handle, sort_keys=True); handle.flush(); os.fsync(handle.fileno())
    publish_json({**claim, 'claim_path': str(claim_path)}, args.output / 'cycle-start.json')
    records, costs, attempted = [], [], 0
    try:
        for request in requests:
            require(all(sha256(ROOT / path) == value for path, value in frozen.items()), 'Frozen source changed')
            require(sha256(args.protocol) == args.protocol_sha256 and sha256(args.conditions) == conditions_hash,
                    'Frozen protocol/conditions changed')
            require(seconds + 1800 <= 21600 and 8 + attempted + 1 <= 13, 'Cumulative budget exhausted')
            area = args.output / ('control-' + request['control'])
            attempted += 1
            print(json.dumps({'launch': request['control'], 'time_utc': datetime.now(timezone.utc).isoformat()}), flush=True)
            attempt_started = time.monotonic()
            try:
                record = run_one(args, area, stage1, protocol, args.protocol_sha256, request)
                records.append(record)
            finally:
                receipt = read_json(area / ('execution.json' if (area / 'execution.json').is_file() else 'failure.json'))
                # Measure after run_one returned/raised: resource queries,
                # lease release and execution/failure receipt writes are charged.
                charge = max(receipt['charged_worker_seconds'], math.ceil(time.monotonic() - attempt_started + 1))
                seconds += charge
                costs.append({'control': request['control'], 'receipt': str(area / ('execution.json' if receipt['complete'] else 'failure.json')),
                              'receipt_worker_seconds': receipt['charged_worker_seconds'],
                              'charged_worker_seconds': charge})
            analysis = record['analysis']
            publish_json({'records': records, 'costs': costs, 'cumulative_worker_seconds': seconds,
                          'cumulative_generation_invocations': 8 + attempted}, args.output / ('progress-' + request['control'] + '.json'))
            print(json.dumps({'control': request['control'], 'classification': analysis['classification'],
                              'passed': analysis['passed'], 'cumulative_worker_seconds': seconds}), flush=True)
            if analysis['classification'].startswith('STOP'):
                # Keep the verified parser/censor verdict; this is not an
                # execution exception, and remaining rows stay explicitly unrun.
                break
        result = s.specificity_gate(records)
    except BaseException as error:
        result = {**s.specificity_gate(records), 'status': 'STOP_infrastructure',
                  'error': type(error).__name__ + ': ' + str(error), 'localization_admitted': False}
        raise
    finally:
        if 'result' not in locals(): result = {'status': 'STOP_infrastructure', 'localization_admitted': False}
        publish_json({**result, 'records': records, 'costs': costs, 'source_sha256': frozen,
                      **unfinished_controls(len(records), attempted),
                      'cumulative_worker_seconds': seconds, 'cumulative_generation_invocations': 8 + attempted,
                      'generation_cap': 13, 'worker_seconds_cap': 21600, 'training': False,
                      'weights_edited': False, 'quality_evaluation': 'NOT_RUN', 'artifacts_preserved': True},
                     args.output / 'specificity-gate.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'compile', 'run', 'inside-compile'))
    for key in ('stage1', 'protocol', 'conditions', 'baseline', 'deps', 'output', 'ledger'):
        parser.add_argument('--' + key, type=Path)
    parser.add_argument('--protocol-sha256')
    args = parser.parse_args()
    if args.mode == 'inside-compile': inside_compile(args); return
    require(os.uname().nodename == '4790', 'Only the .54 admission profile is supported')
    stage1 = s.load_stage1(args.stage1, load_lock())
    if args.mode == 'prepare':
        publish_json(s.protocol_document(stage1, load_lock()), args.protocol)
        print(json.dumps({'protocol_sha256': sha256(args.protocol), 'stage1_manifest_sha256': stage1['manifest_sha256']})); return
    protocol, _ = s.load_protocol(args.protocol, args.protocol_sha256, load_lock())
    if args.mode == 'compile':
        run_one(args, args.output, stage1, protocol, args.protocol_sha256,
                {'role': 'practice', 'generation_invocations': 0}, compile_only=True)
    else: run_cycle(args, stage1, protocol)


if __name__ == '__main__': main()
