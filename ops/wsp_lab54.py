#!/usr/bin/env python3
"""Admit eight frozen activation controls. No candidate/final/native execution."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from ops import direct_edit54 as host
import wsp_lab as lab
from wsp_interventions import ARMS, norm_matched
from wsp_worker import verify_result
from direct_weights import load_lock, publish_json, read_bound_json, read_json
from pipeline import require, sha256


def capture(source, target, expected, maximum=1024*1024):
    require(source.is_file() and not source.is_symlink() and source.stat().st_size <= maximum
            and sha256(source) == expected, 'Bounded immutable input differs')
    raw = source.read_bytes(); require(hashlib.sha256(raw).hexdigest() == expected, 'Input changed while reading')
    with target.open('xb') as handle: handle.write(raw)
    require(sha256(source) == sha256(target) == expected, 'Input changed during capture')


def container_command(name, code, deps, model_files, output, protocol, design, protocol_hash, arm):
    require(arm in ARMS and ',' not in str(design), 'Invalid fixed arm/design path')
    command = host.container_command(name, code, deps, model_files, output, protocol, 10*host.GIB, 'observe')
    index = command.index(host.IMAGE)
    return command[:index] + ['--mount', f'type=bind,source={design},target=/design.md,readonly', host.IMAGE,
        'python', '-u', 'scripts/wsp_worker.py', '--protocol', '/request.json', '--protocol-sha256', protocol_hash,
        '--design', '/design.md', '--arm', arm, '--output', '/out/worker.json']


def run_arm(args, protocol, protocol_hash, arm, area):
    require(not os.path.lexists(area), 'Existing arm evidence must be preserved')
    area.mkdir(mode=0o700); started = time.monotonic()
    try:
        with host.WorkerLease(area) as lease:
            state = host.resources(area); publish_json(state, area/'resources-before.json'); host.admit(state, 10*host.GIB)
            code = area/'source'; hashes = host.freeze_source(code)
            # Freeze policy too: the old source freezer intentionally copies
            # only scripts/ops/configs. Never change its historical behavior.
            capture(ROOT/'research-policy.json', code/'research-policy.json', protocol['source_sha256']['research-policy.json'])
            hashes['research-policy.json'] = sha256(code/'research-policy.json')
            require(all(hashes.get(n) == v for n,v in protocol['source_sha256'].items()), 'Frozen runtime source differs')
            frozen_protocol, frozen_design = area/'protocol.json', area/'design.md'
            capture(args.protocol, frozen_protocol, protocol_hash)
            capture(args.design, frozen_design, protocol['design_sha256'])
            output = area/'worker-output'; output.mkdir(mode=0o700)
            name = 'wsp-direct-lab-' + arm.replace('_','-') + '-' + uuid.uuid4().hex[:12]
            argv = container_command(name, code.resolve(), args.deps.resolve(strict=True),
                                     host.mounted_files(load_lock(), args.baseline), output.resolve(),
                                     frozen_protocol.resolve(), frozen_design.resolve(), protocol_hash, arm)
            publish_json({'container': name, 'command': argv, 'protocol_sha256': protocol_hash,
                          'source_sha256': hashes, 'arm': arm, 'gpu': False, 'training': False,
                          'service_control': False, 'generation_invocations_max': 1}, area/'invocation.json')
            lease.launching(name, 10*host.GIB)
            with (area/'console.log').open('x') as log:
                child = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT)
                def interrupted(signum, _frame): raise InterruptedError('Own lab interrupted: '+str(signum))
                old = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM)}
                try:
                    while child.poll() is None:
                        time.sleep(5); current = host.resources(area)
                        require(current['mem_available_bytes'] >= 3*host.GIB and current['disk_free_bytes'] >= host.GIB,
                                'Host resource reserve lost')
                        require(time.monotonic()-started < 1800, 'Owned worker exceeded its maximum')
                except BaseException:
                    subprocess.run(['docker','stop','--time','10',name], capture_output=True, timeout=30)
                    child.wait(timeout=30); lease.confirm_exit(); raise
                finally:
                    for sig, handler in old.items(): signal.signal(sig, handler)
            publish_json({'containers': lease.confirm_exit()}, area/'container-inspect.json')
            require(child.returncode == 0, 'Owned worker failed; preserve all partial evidence')
            report = verify_result(output, protocol, protocol_hash, arm)
            resources_after = host.resources(area)
        publish_json({'complete': True, 'exit_code': child.returncode, 'resources_after': resources_after,
                      'worker_seconds': math.ceil(time.monotonic()-started+1)}, area/'execution.json')
        return report
    except BaseException as error:
        publish_json({'complete': False, 'error': type(error).__name__+': '+str(error),
                      'worker_seconds': math.ceil(time.monotonic()-started+1), 'preserve': True}, area/'failure.json')
        raise


def summarize(protocol, records):
    from wsp_pairs import assess_records
    require([r['arm'] for r in records] == list(ARMS[:len(records)]), 'Incomplete/duplicated/reordered fixed controls')
    result = {'complete': len(records) == len(ARMS), 'weights_edited': False, 'training': False,
              'weight_candidate_eligible': False, 'quality_evaluation': 'NOT_RUN',
              'scientific_mechanism_accepted': False, 'not_run_arms': list(ARMS[len(records):]),
              'development_metrics': assess_records(protocol,records)}
    if len(records) != len(ARMS): return {**result, 'status': 'NOT_RUN_incomplete'}
    for record in records:
        require(record['complete'] is True and record['parameters_unchanged'] is True
                and record['parameter_postcheck_complete'] is True, 'Unverified/incomplete model preservation')
        require(record['loaded_parameter_sha256_before'] == records[0]['loaded_parameter_sha256_before']
                == record['loaded_parameter_sha256_after'] and record['runtime'] == records[0]['runtime']
                and record.get('loading_info',{}).get('cpu_reference_kernels')
                    == records[0].get('loading_info',{}).get('cpu_reference_kernels'),
                'Mismatched paired parent or runtime')
    baseline, noop = records[:2]
    if any(r['results'][0]['censored'] for r in records): return {**result, 'status': 'STOP_censored'}
    if baseline['results'][0]['output_ids'] != noop['results'][0]['output_ids']:
        return {**result, 'status': 'STOP_noop_replay'}
    if any(r['results'][0]['input_ids'] != baseline['results'][0]['input_ids'] for r in records):
        return {**result, 'status': 'STOP_input_mismatch'}
    if any(r['intervention']['before_sha256'] != baseline['intervention']['before_sha256'] for r in records[:6]):
        return {**result, 'status': 'STOP_site_replay'}
    treatment = records[2]['intervention']
    if not all(norm_matched(r['intervention'], treatment) for r in records[4:6]):
        return {**result, 'status': 'STOP_random_control_mismatch'}
    if any(r['intervention']['after_sha256'] in (treatment['after_sha256'],baseline['intervention']['after_sha256'])
           for r in records[4:6]):
        return {**result, 'status': 'STOP_random_control_degenerate'}
    return {**result, 'status': 'fixed_activation_records_complete',
            'scope': 'Engineering/control records only. Reverse/semantic causality, sparse derivation and independent acceptance still required.'}


def run(args):
    require(os.uname().nodename == '4790', 'This profile is admitted only on .54')
    protocol, protocol_hash, raw = read_bound_json(args.protocol); lab.validate_protocol(protocol)
    require(sha256(args.design) == protocol['design_sha256'], 'Reviewed design differs')
    require(not os.path.lexists(args.output), 'Refusing existing lab run')
    args.output.mkdir(mode=0o700, parents=True)
    with (args.output/'protocol.json').open('xb') as handle: handle.write(raw)
    capture(args.design, args.output/'design.md', protocol['design_sha256'])
    claim = host.DEFAULT_LEASE_PATH.parent / ('wsp-lab-study-'+hashlib.sha256(protocol['id'].encode()).hexdigest()+'.json')
    fd = os.open(claim, os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_NOFOLLOW, 0o600)
    start = {'study_id': protocol['id'], 'protocol_sha256': protocol_hash, 'output': str(args.output.resolve()),
             'started_at_utc': datetime.now(timezone.utc).isoformat(), 'cleanup': False, 'training': False,
             'source_sha256': protocol['source_sha256'], 'claim': str(claim)}
    with os.fdopen(fd,'w') as handle: json.dump(start,handle,sort_keys=True); handle.flush(); os.fsync(handle.fileno())
    publish_json(start,args.output/'start.json')
    records, ledger, attempted = [], [], 0; charged = protocol['budget']['prior_worker_seconds']
    result = {'status': 'STOP_infrastructure', 'complete': False}
    try:
        for arm in ARMS:
            require(lab.source_manifest() == protocol['source_sha256'] and sha256(args.protocol) == protocol_hash,
                    'Frozen source/protocol changed')
            require(charged+1800 <= protocol['budget']['total_worker_seconds'], 'Cumulative budget exhausted')
            area = args.output/arm; attempted += 1; begin = time.monotonic()
            print(json.dumps({'launch': arm, 'at_utc': datetime.now(timezone.utc).isoformat()}),flush=True)
            try: record = run_arm(args,protocol,protocol_hash,arm,area); records.append(record)
            finally:
                receipt = area/('execution.json' if (area/'execution.json').is_file() else 'failure.json')
                data = read_json(receipt) if receipt.is_file() else {'worker_seconds': 0}
                cost = max(data['worker_seconds'],math.ceil(time.monotonic()-begin+1)); charged += cost
                ledger.append({'arm': arm,'worker_seconds': cost,'receipt_sha256':sha256(receipt) if receipt.is_file() else None})
            # Check no-op numeric identity in addition to greedy token identity.
            if arm == 'noop':
                old = args.output/'base/worker-output'; current = area/'worker-output'
                def arrays(p,r):
                    row=r['results'][0];return lab.evidence.numeric_arrays(p,{'path':row['observation'],'sha256':row['observation_sha256']})
                a,b = arrays(old,records[0]),arrays(current,record)
                if (not all(np.array_equal(a[k],b[k]) for k in a)
                        or record['results'][0]['input_ids'] != records[0]['results'][0]['input_ids']
                        or record['results'][0]['output_ids'] != records[0]['results'][0]['output_ids']):
                    result = {'status':'STOP_noop_replay','complete':False}
                    break
            publish_json({'arm':arm,'completed':len(records),'worker_seconds':charged,
                          'censored':record['results'][0]['censored']},args.output/('progress-'+arm+'.json'))
            if record['results'][0]['censored']:
                result = {'status':'STOP_censored','complete':False};break
        else: result = summarize(protocol,records)
    except BaseException as error:
        result = {'status':'STOP_infrastructure','complete':False,'error':type(error).__name__+': '+str(error)}
        raise
    finally:
        publish_json({**result,'records':records,'ledger':ledger,'worker_seconds':charged,
                      'generation_invocations_attempted':attempted,'not_run_arms':list(ARMS[attempted:]),
                      'attempted_unverified_arms':list(ARMS[len(records):attempted]),
                      'training':False,'weights_edited':False,'weight_candidate_eligible':False,
                      'quality_evaluation':'NOT_RUN','scientific_mechanism_accepted':False},args.output/'result.json')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('protocol','design','baseline','deps','output'):p.add_argument('--'+key,type=Path,required=True)
    run(p.parse_args())


if __name__=='__main__':main()
