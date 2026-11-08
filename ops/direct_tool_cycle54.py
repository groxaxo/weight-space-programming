#!/usr/bin/env python3
"""Run only the frozen two-episode original-BF16 tool reproduction on .54.

No candidate, independent final cases, deployment or automatic later stage is
accepted here. Each generation has its own admitted retained broker/container.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import direct_tools as tools
from direct_weights import load_lock, publish_json, read_json
from pipeline import require, sha256
from ops.direct_edit54 import worker_evidence


def request(lock, phase, replay, files):
    value = {'schema_version': 1, 'role': 'practice', 'question': tools.QUESTION,
             'lock_sha256': lock['_sha256'], 'protocol_sha256': sha256(tools.PROTOCOL_FILE),
             'fixture_sha256': tools.FIXTURE_SHA256, 'phase': phase, 'replay': replay,
             'episode_files': files, 'generation': dict(tools.PROFILE)}
    tools.validate_request(value, lock)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline', 'deps', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--preflight', type=Path, required=True)
    args = parser.parse_args()
    require(os.uname().nodename == '4790', 'This cycle is admitted only on .54')
    require(not os.path.lexists(args.output), 'Refusing existing cycle evidence')
    args.output.mkdir(parents=True, mode=0o700)
    lock, records, ledger = load_lock(), [], []
    preflight = read_json(args.preflight / 'worker-output/preflight.json')
    require(preflight.get('status') == 'template_parser_preflight_passed'
            and preflight.get('generation_invocations') == 0 and preflight.get('model_loaded') is False
            and preflight.get('protocol_sha256') == sha256(tools.PROTOCOL_FILE)
            and preflight.get('source_sha256') == tools.evidence.SOURCE_AT_IMPORT,
            'Frozen template/parser preflight is missing or belongs to different source/protocol')
    freeze = {name: sha256(ROOT / name) for name in (
        'configs/direct-tool-finalization-cycle2.protocol.json', 'AGENTS.md', 'research-policy.json',
        'docs/plans/direct-tool-finalization-cycle2-20261108.md',
        'docs/experiments/direct-tool-finalization-cycle2-20261108/SURGEON-PROPOSAL.json')}
    protocol = read_json(tools.PROTOCOL_FILE)
    require(freeze['docs/plans/direct-tool-finalization-cycle2-20261108.md'] == protocol['plan_sha256']
            and freeze['docs/experiments/direct-tool-finalization-cycle2-20261108/SURGEON-PROPOSAL.json'] == protocol['proposal_sha256'],
            'Reviewed scientific proposal no longer matches the frozen protocol')
    publish_json({'preflight_sha256': sha256(args.preflight / 'worker-output/preflight.json'), 'started_at_utc': datetime.now(timezone.utc).isoformat(), 'pid': os.getpid(),
                  'phase': 'original-BF16-reproduction', 'frozen_documents': freeze,
                  'source_sha256': tools.evidence.SOURCE_AT_IMPORT, 'training': False,
                  'weight_editing': False, 'independent_quality': 'NOT_RUN',
                  'model_calls_max': 4, 'generation_invocations_per_worker': 1}, args.output / 'cycle-start.json')
    active = None
    def interrupted(signum, _frame):
        if active is not None and active.poll() is None:
            active.send_signal(signal.SIGTERM)
        raise InterruptedError('Own cycle interrupted: ' + str(signum))
    signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGINT, interrupted)
    try:
        for replay in (1, 2):
            initial = None
            for phase in ('initial', 'continuation'):
                require(all(sha256(ROOT / name) == digest for name, digest in freeze.items()),
                        'Frozen cycle document changed during execution')
                if phase == 'continuation' and initial is None:
                    ledger.append({'replay': replay, 'phase': phase, 'status': 'NOT_RUN_invalid_initial_call'})
                    continue
                files = {name: sha256(initial / name) for name in tools.capsule_names(replay)} if initial else {}
                value = request(lock, phase, replay, files)
                request_path = args.output / f'episode-{replay}-{phase}.request.json'
                publish_json(value, request_path)
                area = args.output / f'episode-{replay}-{phase}'
                command = [sys.executable, str(ROOT / 'ops/direct_edit54.py'), tools.MODE,
                           '--baseline', str(args.baseline), '--deps', str(args.deps),
                           '--request', str(request_path), '--output', str(area)]
                if initial is not None: command += ['--episode', str(initial)]
                print(json.dumps({'launch': phase, 'replay': replay, 'output': str(area),
                                  'time_utc': datetime.now(timezone.utc).isoformat()}), flush=True)
                with (args.output / f'episode-{replay}-{phase}.broker.log').open('x') as log:
                    active = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                    started = time.monotonic()
                    while active.poll() is None:
                        time.sleep(5)
                        require(time.monotonic() - started < 1900, 'Own broker exceeded bounded exit reserve')
                    exit_code = active.returncode
                active = None
                require(exit_code == 0, 'Owned worker failed or was not admitted; retained broker evidence: ' + str(area))
                output = area / 'worker-output'
                frozen_sources = read_json(area / 'invocation.json')['source_sha256']
                proof = worker_evidence(output, tools.MODE, value, sha256(request_path), lock['_sha256'],
                                        source_hashes=frozen_sources)
                require(proof['worker_complete'], 'Tool invocation is incomplete')
                report = read_json(output / 'worker.json')
                analysis = tools.evidence.bound_document(output, report['tool_analysis'])
                records.append({'replay': replay, 'phase': phase, 'worker': report, 'analysis': analysis})
                ledger.append({'replay': replay, 'phase': phase, 'status': 'completed',
                               'classification': analysis['classification'], 'evidence': proof,
                               'container': read_json(area / 'invocation.json')['container'],
                               'request_sha256': sha256(request_path)})
                publish_json({'records': ledger, 'complete': False}, args.output / f'progress-{replay}-{phase}.json')
                print(json.dumps(ledger[-1]), flush=True)
                if phase == 'initial' and analysis['classification'] == 'expected_initial_call':
                    initial = output
        result = tools.reproduction_gate(records)
        publish_json({**result, 'ledger': ledger, 'original_parent': lock['_sha256'],
                      'frozen_documents': freeze, 'artifacts_retained': True}, args.output / 'reproduction-gate.json')
        print(json.dumps(result), flush=True)
    except BaseException as error:
        if active is not None and active.poll() is None:
            # Only the broker started above receives a stop; it stops only its owned container.
            active.send_signal(signal.SIGTERM)
            active.wait(timeout=60)
        publish_json({'status': 'STOP_infrastructure', 'error': type(error).__name__ + ': ' + str(error),
                      'ledger': ledger, 'generation_invocations_completed': len(records),
                      'quality_evaluation': 'NOT_RUN', 'weights_edited': False, 'preserve': True},
                     args.output / 'cycle-failure.json')
        raise


if __name__ == '__main__': main()
