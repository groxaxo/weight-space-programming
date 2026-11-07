#!/usr/bin/env python3
"""Fail-closed comparison for the user-selected Experimentos OpenCode HE9 set.

This diagnostic slice has its own contract. It never asserts that the separate
all-164 HumanEval acceptance gate passed.
"""
import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
from coder_common import check, digest, fingerprint, loads, write_json
from ops.replay_he9 import TASKS, LIMITS, PRIVATE_SHA256, GRADER_IMAGE
from coder_sampling import QWEN35_INSTRUCT

TOTALS = dict(zip(TASKS, [83, 66, 788, 2, 897, 710, 1000, 1000, 1000]))


def validated_replay(report):
    check(report['schema_version'] == 1 and report['suite'] == 'experimentos-selected-he9'
        and report['status'] == 'complete' and report['tasks'] == TASKS, 'Incomplete or different HE9 replay')
    check(report['suite_sha256'] == PRIVATE_SHA256 and report['limits'] == LIMITS
        and report['grader_image'] == GRADER_IMAGE, 'Different frozen suite or runtime limits')
    virgin = report['virgin_guest_proof']
    check(all(virgin.get(k) is True for k in ('no_prior_opencode_binary', 'no_prior_opencode_state',
        'no_prior_hermes_state', 'no_prior_harness_workspace')), 'Missing virgin VM proof')
    check(report['vm']['personal_vms_modified'] is False, 'Unexpected personal VM changes')
    rows = report['results']
    check(len(rows) == 9 and len({r['task_id'] for r in rows}) == 9
        and {r['task_id'] for r in rows} == set(TASKS), 'Nine distinct scored tasks are required')
    first = rows[0]
    check(first['identity']['runtime'].get('sampling')==QWEN35_INSTRUCT,
          'Missing declared Qwen instruct sampling profile')
    for row in rows:
        check(row['identity'] == first['identity'], 'Model identity drift within a replay')
        check(row['limits'] == LIMITS and row['suite_sha256'] == PRIVATE_SHA256
            and row['grader_image'] == GRADER_IMAGE, 'Task protocol drift')
        check(row['grading'] in ('scored', 'scored_artifact_at_budget_limit')
            and not row.get('infrastructure_error') and not row['actor_errors'], 'An unscored/API failure cannot become a model score')
        check(row['actor_status'] in ('finished', 'budget_limit') and
            row['actor_exit'] == (0 if row['actor_status'] == 'finished' else 124), 'Invalid actor completion status')
        check(row['preflight']['empty_client_state'] is True and row['preflight']['interfaces'] == ['lo']
            and row['preflight']['docker_socket_mounted'] is False and row['preflight']['python'] == '3.11.17',
            'Missing fresh actor isolation proof')
        check(row['isolation']['verified'] is True and row['isolation']['network'] == 'none'
            and not row['remaining_actor_containers'], 'Incomplete container isolation or cleanup')
        proof = row['model_attestation']
        check(proof['verified'] is True and proof['requests'] > 0
            and proof['instance_id'] == row['instance_id']
            and proof['identity_sha256'] == fingerprint(row['identity'])
            and proof['proxy_sha256'] == report['protocol_sha256']['ops/client_proxy.py'], 'Missing actual-client model attribution')
        for key in ('actor_image', 'opencode_version', 'opencode_sha256', 'harness_sha256',
                    'reference_selftest_sha256', 'project_source_sha256'):
            check(row[key] == first[key], 'Actor/grader/source drift within a replay: '+key)
        check(all(report['protocol_sha256'].get(n) == h for n, h in row['harness_sha256'].items()), 'Executed harness differs from plan')
        check(type(row['base_pass']) is bool and type(row['plus_pass']) is bool
            and row['plus_total'] == TOTALS[row['task_id']] and type(row['plus_passed']) is int
            and 0 <= row['plus_passed'] <= row['plus_total'], 'Invalid reconstructed grader totals')
        check(not row['plus_pass'] or (row['base_pass'] and row['plus_passed'] == row['plus_total']
            and row['error'] is None), 'Inconsistent passing verdict')
    return {r['task_id']: r for r in rows}


def compare_he9(base, candidate):
    before, after = validated_replay(base), validated_replay(candidate)
    for key in ('vm', 'virgin_guest_proof', 'protocol_sha256', 'limits', 'suite_sha256', 'grader_image'):
        check(base[key] == candidate[key], 'Unpaired replay field: '+key)
    b, c = before[TASKS[0]], after[TASKS[0]]
    for key in ('actor_image', 'opencode_version', 'opencode_sha256', 'harness_sha256',
                'reference_selftest_sha256', 'project_source_sha256'):
        check(b[key] == c[key], 'Unpaired actor or source: '+key)
    bi, ci = b['identity'], c['identity']
    check(bi['adapter'] is None and ci['adapter'] and ci['adapter_application'] == 'native-f32-lora-on-q8_0',
          'Need a fresh base and an unmerged trained adapter')
    for key in ('base_model', 'revision', 'model_proof_sha256', 'artifact_sha256', 'quantization',
                'runtime', 'protocol_sha256', 'chat_stop_token_ids', 'limits'):
        check(bi[key] == ci[key], 'Unpaired native inference setting: '+key)
    for key in ('adapter_sha256', 'adapter_config_sha256', 'lora_artifact_sha256'):
        check(re.fullmatch('[a-f0-9]{64}', ci[key]) is not None, 'Missing adapter provenance')
    gained = [t for t in TASKS if after[t]['plus_pass'] and not before[t]['plus_pass']]
    lost = [t for t in TASKS if before[t]['plus_pass'] and not after[t]['plus_pass']]
    base_lost = [t for t in TASKS if before[t]['base_pass'] and not after[t]['base_pass']]
    completed = lambda rows: sum(r['actor_status'] == 'finished' and r['final_step_observed'] for r in rows.values())
    strict = sum(r['plus_pass'] for r in after.values()) > sum(r['plus_pass'] for r in before.values())
    qualified = strict and not lost and not base_lost and completed(after) >= completed(before)
    return {'schema_version': 1, 'status': 'he9_qualified_pending_compatibility' if qualified else 'not_promoted',
        'strict_improvement': strict, 'improvements': gained, 'regressions': lost,
        'base_test_regressions': base_lost,
        'baseline': {'base': sum(r['base_pass'] for r in before.values()),
            'plus': sum(r['plus_pass'] for r in before.values()), 'completed_actors': completed(before)},
        'candidate': {'base': sum(r['base_pass'] for r in after.values()),
            'plus': sum(r['plus_pass'] for r in after.values()), 'completed_actors': completed(after)},
        'total': 9, 'adapter_promoted': False,
        'scope': 'Exact nine historical Qwen failures through fresh OpenCode. Frozen reconstructed Base+Plus '
            'grader; diagnostic selected subset. This is not full HumanEval or official EvalPlus parity. '
            'Independent OpenCode/Hermes development compatibility must pass before local promotion.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline', 'candidate', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    result = compare_he9(loads(args.baseline.read_text()), loads(args.candidate.read_text()))
    result['input_sha256'] = {str(p): digest(p) for p in (args.baseline, args.candidate)}
    write_json(args.output, result)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['strict_improvement'] and result['status'] != 'not_promoted' else 2)


if __name__ == '__main__': main()
