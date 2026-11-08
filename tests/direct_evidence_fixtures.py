"""Synthetic host evidence only. No model, training or benchmark is executed."""
import json
from pathlib import Path

import numpy as np

import direct_evidence as e
import direct_causal as causal
import direct_probe as probe
from direct_weights import load_lock, parent_identity
from pipeline import sha256
from test_direct_probe import first_assertion_fixture
from test_direct_causal import score, stats


def write(path, value):
    path.write_text(json.dumps(value))
    return e.artifact(path)


def make_worker(output, mode='causal', *, stage=1, skipped=()):
    root = output / mode if mode in ('probe', 'causal') else output
    root.mkdir(parents=True, exist_ok=True)
    tokenizer, original, patient, source, lock = first_assertion_fixture()
    request = original
    if mode == 'causal':
        from test_direct_causal_host import request as causal_request
        request = causal_request()
        request.update(lock_sha256=lock['_sha256'], stage=stage, winner_group=None if stage == 1 else 1)
    if mode in ('verify', 'observe'):
        from test_direct_forward import request as forward_request
        request = forward_request()
    parent = parent_identity(lock)
    baseline = {'status': 'baseline_verified', 'parent': parent,
                'files': {name: {**spec, 'path': '/model/' + name,
                                'identity': {'dev': 1, 'ino': i + 1, 'size': spec['size'], 'mtime_ns': 2, 'ctime_ns': 2}}
                          for i, (name, spec) in enumerate(lock['model']['files'].items())},
                'tensors': lock['permitted_tensors']}
    write(root / 'baseline-verification.json', baseline)
    report = {'request_sha256': 'f' * 64, 'protocol_sha256': request['protocol_sha256'],
              'parent': parent, 'training': False, 'complete': not skipped, **e.bindings(root)}
    if mode == 'verify':
        report.update(mode=mode, patch_applied=False, quality_improvement_claimed=False)
        path = root / 'worker.json'; write(path, report)
        return request, report, path
    parameters = {name: 'd' * 64 for name in e.expected_parameter_names()}
    report.update(loaded_parameter_sha256_before=parameters, loaded_parameter_sha256_after=dict(parameters),
                  parameter_postcheck_complete=True, parameters_unchanged=True, loading_info={})
    report['loaded_parent_proof'] = write(root / 'loaded-parent-proof.json', report)
    if mode == 'observe':
        from direct_forward import completion_status
        from types import SimpleNamespace
        case_id = request['cases'][0]['id']
        row = {'id': case_id, 'input_ids': [1], 'output_ids': [3, 2], 'text': 'fixture',
               'input_tokens': 1, 'generated_tokens': 2,
               **completion_status([3, 2], request['generation']['max_new_tokens'],
                                   SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=2)),
                                   SimpleNamespace(eos_token_id=2))}
        row['observation'] = case_id + '-prefill.npz'
        arrays(root / row['observation'])
        row['observation_sha256'] = sha256(root / row['observation'])
        row['generation_sha256'] = write(root / (case_id + '-generation.json'), row)['sha256']
        report.update(mode=mode, patch_applied=False, quality_improvement_claimed=False, results=[row])
        path = root / 'worker.json'; write(path, report)
        return request, report, path
    report.update(role='practice', status='practice_only', quality_evaluation='NOT_RUN', weights_edited=False,
                  patient_sha256=request['patient_sha256'], source_request_sha256=request['source_request_sha256'])
    # Production proof is written after identity/scope fields have been assembled.
    report['loaded_parent_proof'] = write(root / 'loaded-parent-proof.json',
                                        {k: v for k, v in report.items() if k != 'loaded_parent_proof'})
    plans = probe.prepare_plans(tokenizer, original, patient)
    plan_doc = {'plans': plans, 'off_target_position': 1} if mode == 'causal' else plans
    report['tokenization_proof'] = write(root / 'tokenization-proof.json', plan_doc)
    specs = causal.specifications(stage, request['winner_group']) if mode == 'causal' else [
        {'id': name, 'context': name} for name in probe.required_contexts(request)]
    rows = []
    for spec in specs:
        name = spec['id']
        if name in skipped:
            rows.append({'id': name, 'status': 'NOT_RUN_budget'}); continue
        plan = plans[spec['context']]
        value = 6 if spec['context'] == 'cue' else -3.8 if spec['context'] == 'length_control' else -4
        row = score(name, plan, value)
        row.pop('actual_interventions')
        row.update(context=spec['context'], decision_position=plan['decision_position'],
                   input_tokens_per_sequence_max=plan['max_forward_input_tokens'], activation_arm=None,
                   intervention=[], observation=name + '-decision.npz')
        arrays(root / row['observation'])
        row['observation_sha256'] = sha256(root / row['observation'])
        row['score_artifact'] = write(root / (name + '-score.json'), row)
        if mode == 'causal':
            row.update(activation_addition=spec, position=plan['decision_position'],
                       actual_interventions=stats(spec['group']) if 'group' in spec else [],
                       capture_score_sha256=row['score_artifact']['sha256'])
            row['causal_score_artifact'] = write(root / (name + '-causal-score.json'), row)
            if name in ('neutral-replay', 'no-op', 'semantic-replay'):
                row['replay'] = {'all_64_input_output_vectors_bit_identical': True,
                                 'max_token_logprob_difference': 0.0}
        rows.append(row)
    report['arms'] = rows
    if mode == 'causal':
        report.update(stage=stage, question=causal.QUESTION, donor_files=request['donor_files'],
                      diagnostic_gate=causal.diagnostic_gate(stage, rows, request['winner_group']))
    else:
        report.update(contrast_complete=not skipped, all_requested_arms_completed=not skipped,
                      required_contexts=list(probe.required_contexts(request)),
                      diagnostic_gate=probe.first_assertion_gate(rows))
    path = root / 'result.json'; write(path, report)
    return request, report, path


def arrays(path, applied=()):
    values = {f'layer_{i}_{kind}_bf16': np.zeros(9216 if kind == 'input' else 2560, dtype='<u2')
              for i in range(32) for kind in ('input', 'output')}
    values.update({f'layer_{i}_applied_output_bf16': np.zeros(2560, dtype='<u2') for i in applied})
    with path.open('wb') as handle:
        np.savez(handle, **values)
