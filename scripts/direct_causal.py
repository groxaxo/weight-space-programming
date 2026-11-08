#!/usr/bin/env python3
"""Finite practice-only activation causality experiment; never edits weights.

Stage 1 has six fixed forwards; stage 2 has eight. There are no supplied vectors,
programs, fitted directions, adaptive strengths, alternative seeds or quality gates.
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import signal
import time
import zipfile

import numpy as np

import direct_probe as probe
import direct_forward
from direct_weights import (DEFAULT_LOCK, load_lock, parent_identity, publish_json,
                            read_json, verify_baseline, bf16_float)
from pipeline import require, sha256

QUESTION = 'half_open_overlap-first-semantic-causality'
DONOR_NAMES = {'request-proof.json', 'result.json', 'tokenization-proof.json',
               'cue-decision.npz', 'length_control-decision.npz', 'runtime-identity.json'}
ARRAY_NAMES = {f'layer_{layer}_{kind}_bf16' for layer in range(32) for kind in ('input', 'output')}
RANDOM_SEEDS = (20261108, 20261109)
PARAMETER_COUNT = 426


def validate_request(request, lock, *, patient_sha256, source_request_sha256):
    probe.keys(request, {'schema_version', 'role', 'question', 'stage', 'lock_sha256',
                         'protocol_sha256', 'patient_sha256', 'source_request_sha256',
                         'donor_files', 'winner_group', 'stage1_result_sha256', 'budget'}, 'causal request')
    require(type(request['schema_version']) is int and request['schema_version'] == 1
            and request['role'] == 'practice' and request['question'] == QUESTION,
            'Only the declared exposed-practice causal experiment is supported')
    require(type(request['stage']) is int and request['stage'] in (1, 2), 'Invalid causal stage')
    for name in ('lock_sha256', 'protocol_sha256', 'patient_sha256', 'source_request_sha256'):
        probe.digest_string(request[name], name)
    require(request['lock_sha256'] == lock['_sha256'] and request['patient_sha256'] == patient_sha256
            and request['source_request_sha256'] == source_request_sha256, 'Wrong source/parent identity')
    files = request['donor_files']
    expected = DONOR_NAMES | ({'stage1-result.json'} if request['stage'] == 2 else set())
    probe.keys(files, expected, 'donor file')
    for digest in files.values():
        probe.digest_string(digest, 'donor file SHA-256')
    if request['stage'] == 1:
        require(request['winner_group'] is None and request['stage1_result_sha256'] is None,
                'Stage 1 cannot preselect a group')
    else:
        require(type(request['winner_group']) is int and 0 <= request['winner_group'] <= 3,
                'Stage 2 must bind one group')
        require(request['stage1_result_sha256'] == files['stage1-result.json'], 'Stage 1 hash binding mismatch')
    budget = request['budget']
    probe.keys(budget, {'max_forward_calls', 'max_wall_seconds', 'postcheck_reserve_seconds'}, 'causal budget')
    require(type(budget['max_forward_calls']) is int
            and budget['max_forward_calls'] == (6 if request['stage'] == 1 else 8), 'Wrong finite call budget')
    require(type(budget['max_wall_seconds']) is int and 300 <= budget['max_wall_seconds'] <= 1700
            and type(budget['postcheck_reserve_seconds']) is int
            and budget['postcheck_reserve_seconds'] == 120, 'Wall-time/postcheck budget is unsupported')


def load_numeric_npz(path):
    require(path.stat().st_size <= 2 * 1024 * 1024, 'Donor NPZ exceeds bounded size')
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        require(len(entries) == 64 and {e.filename for e in entries} == {n + '.npy' for n in ARRAY_NAMES},
                'Duplicate, missing or undeclared NPZ members')
        require(all(e.compress_type == zipfile.ZIP_STORED and not e.flag_bits & 1
                    and 0 < e.file_size <= 20000 for e in entries)
                and sum(e.file_size for e in entries) <= 1024 * 1024, 'Unsafe NPZ ZIP structure')
    with np.load(path, allow_pickle=False, max_header_size=512) as archive:
        require(set(archive.files) == ARRAY_NAMES, 'Unexpected donor arrays')
        arrays = {}
        for name in sorted(ARRAY_NAMES):
            value = archive[name]
            width = 9216 if '_input_' in name else 2560
            require(value.dtype == np.dtype('<u2') and value.shape == (width,), 'Wrong donor dtype/shape')
            require(not bool(np.any((value & 0x7f80) == 0x7f80)), 'Non-finite BF16 donor')
            arrays[name] = value.copy()
    return arrays


def validate_parameter_map(values):
    require(type(values) is dict and len(values) == PARAMETER_COUNT, 'Expected all 426 parent parameters')
    for name, digest in values.items():
        require(type(name) is str and bool(name), 'Invalid parameter name')
        probe.digest_string(digest, 'parameter SHA-256')


def load_donors(request, lock, root, patient, source_request):
    for name, digest in request['donor_files'].items():
        path = root / name
        maximum = 2 * 1024 * 1024 if name.endswith('.npz') else 4 * 1024 * 1024
        require(path.is_file() and path.stat().st_size <= maximum, 'Missing/oversized donor artifact: ' + name)
        require(sha256(path) == digest, 'Donor artifact hash mismatch: ' + name)
    result = read_json(root / 'result.json')
    require(result.get('schema_version') == 2 and result.get('question') == probe.FIRST_ASSERTION_QUESTION
            and result.get('role') == 'practice' and result.get('status') == 'practice_only'
            and result.get('complete') is True and result.get('parameters_unchanged') is True
            and result.get('parameter_postcheck_complete') is True
            and result.get('weights_edited') is False and result.get('quality_evaluation') == 'NOT_RUN'
            and result.get('diagnostic_gate', {}).get('status') == 'diagnostic_gate_passed',
            'Donor study lacks complete authenticated practice evidence')
    require(result.get('parent') == parent_identity(lock) and result.get('lock_sha256') == lock['_sha256']
            and all(result.get(k) == request[k] for k in ('patient_sha256', 'source_request_sha256')),
            'Donor parent/source differs')
    before = result.get('loaded_parameter_sha256_before')
    validate_parameter_map(before)
    require(before == result.get('loaded_parameter_sha256_after'), 'Donor parameters were changed')
    proof = read_json(root / 'request-proof.json')
    require(proof.get('request_sha256') == result.get('request_sha256')
            and all(proof.get(k) == request[k] for k in ('patient_sha256', 'source_request_sha256')),
            'Donor request proof mismatch')
    original = proof.get('request')
    require(type(original) is dict and original.get('schema_version') == 2, 'Missing v2 donor request')
    probe.validate_request(original, lock, patient, source_request,
                           patient_sha256=request['patient_sha256'],
                           source_request_sha256=request['source_request_sha256'])
    require(original['protocol_sha256'] == result.get('protocol_sha256'), 'Donor protocol mismatch')
    scores = {row['id']: row for row in result.get('arms', []) if row.get('status') == 'completed'}
    require(all(name in scores for name in ('cue', 'length_control')), 'Missing donor context scores')
    for context in ('cue', 'length_control'):
        require(scores[context].get('observation_sha256') == request['donor_files'][context + '-decision.npz'],
                'Donor observation hash differs from authenticated score')
    runtime = read_json(root / 'runtime-identity.json')
    probe.keys(runtime, {'torch', 'transformers', 'numpy', 'cpu_reference_kernels', 'probe_sha256',
                         'direct_forward_sha256', 'provenance'}, 'runtime identity')
    provenance = runtime['provenance']
    probe.keys(provenance, {'torch_observation_sha256', 'environment_lock_sha256', 'scope'}, 'runtime provenance')
    for key in ('torch_observation_sha256', 'environment_lock_sha256'):
        probe.digest_string(provenance[key], key)
    for key in ('probe_sha256', 'direct_forward_sha256'):
        probe.digest_string(runtime[key], key)
    require(all(type(runtime[k]) is str and 0 < len(runtime[k]) <= 128
                for k in ('torch', 'transformers', 'numpy')), 'Invalid runtime version pins')
    require(provenance['scope'] == 'Historical Torch observed separately; Transformers/NumPy controller pins; '
                                  'no historical binary closure', 'Runtime provenance scope must be explicit')
    require(runtime['cpu_reference_kernels'] == result['loading_info']['cpu_reference_kernels'],
            'Historical model-kernel identity mismatch')
    stage1 = None
    if request['stage'] == 2:
        stage1 = read_json(root / 'stage1-result.json')
        require(stage1.get('stage') == 1 and stage1.get('question') == QUESTION
                and stage1.get('role') == 'practice' and stage1.get('status') == 'practice_only'
                and stage1.get('complete') is True and stage1.get('parameters_unchanged') is True
                and stage1.get('parameter_postcheck_complete') is True
                and stage1.get('weights_edited') is False and stage1.get('quality_evaluation') == 'NOT_RUN'
                and stage1.get('diagnostic_gate', {}).get('status') == 'activation_group_qualified'
                and stage1['diagnostic_gate'].get('winner_group') == request['winner_group'],
                'Stage 2 lacks a complete qualified fixed-screen result')
        require(stage1.get('parent') == parent_identity(lock) and stage1.get('loaded_parameter_sha256_before') == before
                and stage1.get('loaded_parameter_sha256_after') == before
                and all(stage1.get(k) == request[k] for k in ('patient_sha256', 'source_request_sha256'))
                and all(stage1.get('donor_files', {}).get(k) == request['donor_files'][k] for k in DONOR_NAMES),
                'Stage 1 ancestry or donor identities differ')
        require(diagnostic_gate(1, stage1.get('arms', [])) == stage1['diagnostic_gate'],
                'Stage 1 winner does not follow the frozen selection rule')
        by_id = {row['id']: row for row in stage1['arms']}
        for name, tolerance in (('neutral-replay', 1e-6), ('no-op', 0)):
            replay = by_id[name].get('replay', {})
            require(replay.get('all_64_input_output_vectors_bit_identical') is True
                    and type(replay.get('max_token_logprob_difference')) in (float, int)
                    and 0 <= replay['max_token_logprob_difference'] <= tolerance,
                    'Stage 1 lacks complete replay/no-op proof')
    return {'result': result, 'request': original, 'plans': read_json(root / 'tokenization-proof.json'),
            'scores': scores, 'runtime': runtime, 'stage1': stage1,
            'cue': load_numeric_npz(root / 'cue-decision.npz'),
            'length_control': load_numeric_npz(root / 'length_control-decision.npz')}


def runtime_identity(loading):
    import torch
    import transformers
    return {'torch': str(torch.__version__), 'transformers': str(transformers.__version__),
            'numpy': str(np.__version__), 'cpu_reference_kernels': loading['cpu_reference_kernels'],
            'probe_sha256': sha256(Path(probe.__file__)),
            'direct_forward_sha256': sha256(Path(direct_forward.__file__))}


def score_difference(actual, expected):
    require(actual.get('next_token_id') == expected.get('next_token_id'), 'Replay next token differs')
    differences = []
    for label in probe.LABELS:
        a, b = actual['branches'][label], expected['branches'][label]
        require(a['ids'] == b['ids'] and len(a['token_logprobs']) == len(b['token_logprobs']),
                'Replay candidate tokenization differs')
        differences.extend(abs(float(x) - float(y)) for x, y in zip(a['token_logprobs'], b['token_logprobs']))
    for row in (actual, expected):
        totals = {label: sum(row['branches'][label]['token_logprobs']) for label in probe.LABELS}
        require(all(abs(totals[label] - row['branches'][label]['sequence_logprob']) <= 1e-9
                    for label in probe.LABELS)
                and abs(totals['correct'] - totals['incorrect'] - margin(row)) <= 1e-9,
                'Replay log probabilities have inconsistent totals')
    require(differences and all(math.isfinite(x) for x in differences), 'Invalid replay log probabilities')
    return max(differences)


def check_replay(score, arrays, cached_score, cached_arrays, tolerance=1e-6):
    maximum = score_difference(score, cached_score)
    require(maximum <= tolerance, 'Replay log probabilities differ beyond tolerance')
    require(set(arrays) == set(cached_arrays)
            and all(np.array_equal(arrays[k], cached_arrays[k]) for k in arrays),
            'Replay decision vectors are not bit-identical')
    return {'max_token_logprob_difference': maximum, 'all_64_input_output_vectors_bit_identical': True}


def frozen_deltas(cue, neutral):
    result = {}
    for layer in range(32):
        key = f'layer_{layer}_output_bf16'
        delta = bf16_float(cue[key]) - bf16_float(neutral[key])
        require(bool(np.isfinite(delta).all()), 'Non-finite frozen semantic delta')
        result[layer] = delta
    return result


def sign_randomized(deltas, seed):
    require(seed in RANDOM_SEEDS, 'Undeclared random-control seed')
    rng = np.random.default_rng(seed)
    return {layer: delta * (rng.integers(0, 2, size=delta.shape) * 2 - 1).astype(np.float32)
            for layer, delta in sorted(deltas.items())}


def delta_stats(before, after):
    delta = bf16_float(after).astype(np.float64) - bf16_float(before).astype(np.float64)
    return {'delta_l2': float(np.linalg.norm(delta)),
            'changed_entries': int(np.count_nonzero(before != after)),
            'max_absolute_delta': float(np.max(np.abs(delta), initial=0))}


def matched(actual, reference):
    def valid(rows):
        if type(rows) is not list or len(rows) != 8:
            return False
        for row in rows:
            if type(row) is not dict or set(row) != {'layer', 'delta_l2', 'changed_entries', 'max_absolute_delta'}:
                return False
            if type(row['layer']) is not int or not 0 <= row['layer'] < 32 \
                    or type(row['changed_entries']) is not int or not 0 <= row['changed_entries'] <= 2560:
                return False
            if not all(type(row[k]) in (float, int) and math.isfinite(row[k]) and row[k] >= 0
                       for k in ('delta_l2', 'max_absolute_delta')):
                return False
        layers = {row['layer'] for row in rows}
        return len(layers) == 8 and min(layers) % 8 == 0 and layers == set(range(min(layers), min(layers) + 8))
    if not valid(actual) or not valid(reference) or {r['layer'] for r in actual} != {r['layer'] for r in reference}:
        return False
    by_layer = {row['layer']: row for row in reference}
    return all(all(abs(row[key] - by_layer[row['layer']][key]) <= 0.05 * by_layer[row['layer']][key]
                   for key in ('delta_l2', 'changed_entries')) for row in actual)


def off_target_position(tokenizer, plan, patient):
    count = plan['tokenization']['patient_output_prefix_tokens']
    prompt = len(plan['prefix_ids']) - count
    require(patient['text'].startswith('```python\n') and count >= 2
            and probe.decode(tokenizer, [plan['prefix_ids'][prompt + 1]]) == 'python',
            'Off-target Python language-tag token is not the frozen expected position')
    return prompt + 1


def specifications(stage, group=None):
    if stage == 1:
        return [{'id': 'neutral-replay', 'context': 'length_control'},
                {'id': 'no-op', 'context': 'length_control', 'group': 0, 'strength': 0, 'mode': 'semantic'}] + [
                    {'id': f'group-{g}', 'context': 'length_control', 'group': g, 'strength': 1,
                     'mode': 'semantic'} for g in range(4)]
    return [{'id': 'neutral-replay', 'context': 'length_control'},
            {'id': 'full', 'context': 'length_control', 'group': group, 'strength': 1, 'mode': 'semantic'},
            {'id': 'half', 'context': 'length_control', 'group': group, 'strength': 0.5, 'mode': 'semantic'},
            *[{'id': f'random-{seed}', 'context': 'length_control', 'group': group, 'strength': 1,
               'mode': 'random', 'seed': seed} for seed in RANDOM_SEEDS],
            {'id': 'off-target', 'context': 'length_control', 'group': group, 'strength': 1,
             'mode': 'semantic', 'off_target': True},
            {'id': 'semantic-replay', 'context': 'cue'},
            {'id': 'reverse', 'context': 'cue', 'group': group, 'strength': -1, 'mode': 'semantic'}]


def run_addition(model, tokenizer, plan, outdir, spec, deltas, off_position):
    """Outer hooks add the frozen vector; frozen probe handles capture/scoring."""
    import torch
    handles, stats = [], []
    position = off_position if spec.get('off_target') else plan['decision_position']
    publish_json({'activation_addition': spec, 'position': position,
                  'executor': 'direct_probe.run_forward with registered additive output hooks',
                  'weights_edited': False}, outdir / (spec['id'] + '-arm.json'))
    vectors = sign_randomized(deltas, spec['seed']) if spec.get('mode') == 'random' else deltas
    def addition(layer):
        def hook(module, args, output):
            require(not torch.is_grad_enabled() and output.dtype == torch.bfloat16,
                    'Additive intervention requires BF16 inference mode')
            require(torch.equal(output[0, position], output[1, position]), 'Shared-prefix intervention rows differ')
            before = probe._words(output[0, position], probe.OUTPUT_WIDTH)
            changed = output.clone()
            applied = output[0, position] if spec['strength'] == 0 else (
                output[0, position].float() + spec['strength'] * torch.from_numpy(vectors[layer].copy())
            ).to(torch.bfloat16)
            changed[:, position, :] = applied
            after = probe._words(applied, probe.OUTPUT_WIDTH)
            stats.append({'layer': layer, **delta_stats(before, after)})
            return changed
        return hook
    try:
        if 'group' in spec:
            for layer in range(spec['group'] * 8, (spec['group'] + 1) * 8):
                handles.append(model.model.layers[layer].mlp.down_proj.register_forward_hook(addition(layer)))
        score, arrays = probe.run_forward(model, tokenizer, plan, outdir, spec['id'])
        score.update({'activation_addition': spec, 'position': position, 'actual_interventions': stats,
                      'capture_score_sha256': sha256(outdir / (spec['id'] + '-score.json'))})
        publish_json(score, outdir / (spec['id'] + '-causal-score.json'))
        return score, arrays
    except BaseException as error:
        publish_json({'error': type(error).__name__ + ': ' + str(error), 'activation_addition': spec,
                      'actual_interventions': stats, 'complete': False},
                     outdir / (spec['id'] + '-addition-interruption.json'))
        raise
    finally:
        for handle in handles:
            handle.remove()


def correct(score):
    return score['next_token_id'] == score['branches']['correct']['ids'][0]


def wrong(score):
    return score['next_token_id'] == score['branches']['incorrect']['ids'][0]


def margin(score):
    return score['correct_minus_incorrect_logprob']


def diagnostic_gate(stage, rows, winner=None):
    by_id = {row['id']: row for row in rows if row.get('status') == 'completed'}
    base = {'weight_candidate_eligible': False, 'independent_paired_quality': 'NOT_RUN'}
    needed = specifications(stage, winner)
    if any(spec['id'] not in by_id for spec in needed):
        return {**base, 'status': 'NOT_RUN_incomplete', 'winner_group': None}
    neutral = by_id['neutral-replay']
    if stage == 1:
        qualifying = [(g, margin(by_id[f'group-{g}']) - margin(neutral)) for g in range(4)
                      if correct(by_id[f'group-{g}']) and margin(by_id[f'group-{g}']) > 0
                      and margin(by_id[f'group-{g}']) - margin(neutral) >= 2]
        if not qualifying:
            return {**base, 'status': 'STOP_no_group', 'winner_group': None}
        best = max(effect for _, effect in qualifying)
        selected = min(g for g, effect in qualifying if effect >= best - 1e-6)
        return {**base, 'status': 'activation_group_qualified', 'winner_group': selected,
                'qualifying_group_effects': {str(g): effect for g, effect in qualifying}}
    full, half = by_id['full'], by_id['half']
    effect = margin(full) - margin(neutral)
    checks = {'full_correct': correct(full) and margin(full) > 0 and effect >= 2,
              'dose_monotone': 0 < margin(half) - margin(neutral) < effect,
              'reverse_incorrect': wrong(by_id['reverse']) and margin(by_id['reverse']) < 0,
              'reverse_matched': matched(by_id['reverse']['actual_interventions'], full['actual_interventions'])}
    for name in [f'random-{seed}' for seed in RANDOM_SEEDS] + ['off-target']:
        row = by_id[name]
        checks[name + '_matched'] = matched(row['actual_interventions'], full['actual_interventions'])
        checks[name + '_specific'] = wrong(row) and abs(margin(row) - margin(neutral)) < 0.25 * effect
    return {**base, 'status': 'activation_controls_passed' if all(checks.values()) else 'STOP_controls_failed',
            'winner_group': winner, 'checks': checks,
            'scope': 'Information carrier only; no weight defect, coordinate selection or repair is established'}


def run_causal(request, lock, verified, donor_root, outdir, patient, source_request, *, bindings, started=None):
    started = time.monotonic() if started is None else started
    validate_request(request, lock, patient_sha256=bindings['patient_sha256'],
                     source_request_sha256=bindings['source_request_sha256'])
    donors = load_donors(request, lock, donor_root, patient, source_request)
    model, tokenizer, loading, before = direct_forward.load_text_model(lock, verified)
    report = {'schema_version': 1, 'role': 'practice', 'status': 'practice_only', 'question': QUESTION,
              'stage': request['stage'], **bindings, 'lock_sha256': lock['_sha256'],
              'protocol_sha256': request['protocol_sha256'], 'parent': parent_identity(lock),
              'donor_files': request['donor_files'], 'stage1_result_sha256': request['stage1_result_sha256'],
              'loaded_parameter_sha256_before': before, 'loading_info': loading,
              'causal_code_sha256': sha256(Path(__file__)), 'arms': [], 'complete': False,
              'quality_evaluation': 'NOT_RUN', 'weights_edited': False, 'training': False,
              'weight_candidate_eligible': False, 'runtime_attestation_scope':
              'Current runtime checked against controller-attested historical environment; fresh replay required'}
    error = None
    try:
        validate_parameter_map(before)
        require(before == donors['result']['loaded_parameter_sha256_before']
                and loading == donors['result']['loading_info'], 'Loaded parent/runtime differs from donor model')
        current_runtime = runtime_identity(loading)
        require(current_runtime == {k: v for k, v in donors['runtime'].items() if k != 'provenance'},
                'Current Torch/kernel/runtime identity differs')
        report['runtime_identity'] = current_runtime
        report['historical_runtime_provenance'] = donors['runtime']['provenance']
        if donors['stage1']:
            require(donors['stage1'].get('causal_code_sha256') == report['causal_code_sha256'],
                    'Causal implementation changed since Stage 1')
        publish_json(report, outdir / 'loaded-parent-proof.json')
        plans = probe.prepare_plans(tokenizer, donors['request'], patient)
        require(plans == donors['plans'], 'Fresh tokenization differs from cached donor proof')
        off_position = off_target_position(tokenizer, plans['length_control'], patient)
        require(off_position == off_target_position(tokenizer, plans['cue'], patient),
                'Semantic and neutral off-target positions differ')
        publish_json({'plans': plans, 'off_target_position': off_position}, outdir / 'tokenization-proof.json')
        deltas = frozen_deltas(donors['cue'], donors['length_control'])
        timings, neutral_score, neutral_arrays = [], None, None
        specs = specifications(request['stage'], request['winner_group'])
        for index, spec in enumerate(specs):
            if not probe.forward_admitted(request['budget'], time.monotonic() - started, timings, len(timings)):
                report['arms'].append({'id': spec['id'], 'status': 'NOT_RUN_budget'})
                continue
            score, arrays = run_addition(model, tokenizer, plans[spec['context']], outdir, spec, deltas, off_position)
            timings.append(score['seconds']); report['arms'].append(score)
            if spec['id'] == 'neutral-replay':
                score['replay'] = check_replay(score, arrays, donors['scores']['length_control'], donors['length_control'])
                neutral_score, neutral_arrays = score, arrays
            elif spec['id'] == 'no-op':
                score['replay'] = check_replay(score, arrays, neutral_score, neutral_arrays, tolerance=0)
            elif spec['id'] == 'semantic-replay':
                score['replay'] = check_replay(score, arrays, donors['scores']['cue'], donors['cue'])
            elif spec['id'] == 'full':
                previous = next(row for row in donors['stage1']['arms'] if row['id'] == f'group-{request["winner_group"]}')
                require(score_difference(score, previous) <= 1e-6, 'Winning intervention does not reproduce Stage 1')
                require(score['actual_interventions'] == previous['actual_interventions'],
                        'Winning intervention magnitude/support differs from Stage 1')
            publish_json({'arms': report['arms'], 'complete': False}, outdir / (spec['id'] + '-progress.json'))
        report['complete'] = len(report['arms']) == len(specs) and all(row['status'] == 'completed' for row in report['arms'])
        report['diagnostic_gate'] = diagnostic_gate(request['stage'], report['arms'], request['winner_group'])
    except BaseException as caught:
        error = caught
        report['error'] = type(caught).__name__ + ': ' + str(caught)
        report['diagnostic_gate'] = {'status': 'STOP_integrity_or_replay', 'weight_candidate_eligible': False}
    finally:
        try:
            after = {name: direct_forward.parameter_sha256(model.get_parameter(name)) for name in before}
            report['loaded_parameter_sha256_after'] = after
            report['parameters_unchanged'] = after == before
            require(after == before, 'Parameters changed during activation experiment')
            report['parameter_postcheck_complete'] = True
        except BaseException as caught:
            report['parameter_postcheck_complete'] = False
            report['postcheck_error'] = type(caught).__name__ + ': ' + str(caught)
            if error is None:
                error = caught
        report['elapsed_seconds'] = time.monotonic() - started
        if error:
            report['complete'] = False
            report['diagnostic_gate'] = {'status': 'STOP_integrity_or_replay', 'weight_candidate_eligible': False}
        publish_json(report, outdir / 'result.json')
    if error:
        raise error
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('request', 'patient', 'source-request', 'donors', 'output'):
        parser.add_argument('--' + field, type=Path, required=True)
    parser.add_argument('--base', type=Path, default=Path('/model'))
    parser.add_argument('--lock', type=Path, default=DEFAULT_LOCK)
    args = parser.parse_args()
    started = time.monotonic()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU sandbox is required')
    require(not os.path.lexists(args.output), 'Refusing existing causal artifact directory')
    args.output.mkdir(mode=0o700)
    def expired(_signum, _frame):
        raise TimeoutError('Own causal worker reached its bounded wall time')
    signal.signal(signal.SIGALRM, expired); signal.alarm(1700)
    try:
        lock = load_lock(args.lock)
        request, patient, source = (read_json(path) for path in (args.request, args.patient, args.source_request))
        bindings = {'request_sha256': sha256(args.request), 'patient_sha256': sha256(args.patient),
                    'source_request_sha256': sha256(args.source_request)}
        validate_request(request, lock, patient_sha256=bindings['patient_sha256'],
                         source_request_sha256=bindings['source_request_sha256'])
        signal.alarm(max(1, int(request['budget']['max_wall_seconds'] - (time.monotonic() - started))))
        publish_json({'request': request, **bindings}, args.output / 'request-proof.json')
        verified = verify_baseline(lock, args.base)
        publish_json(verified, args.output / 'baseline-verification.json')
        run_causal(request, lock, verified, args.donors, args.output, patient, source,
                   bindings=bindings, started=started)
    except BaseException as error:
        publish_json({'error': type(error).__name__ + ': ' + str(error), 'complete': False,
                      'quality_evaluation': 'NOT_RUN', 'weights_edited': False, 'preserve': True},
                     args.output / 'failure.json')
        raise
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    main()
