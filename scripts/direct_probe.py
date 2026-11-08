#!/usr/bin/env python3
"""Bounded practice-only causal probe of the confirmed empty-interval guard error.

Trusted forward execution only: no generated code execution, fitting, gradients,
weight changes, candidate selection or independent quality evaluation.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import tempfile
import time

import numpy as np

from direct_forward import load_text_model, parameter_sha256
from direct_weights import (DEFAULT_LOCK, ValidationError, load_lock, parent_identity,
                            publish_json, read_json, read_bound_json, verify_baseline, assert_baseline_unchanged)
from pipeline import require, sha256, sync_directory
import direct_evidence as evidence

QUESTION = 'half_open_overlap-empty-interval-guard'
FIRST_ASSERTION_QUESTION = 'half_open_overlap-first-semantic-assertion'
FIRST_ASSERTION_OPERATORS = {'correct': '>', 'incorrect': '>=', 'suffix': ' end).'}
LENGTH_CONTROL = ('For presentation, use a plain style. Prefer descriptive names, consistent indentation, '
                  'and ordinary spacing throughout the body of the function.')
ACTIVE_WALL_LIMIT = 1700
LAYERS, INPUT_WIDTH, OUTPUT_WIDTH = 32, 9216, 2560
PROFILE = {'max_input_tokens': 512, 'thinking': False, 'threads': 1, 'seed': 20261108}
OPERATORS = {'correct': '>', 'incorrect': '>=', 'suffix': ' end:'}
LABELS = ('correct', 'incorrect')


def keys(value, expected, label):
    require(type(value) is dict and set(value) == set(expected), 'Unexpected ' + label + ' fields')


def digest_string(value, label):
    require(type(value) is str and len(value) == 64
            and all(c in '0123456789abcdef' for c in value), 'Invalid ' + label)


def required_contexts(request):
    return ('baseline', 'cue', 'length_control') if request['schema_version'] == 2 else ('baseline', 'cue')


def validate_request(request, lock, patient, source_request, *, patient_sha256, source_request_sha256):
    require(type(request) is dict and type(request.get('schema_version')) is int
            and request['schema_version'] in (1, 2),
            'Unsupported probe schema')
    version = request['schema_version']
    expected = {'schema_version', 'role', 'question', 'lock_sha256', 'protocol_sha256',
                'patient_sha256', 'source_request_sha256', 'original_prompt', 'assistant_prefix',
                'semantic_cue', 'operators', 'profile', 'arms', 'budget'}
    if version == 2:
        expected.add('length_control')
    keys(request, expected, 'probe request')
    require(request['role'] == 'practice'
            and request['question'] == (FIRST_ASSERTION_QUESTION if version == 2 else QUESTION),
            'Only the declared exposed empty-interval research question is supported')
    for name in ('lock_sha256', 'protocol_sha256', 'patient_sha256', 'source_request_sha256'):
        digest_string(request[name], name)
    require(request['lock_sha256'] == lock['_sha256'], 'Wrong immutable parent lock')
    require(request['patient_sha256'] == patient_sha256
            and request['source_request_sha256'] == source_request_sha256, 'Source file hash mismatch')
    require(type(source_request) is dict and source_request.get('role') == 'practice'
            and source_request.get('lock_sha256') == lock['_sha256'], 'Invalid original practice request')
    require(type(patient) is dict and patient.get('hit_token_limit') is False
            and type(patient.get('text')) is str, 'Patient must be a completed uncensored practice generation')
    for field in ('input_ids', 'output_ids'):
        values = patient.get(field)
        require(type(values) is list and 1 <= len(values) <= 1024
                and all(type(v) is int and v >= 0 for v in values), 'Invalid patient ' + field)
    require(type(patient.get('generated_tokens')) is int
            and patient['generated_tokens'] == len(patient['output_ids']), 'Invalid patient token count')
    cases = source_request.get('cases')
    require(type(cases) is list, 'Missing original practice cases')
    matches = [c for c in cases if type(c) is dict and c.get('id') == patient.get('id')]
    require(len(matches) == 1, 'Patient identity not uniquely bound to source request')
    messages = matches[0].get('messages')
    require(type(messages) is list and len(messages) == 1
            and type(messages[0]) is dict and set(messages[0]) == {'role', 'content'}
            and messages[0]['role'] == 'user', 'Only the original single-user practice prompt is supported')
    require(type(request['original_prompt']) is str
            and request['original_prompt'] == messages[0]['content']
            and 1 <= len(request['original_prompt']) <= 4000, 'Original prompt differs from source')
    prefix, text = request['assistant_prefix'], patient['text']
    require(type(prefix) is str and 1 <= len(prefix) <= 20000, 'Invalid assistant prefix')
    if version == 1:
        guards = list(re.finditer(r'(?m)^[ \t]+if start (?P<operator>>=) end:', text))
        require(bool(guards) and prefix == text[:guards[-1].start('operator')],
                'Prefix must end immediately before the last actual if start >= end: guard, not its docstring')
    else:
        assertion = re.search(r'(?m)^[ \t]+ValueError: If any input interval is reversed '
                              r'\(start (?P<operator>>=) end\)\.', text)
        require(assertion is not None and assertion.start('operator') == text.find('>=')
                and prefix == text[:assertion.start('operator')],
                'Prefix must end before the first incorrect semantic docstring assertion')
        require(request['length_control'] == LENGTH_CONTROL, 'The frozen length/format control must be preserved')
    cue = request['semantic_cue']
    require(type(cue) is str and 1 <= len(cue) <= 1000, 'Invalid semantic cue')
    require(request['operators'] == (FIRST_ASSERTION_OPERATORS if version == 2 else OPERATORS),
            'Only > versus >= with the version-specific fixed right context is supported')
    profile = request['profile']
    keys(profile, PROFILE, 'profile')
    require(all(type(profile[k]) is type(v) for k, v in PROFILE.items()) and profile == PROFILE,
            'Only the explicit 512-token CPU diagnostic profile is supported')
    budget = request['budget']
    keys(budget, {'max_forward_calls', 'max_wall_seconds', 'postcheck_reserve_seconds',
                  'controls_admitted'}, 'budget')
    minimum = len(required_contexts(request))
    require(type(budget['max_forward_calls']) is int and minimum <= budget['max_forward_calls'] <= 18,
            'Invalid forward-call budget')
    require(type(budget['max_wall_seconds']) is int
            and 300 <= budget['max_wall_seconds'] <= (ACTIVE_WALL_LIMIT if version == 2 else 2300),
            'Invalid wall-time budget')
    require(type(budget['postcheck_reserve_seconds']) is int
            and 60 <= budget['postcheck_reserve_seconds'] <= 300, 'Invalid postcheck reserve')
    require(type(budget['controls_admitted']) is bool, 'Control admission must be boolean')
    arms = request['arms']
    require(type(arms) is list and len(arms) <= 16, 'Too many activation arms')
    require(not arms or budget['controls_admitted'], 'Activation controls lack explicit admission')
    require(minimum + len(arms) <= budget['max_forward_calls'], 'Declared arms exceed call budget')
    if version == 2:
        require(not arms and not budget['controls_admitted'] and budget['max_forward_calls'] == 3,
                'First-semantic-assertion v2 permits exactly three contexts and no activation arms')
    names = set(required_contexts(request))
    for arm in arms:
        keys(arm, {'id', 'source', 'group', 'strength'}, 'activation arm')
        name = arm['id']
        require(type(name) is str and re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,63}', name)
                and name not in names, 'Invalid/duplicate arm identity')
        names.add(name)
        require(arm['source'] in ('self', 'cue') and type(arm['group']) is int
                and 0 <= arm['group'] <= 3, 'Only the four fixed eight-layer groups are supported')
        require(type(arm['strength']) in (int, float) and math.isfinite(arm['strength'])
                and arm['strength'] in (0, 0.5, 1), 'Only predeclared strengths 0, 0.5, 1 are supported')


def decode(tokenizer, ids, *, skip_special_tokens=False):
    return tokenizer.decode(ids, skip_special_tokens=skip_special_tokens,
                            clean_up_tokenization_spaces=False)


def _ids(values):
    if isinstance(values, Mapping):
        require('input_ids' in values, 'Tokenizer mapping lacks input_ids')
        values = values['input_ids']
    if hasattr(values, 'tolist'):
        values = values.tolist()
    if values and isinstance(values[0], list):
        require(len(values) == 1, 'Expected one rendered prompt')
        values = values[0]
    require(type(values) is list and values and all(type(v) is int and v >= 0 for v in values),
            'Invalid tokenizer IDs')
    return values


def first_difference(a, b):
    for index, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return index
    return None if len(a) == len(b) else min(len(a), len(b))


def prepare_context(tokenizer, request, patient, context):
    """Preserve patient prefix IDs; explicitly expose canonical BPE differences."""
    require(context in required_contexts(request), 'Unknown probe context')
    prompt = request['original_prompt']
    if context == 'cue':
        prompt += '\n\n' + request['semantic_cue']
    elif context == 'length_control':
        prompt += '\n\n' + request['length_control']
    messages = [{'role': 'user', 'content': prompt}]
    chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)
    require(type(chat) is str, 'Chat template must render text')
    prompt_ids = _ids(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                   enable_thinking=False))
    require(decode(tokenizer, prompt_ids) == chat, 'Rendered chat does not round-trip exactly')
    if context == 'baseline':
        require(prompt_ids == patient['input_ids'], 'Original prompt token IDs differ from actual patient')
    require(decode(tokenizer, patient['output_ids'], skip_special_tokens=True) == patient['text'],
            'Patient text differs from its actual generated token IDs')
    prefix = request['assistant_prefix']
    boundary, replay_text = 0, ''
    for count in range(1, len(patient['output_ids']) + 1):
        text = decode(tokenizer, patient['output_ids'][:count])
        if prefix.startswith(text):
            boundary, replay_text = count, text
        elif text.startswith(prefix):
            break
        else:
            require(False, 'Generated prefix cannot be aligned without changing patient tokens')
    require(boundary > 0 and len(prefix) - len(replay_text) <= 32,
            'No bounded exact patient token boundary before guard operator')
    prefix_ids = prompt_ids + patient['output_ids'][:boundary]
    suffix = prefix[len(replay_text):]
    candidate_ids, proof = {}, {}
    for label in LABELS:
        continuation = suffix + request['operators'][label] + request['operators']['suffix']
        candidate = _ids(tokenizer.encode(continuation, add_special_tokens=False))
        require(1 <= len(candidate) <= 8 and decode(tokenizer, candidate) == continuation,
                'Operator continuation must round-trip in at most eight tokens')
        replay = prefix_ids + candidate
        expected = chat + prefix + request['operators'][label] + request['operators']['suffix']
        require(decode(tokenizer, replay) == expected, 'Candidate replay changes the exact character prefix')
        if label == 'incorrect':
            require(candidate == patient['output_ids'][boundary:boundary + len(candidate)],
                    'Incorrect branch does not replay the actual patient operator tokens')
        canonical = _ids(tokenizer.encode(expected, add_special_tokens=False))
        proof[label] = {'candidate_ids': candidate, 'canonical_matches_replay': canonical == replay,
                        'canonical_first_difference': first_difference(canonical, replay),
                        'canonical_token_count': len(canonical), 'replay_token_count': len(replay)}
        candidate_ids[label] = candidate
    require(candidate_ids['correct'] != candidate_ids['incorrect'], 'Operators have identical token sequences')
    require(not any(a != b and candidate_ids[b][:len(candidate_ids[a])] == candidate_ids[a]
                    for a in LABELS for b in LABELS), 'Scored branches must not be prefix events')
    maximum = len(prefix_ids) + max(map(len, candidate_ids.values())) - 1
    require(maximum <= request['profile']['max_input_tokens'],
            f'Probe input needs {maximum} tokens; no silent truncation at 512 is allowed')
    return {'context': context, 'prefix_ids': prefix_ids, 'candidate_ids': candidate_ids,
            'decision_position': len(prefix_ids) - 1, 'max_forward_input_tokens': maximum,
            'tokenization': {'mode': 'original-generated-assistant-token-prefix',
                             'patient_output_prefix_tokens': boundary,
                             'replayed_assistant_prefix_text': replay_text,
                             'prefix_suffix_in_scored_continuation': suffix,
                             'original_prompt_ids_match_patient': context == 'baseline',
                             'branches': proof, 'silent_truncation': False}}


def prepare_plans(tokenizer, request, patient):
    """Prepare all required contexts and enforce v2 length/position matching."""
    plans = {context: prepare_context(tokenizer, request, patient, context)
             for context in required_contexts(request)}
    if request['schema_version'] == 2:
        cue, control = plans['cue'], plans['length_control']
        require(cue['decision_position'] == control['decision_position']
                and cue['max_forward_input_tokens'] == control['max_forward_input_tokens']
                and cue['candidate_ids'] == control['candidate_ids'],
                'Semantic cue and fixed control must match exact token length, position and branches')
    return plans


def reproduces_incorrect_operator(score):
    branches = score.get('branches', {})
    wrong = branches.get('incorrect', {}).get('ids', [])
    return bool(wrong) and score.get('next_token_id') == wrong[0] \
        and score.get('correct_minus_incorrect_logprob', math.inf) < 0


def validate_score(row):
    """Check the raw joint-score arithmetic before using a cached verdict."""
    require(type(row) is dict and row.get('status') == 'completed', 'Incomplete score row')
    branches = row.get('branches')
    keys(branches, set(LABELS), 'scored branches')
    totals = {}
    for label in LABELS:
        branch = branches[label]
        require(type(branch) is dict, 'Invalid score branch')
        ids, values, total = (branch.get(k) for k in ('ids', 'token_logprobs', 'sequence_logprob'))
        require(type(ids) is list and 0 < len(ids) <= 64
                and all(type(token) is int and token >= 0 for token in ids), 'Invalid branch token IDs')
        require(type(values) is list and len(values) == len(ids)
                and all(type(value) in (int, float) and math.isfinite(value) and value <= 0 for value in values)
                and type(total) in (int, float) and math.isfinite(total)
                and math.isclose(sum(values), total, rel_tol=0, abs_tol=1e-9),
                'Inconsistent joint token log probabilities')
        totals[label] = total
    require(branches['correct']['ids'] != branches['incorrect']['ids'], 'Identical scored branches')
    value = row.get('correct_minus_incorrect_logprob')
    require(type(value) in (int, float) and math.isfinite(value)
            and math.isclose(value, totals['correct'] - totals['incorrect'], rel_tol=0, abs_tol=1e-9),
            'Score margin does not match joint probabilities')
    require(type(row.get('next_token_id')) is int and row['next_token_id'] >= 0,
            'Missing greedy token identity')
    if 'preferred_scored_branch' in row:
        expected = 'correct' if value > 0 else 'incorrect' if value < 0 else 'tie'
        require(row['preferred_scored_branch'] == expected, 'Preferred branch differs from score')


def validate_context_scores(rows, contexts):
    require(type(rows) is list and len(rows) == len(contexts)
            and [row.get('id') for row in rows if type(row) is dict] == list(contexts),
            'Missing, duplicate, reordered or unexpected donor context scores')
    for row in rows:
        validate_score(row)


def first_assertion_gate(rows):
    """Fixed diagnostic stop rules; passing never authorizes a weight candidate."""
    by_name = {row['id']: row for row in rows if row.get('status') == 'completed'}
    result = {'weight_candidate_eligible': False, 'activation_screening_admitted': False,
              'independent_paired_quality': 'NOT_RUN'}
    baseline = by_name.get('baseline')
    if baseline is not None and not reproduces_incorrect_operator(baseline):
        return {**result, 'status': 'STOP_localization', 'checks': {'baseline_reproduces_incorrect': False},
                'reason': 'Baseline does not reproduce patient operator; investigate conditional/cached arithmetic replay differences'}
    if not all(context in by_name for context in ('baseline', 'cue', 'length_control')):
        return {**result, 'status': 'NOT_RUN_incomplete', 'checks': {},
                'reason': 'All three authenticated contexts are required'}
    cue, control = by_name['cue'], by_name['length_control']
    bm, cm, nm = (row['correct_minus_incorrect_logprob'] for row in (baseline, cue, control))
    checks = {'baseline_reproduces_incorrect': reproduces_incorrect_operator(baseline),
              'length_control_reproduces_incorrect': reproduces_incorrect_operator(control),
              'length_control_margin_stable': abs(nm - bm) <= 1.0,
              'semantic_greedy_correct': cue['next_token_id'] == cue['branches']['correct']['ids'][0],
              'semantic_margin_correct': cm > 0,
              'semantic_advantage_over_control': cm - nm >= 2.0}
    passed = all(checks.values())
    return {**result, 'status': 'diagnostic_gate_passed' if passed else 'STOP_localization',
            'checks': checks, 'length_control_minus_baseline_nats': nm - bm,
            'semantic_minus_control_nats': cm - nm,
            'reason': 'Conditional semantic contrast merits separate causal review' if passed else
                      'Semantic contrast is absent or confounded by the length/format control; stop screening'}


def score_logits(logits, candidate_ids):
    """Stable sequence log probabilities; no fitting or length normalization."""
    require(isinstance(logits, np.ndarray) and logits.ndim == 3 and logits.shape[0] == 2
            and bool(np.isfinite(logits).all()), 'Invalid/non-finite conditional logits')
    require(np.array_equal(logits[0, 0], logits[1, 0]),
            'Same-prefix candidate rows disagree before their first token')
    scores = {}
    for row, label in enumerate(LABELS):
        ids = candidate_ids[label]
        require(len(ids) <= logits.shape[1], 'Insufficient retained logit positions')
        values = []
        for offset, token in enumerate(ids):
            require(type(token) is int and 0 <= token < logits.shape[2], 'Candidate token out of vocabulary')
            vector = logits[row, offset].astype(np.float64)
            peak = float(np.max(vector))
            normalizer = peak + math.log(float(np.exp(vector - peak).sum()))
            values.append(float(vector[token]) - normalizer)
        scores[label] = {'ids': ids, 'token_logprobs': values, 'sequence_logprob': sum(values)}
    margin = scores['correct']['sequence_logprob'] - scores['incorrect']['sequence_logprob']
    return {'branches': scores, 'correct_minus_incorrect_logprob': margin,
            'preferred_scored_branch': 'correct' if margin > 0 else 'incorrect' if margin < 0 else 'tie',
            'next_token_id': int(np.argmax(logits[0, 0])),
            'scoring': 'Joint log probability of fixed operator plus right context; no length normalization',
            'scope': 'Conditional decision diagnostic; not an unconditional generation or accuracy score'}


def publish_arrays(arrays, output):
    require(arrays and all(type(k) is str and re.fullmatch(r'layer_[0-9]+_(input|output|applied_output)_bf16', k)
                          and isinstance(v, np.ndarray) and v.dtype == np.dtype('<u2') and v.ndim == 1
                          for k, v in arrays.items()), 'Only named numeric BF16 word vectors may be saved')
    fd, stage = tempfile.mkstemp(prefix='.' + output.name + '.building-', dir=output.parent)
    with os.fdopen(fd, 'wb') as handle:
        np.savez(handle, **arrays)
        handle.flush(); os.fsync(handle.fileno())
    os.link(stage, output, follow_symlinks=False)
    sync_directory(output.parent)
    return sha256(output)


def _words(tensor, width):
    import torch
    require(tensor.dtype == torch.bfloat16 and tuple(tensor.shape) == (width,),
            'Decision activation is not the declared BF16 vector')
    words = tensor.detach().contiguous().view(torch.uint16).numpy().copy()
    require(not bool(np.any((words & 0x7f80) == 0x7f80)), 'Non-finite decision activation')
    return words


def run_forward(model, tokenizer, plan, outdir, name, *, arm=None, donors=None):
    """One batch of two causal teacher-forced branches; capture one position only."""
    import torch
    prefix, candidates = plan['prefix_ids'], plan['candidate_ids']
    count = max(len(candidates[k]) for k in LABELS)
    length = len(prefix) + count - 1
    pad = getattr(tokenizer, 'pad_token_id', None)
    if pad is None:
        pad = tokenizer.eos_token_id
    require(type(pad) is int and pad >= 0, 'A numeric padding token is required')
    ids = torch.full((2, length), pad, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for row, label in enumerate(LABELS):
        sequence = prefix + candidates[label][:-1]
        ids[row, :len(sequence)] = torch.tensor(sequence, dtype=torch.long)
        mask[row, :len(sequence)] = 1
    position = plan['decision_position']
    require(position == len(prefix) - 1, 'Decision position is not the preserved prefix boundary')
    arrays, handles, intervention = {}, [], []
    require(len(model.model.layers) == LAYERS, 'Expected exactly 32 language layers')
    selected = set(range(arm['group'] * 8, (arm['group'] + 1) * 8)) if arm else set()
    def capture(layer):
        def hook(module, args, output):
            require(not torch.is_grad_enabled(), 'Probe must run in inference mode')
            x, y = args[0], output
            require(x.ndim == y.ndim == 3 and x.shape[0] == y.shape[0] == 2
                    and x.shape[1] > position and y.shape[1] > position, 'Invalid hook dimensions')
            require(torch.equal(x[0, position], x[1, position])
                    and torch.equal(y[0, position], y[1, position]), 'Candidate rows differ at shared prefix')
            arrays[f'layer_{layer}_input_bf16'] = _words(x[0, position], INPUT_WIDTH)
            arrays[f'layer_{layer}_output_bf16'] = _words(y[0, position], OUTPUT_WIDTH)
            if layer not in selected:
                return None
            source = donors[arm['source']][f'layer_{layer}_output_bf16']
            require(isinstance(source, np.ndarray) and source.dtype == np.dtype('<u2')
                    and source.shape == (OUTPUT_WIDTH,), 'Missing/invalid activation donor')
            donor = torch.from_numpy(source.copy()).view(torch.bfloat16)
            strength = arm['strength']
            changed = y.clone()
            applied = y[0, position] if strength == 0 else (
                donor if strength == 1 else ((1 - strength) * y[0, position].float()
                                             + strength * donor.float()).to(torch.bfloat16))
            changed[:, position, :] = applied
            arrays[f'layer_{layer}_applied_output_bf16'] = _words(applied, OUTPUT_WIDTH)
            delta = applied.float() - y[0, position].float()
            intervention.append({'layer': layer, 'strength': strength,
                                 'delta_l2': float(torch.linalg.vector_norm(delta)),
                                 'max_absolute_delta': float(delta.abs().max())})
            return changed
        return hook
    started = time.monotonic()
    try:
        for layer, block in enumerate(model.model.layers):
            handles.append(block.mlp.down_proj.register_forward_hook(capture(layer)))
        with torch.inference_mode():
            output = model(input_ids=ids, attention_mask=mask, use_cache=False,
                           logits_to_keep=count, return_dict=True)
        require(len([k for k in arrays if '_applied_' not in k]) == LAYERS * 2,
                'Missing decision-position observations')
        logits = output.logits.detach().float().numpy()
        require(logits.shape[:2] == (2, count), 'Unexpected retained logit positions')
        score = score_logits(logits, candidates)
        score['next_token_text'] = decode(tokenizer, [score['next_token_id']])
        observation = outdir / (name + '-decision.npz')
        score.update({'status': 'completed', 'id': name, 'context': plan['context'],
                      'seconds': time.monotonic() - started, 'batch_sequences': 2,
                      'input_tokens_per_sequence_max': length, 'decision_position': position,
                      'observation': observation.name, 'observation_sha256': publish_arrays(arrays, observation),
                      'activation_arm': arm, 'intervention': intervention})
        score_path = outdir / (name + '-score.json')
        publish_json(score, score_path)
        score['score_artifact'] = evidence.artifact(score_path)
        return score, arrays
    except BaseException as error:
        if arrays:
            partial = outdir / (name + '-interrupted-decision.npz')
            digest = publish_arrays(arrays, partial)
            publish_json({'id': name, 'error': type(error).__name__ + ': ' + str(error),
                          'observation': partial.name, 'observation_sha256': digest,
                          'seconds': time.monotonic() - started, 'complete': False},
                         outdir / (name + '-interruption.json'))
        raise
    finally:
        for handle in handles:
            handle.remove()


def forward_admitted(budget, elapsed, timings, calls):
    if calls >= budget['max_forward_calls']:
        return False
    estimate = max(30.0, max(timings, default=0) * 1.5)
    return elapsed + estimate + budget['postcheck_reserve_seconds'] < min(budget['max_wall_seconds'], ACTIVE_WALL_LIMIT)


def run_probe(request, lock, verified, outdir, patient, source_request, *, request_sha256,
              patient_sha256, source_request_sha256, started=None):
    validate_request(request, lock, patient, source_request, patient_sha256=patient_sha256,
                     source_request_sha256=source_request_sha256)
    started = time.monotonic() if started is None else started
    model, tokenizer, loading, before = load_text_model(lock, verified)
    report = {'schema_version': request['schema_version'], 'role': 'practice', 'status': 'practice_only', 'complete': False,
              'question': request['question'], 'parent': parent_identity(lock), 'lock_sha256': lock['_sha256'],
              'request_sha256': request_sha256, 'patient_sha256': patient_sha256,
              'source_request_sha256': source_request_sha256, 'protocol_sha256': request['protocol_sha256'],
              'profile': request['profile'], 'budget': request['budget'], 'loading_info': loading,
              'effective_wall_limit_seconds': min(request['budget']['max_wall_seconds'], ACTIVE_WALL_LIMIT),
              'required_contexts': list(required_contexts(request)),
              'loaded_parameter_sha256_before': before, 'quality_evaluation': 'NOT_RUN',
              'weights_edited': False, 'training': False, 'arms': [],
              'controls_scope': 'Partial controls only: optional grouped self/cue replacements and strengths; '
                                'no matched-random, irrelevant-position or reverse-transfer controls',
              'scope': 'Conditional activation causality diagnostic; no weight candidate or quality gate',
              **evidence.bindings(outdir)}
    error = None
    try:
        publish_json(report, outdir / 'loaded-parent-proof.json')
        report['loaded_parent_proof'] = evidence.artifact(outdir / 'loaded-parent-proof.json')
        plans = prepare_plans(tokenizer, request, patient)
        publish_json(plans, outdir / 'tokenization-proof.json')
        report['tokenization_proof'] = evidence.artifact(outdir / 'tokenization-proof.json')
        timings, donors = [], {}
        specifications = [(context, None) for context in required_contexts(request)] + [(arm['id'], arm) for arm in request['arms']]
        for index, (name, arm) in enumerate(specifications):
            if not forward_admitted(request['budget'], time.monotonic() - started, timings, len(timings)):
                report['arms'].append({'id': name, 'status': 'NOT_RUN_budget'})
                continue
            if arm and not all(k in donors for k in ('self', 'cue')):
                report['arms'].append({'id': name, 'status': 'NOT_RUN_missing_contrast'})
                continue
            plan = plans['baseline' if arm else name]
            score, arrays = run_forward(model, tokenizer, plan, outdir, name, arm=arm, donors=donors)
            timings.append(score['seconds'])
            report['arms'].append(score)
            if arm is None and name in ('baseline', 'cue'):
                donors['self' if name == 'baseline' else 'cue'] = arrays
            publish_json({'arms': report['arms'], 'elapsed_seconds': time.monotonic() - started,
                          'complete': False}, outdir / (name + '-progress.json'))
            if request['schema_version'] == 2 and name == 'baseline' and not reproduces_incorrect_operator(score):
                report['arms'].extend({'id': remaining, 'status': 'NOT_RUN_replay_mismatch'}
                                      for remaining, _ in specifications[index + 1:])
                break
        report['contrast_complete'] = all(any(row['id'] == name and row['status'] == 'completed'
                                     for row in report['arms']) for name in required_contexts(request))
        report['all_requested_arms_completed'] = all(row['status'] == 'completed' for row in report['arms'])
        report['complete'] = report['contrast_complete'] and report['all_requested_arms_completed']
        if request['schema_version'] == 2:
            report['diagnostic_gate'] = first_assertion_gate(report['arms'])
    except BaseException as caught:
        error = caught
        report['error'] = type(caught).__name__ + ': ' + str(caught)
    finally:
        # Recheck ALL loaded language parameters even on interrupted/failed probes.
        try:
            after = {name: parameter_sha256(model.get_parameter(name)) for name in before}
            report['loaded_parameter_sha256_after'] = after
            report['parameters_unchanged'] = after == before
            require(after == before, 'Loaded parameter bytes changed during activation diagnosis')
            if verified.get('status') == 'baseline_verified':
                assert_baseline_unchanged(verified)
            report['parameter_postcheck_complete'] = True
        except BaseException as caught:
            report['parameter_postcheck_complete'] = False
            report['postcheck_error'] = type(caught).__name__ + ': ' + str(caught)
            if error is None:
                error = caught
        report['elapsed_seconds'] = time.monotonic() - started
        if error is not None:
            report['complete'] = False
        publish_json(report, outdir / 'result.json')
    if error is not None:
        raise error
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('request', 'patient', 'source-request', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--base', type=Path, default=Path('/model'))
    parser.add_argument('--lock', type=Path, default=DEFAULT_LOCK)
    args = parser.parse_args()
    started = time.monotonic()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'No CUDA access is permitted')
    require(not os.path.lexists(args.output), 'Refusing an existing artifact directory')
    args.output.mkdir(mode=0o700)
    def expired(_signum, _frame):
        raise TimeoutError('Own decision probe reached its bounded wall-time limit')
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(ACTIVE_WALL_LIMIT)
    try:
        lock = load_lock(args.lock)
        captured = [read_bound_json(path) for path in (args.request, args.patient, args.source_request)]
        request, patient, source = (item[0] for item in captured)
        bindings = dict(zip(('request_sha256', 'patient_sha256', 'source_request_sha256'),
                            (item[1] for item in captured)))
        validate_request(request, lock, patient, source, patient_sha256=bindings['patient_sha256'],
                         source_request_sha256=bindings['source_request_sha256'])
        signal.alarm(max(1, int(min(request['budget']['max_wall_seconds'], ACTIVE_WALL_LIMIT)
                                - (time.monotonic() - started))))
        publish_json({'request': request, **bindings}, args.output / 'request-proof.json')
        verified = verify_baseline(lock, args.base)
        publish_json(verified, args.output / 'baseline-verification.json')
        run_probe(request, lock, verified, args.output, patient, source, **bindings, started=started)
    except BaseException as error:
        publish_json({'error': type(error).__name__ + ': ' + str(error), 'complete': False,
                      'quality_evaluation': 'NOT_RUN', 'preserve': True}, args.output / 'failure.json')
        raise
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    main()
