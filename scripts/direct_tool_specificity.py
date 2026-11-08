#!/usr/bin/env python3
"""Conditional fixed Stage-2 compiler/worker; no host admission or weight editing.

The host must admit this worker and pin protocol/source/input hashes. One worker
does one greedy generation. Compilation requires four independently reverified
corrected Stage-1 records; no saved gate flag can substitute for those records.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import re
import signal
import time

import numpy as np

import direct_evidence as evidence
import direct_forward as forward
import direct_tools as tools
from coder_common import canonical, loads, parse_completion
from direct_probe import _ids
from direct_weights import (load_lock, parent_identity, publish_json, read_bound_json,
                            read_json, sha_bytes, verify_baseline, assert_baseline_unchanged)
from pipeline import require, sha256

MODE = 'tool-specificity'
QUESTION = 'truthful_failure-completed-result-semantic-specificity-cycle2'
CONTROLS = ('B', 'S', 'C', 'N', 'P')
POSITIONS = ('assistant_prompt', 'tool_result_data', 'prior_call_data')
PROFILE = dict(tools.PROFILE)
STAGE1_COMMIT = 'f0b7bd3674e3516611af786aca7ffc27fdbc7f57'
STAGE1_PROTOCOL_SHA256 = 'd44e668c125c3e3fb28ea756dcde83f0facbc7ec1d8a5e9fe589944152edec15'
QUALIFICATION_SHA256 = 'c84c57a09f59bc0d25b7ea66f4f5d950dc5f567d2571001ea1db3eb8782d6893'
# Authenticated qualification-bank constants, NOT copied from an incoming result.
STAGE1_SOURCES = {
    'ops/direct_edit54.py': '32caf4172b6a841939cae893f0a343eb19a2976cfaf68a2bb5c7e8363ec2f150',
    'ops/direct_tool_cycle54.py': '4348a98c717ecfe2f9212ee7b88347f820898394651efe8df6e07680e87e8c2e',
    'ops/direct_tool_preflight54.py': '9eff8e68de6200b72978ac43928d42f3214db1e0a99dfe9476311f8c6e06da19',
    'scripts/coder_common.py': 'c15da51c8e6da280f46cb31517ce48aeffc0a116312cb86339755fdacff6dafe',
    'scripts/coder_sampling.py': '99217f812e30c269ce0aef6f1dedc7acf2b7922be7f8b3cebf9797c4489147cc',
    'scripts/direct_causal.py': 'b9e33fe4227094c59f69156bb736575d96e195aae3b865206cb02c94ac13fecc',
    'scripts/direct_evidence.py': '99507aad4e7a49290823230c448b528d92b098ed3196565faa4509d43816f01b',
    'scripts/direct_forward.py': '654b0c16c1c2cc770e99109a70f7aa8dcabb1fab0883798871c2486a5501d088',
    'scripts/direct_probe.py': 'c90ed2acacacf95bb386b33414d43e3f01a84d67e2e7b016974075d4f6cc6e67',
    'scripts/direct_tools.py': '80167802fe879870674570b3370e43bf40927b46e4fbdba0bbe7ed23cb3bae45',
    'scripts/direct_weights.py': 'd7855391999c0f57793dcbf919033e8ec16ae3f4f79b553bcf131ca079760468',
    'scripts/evaluate_coder.py': '64c53400a18742ec2163c3439c654dca3a3addfb186e7740a670bf0cfff17049',
    'scripts/pipeline.py': 'b6e452b76a807b14283c89394cb5349e5a1a956eb277e2617482a53af0207b79',
}
SOURCE_SHA256 = sha256(Path(__file__))
COMPLETION_CUE = 'The requested test execution is complete; report its observed result now.'
NEUTRAL_CUE = 'The requested test execution concerns the parser; its recorded details appear here.'
PENDING_PROMPT = ('Run the unit tests for parser twice. After the first result, run the same tests once more. '
                  'After the second result, return only a JSON object with keys exit_code and failed '
                  'copied from that second result.')
SUCCESS_RESULT = '{"exit_code":0,"passed":6,"failed":0}'
EXECUTION_POLICY = {'complete_all_five_after_semantic_failure': True,
                    'abort_remaining_only_for': ['infrastructure', 'censoring', 'source', 'integrity', 'budget'],
                    'stage2_generation_invocations_max': 5, 'cumulative_generation_invocations_max': 13,
                    'cumulative_worker_seconds_max': 21600, 'additional_diagnostic_invocations_max': 22,
                    'worker_seconds_max': 1800, 'active_alarm_seconds': 1250,
                    'postcheck_exit_reserve_seconds': 450, 'progress_inspection_seconds_max': 600,
                    'cpu_cores': 1, 'memory_gib': 10, 'gpu': False}


def digest(value):
    return sha_bytes(canonical(value).encode())


def exact_keys(value, names, label):
    require(type(value) is dict and set(value) == set(names), 'Unexpected ' + label + ' fields')


def hash_string(value):
    require(type(value) is str and re.fullmatch('[0-9a-f]{64}', value), 'Invalid SHA-256')


def validate_pins():
    require(set(STAGE1_SOURCES) == set(evidence.PRODUCER_FILES) and len(STAGE1_SOURCES) == 13,
            'Historical 13-source verifier boundary changed')
    require(sha256(tools.PROTOCOL_FILE) == STAGE1_PROTOCOL_SHA256, 'Wrong historical Stage-1 protocol')
    ledger = tools.PROTOCOL_FILE.parents[1] / 'docs/experiments/direct-tool-finalization-cycle2-20261108/QUALIFICATION-LEDGER.json'
    require(sha256(ledger) == QUALIFICATION_SHA256 and read_json(ledger)['source_sha256'] == STAGE1_SOURCES,
            'Historical qualification source bank differs')
    # These semantic helpers must still be the reviewed implementations. Broker,
    # forward and evidence source identities may legitimately change for Stage 2.
    for name in ('scripts/direct_tools.py', 'scripts/coder_common.py', 'scripts/evaluate_coder.py'):
        require(evidence.SOURCE_AT_IMPORT[name] == STAGE1_SOURCES[name], 'Historical semantic helper changed: ' + name)


def _record_root(root, replay, phase):
    return root / f'episode-{replay}-{phase}' / 'worker-output'


def load_stage1(root, lock):
    """Authenticate historical producers by pinned bank, not current hashes."""
    validate_pins()
    root = Path(root)
    records, files, initials = [], {}, {}
    for replay in (1, 2):
        for phase in ('initial', 'continuation'):
            area = _record_root(root, replay, phase)
            request, request_hash, _ = read_bound_json(area / 'request.json')
            tools.validate_request(request, lock)
            require(request['protocol_sha256'] == STAGE1_PROTOCOL_SHA256
                    and request['phase'] == phase and request['replay'] == replay, 'Wrong Stage-1 episode')
            report = read_json(area / 'worker.json')
            require(report.get('request_sha256') == request_hash and report.get('parent') == parent_identity(lock)
                    and report.get('protocol_sha256') == STAGE1_PROTOCOL_SHA256,
                    'Stage-1 request/parent differs')
            evidence.verify_common(area, tools.MODE, report, lock=lock, source_hashes=STAGE1_SOURCES)
            evidence.verify_observation(area, report, tools.observer_request(request))
            if phase == 'initial':
                # No recursion and no current-source comparison in INITIAL phase.
                tools.verify_tool_evidence(area, report, request, lock)
                saved = evidence.bound_document(area, report['tool_analysis'])
                require(saved['classification'] == 'expected_initial_call', 'Invalid spontaneous initial call')
                initials[replay] = {'request': request, 'report': report, 'analysis': saved, 'root': area}
            else:
                capsule = initials[replay]
                expected_files = {name: sha256(capsule['root'] / name) for name in tools.capsule_names(replay)}
                require(request['episode_files'] == expected_files and report.get('episode_files') == expected_files,
                        'Continuation is not bound to the actual initial capsule')
                require(report.get('mode') == tools.MODE and report.get('question') == tools.QUESTION
                        and report.get('fixture_sha256') == tools.FIXTURE_SHA256
                        and report.get('phase') == phase and report.get('replay') == replay
                        and report.get('protocol_sha256') == STAGE1_PROTOCOL_SHA256
                        and report.get('training') is False and report.get('patch_applied') is False
                        and report.get('quality_improvement_claimed') is False, 'Invalid historical continuation scope')
                row = report['results'][0]
                proof = evidence.bound_document(area, row['prompt_proof'])
                require(proof.get('messages') == tools.messages_for(request, capsule)
                        and proof.get('tools') == tools.CASE['tools'] and proof.get('input_ids') == row['input_ids']
                        and proof.get('harness_call_id_tokenization_invariant') is True
                        and proof.get('silent_truncation') is False and proof.get('thinking') is False
                        and proof.get('template_sha256') == lock['model']['files']['chat_template.jinja']['sha256'],
                        'Historical continuation tokenization proof differs')
                saved = evidence.bound_document(area, report['tool_analysis'])
                require(tools.semantic_analysis(saved) == tools.semantic_analysis(tools.analyze(row, phase)),
                        'Historical classification differs from raw output')
            row = report['results'][0]
            require(row.get('eos_token_ids') == [248044, 248046] and row.get('input_tokens') == len(row['input_ids'])
                    and len(row['input_ids']) <= 512, 'Historical chat-stop/input profile differs')
            name = tools.case_id(replay, phase)
            names = {'request.json', 'worker.json', 'baseline-verification.json', 'loaded-parent-verification.json',
                     name + '-generation.json', name + '-prefill.npz', name + '-prompt.json', 'tool-analysis.json'}
            for filename in sorted(names):
                path = area / filename
                require(path.is_file() and not path.is_symlink(), 'Unsafe historical artifact')
                files[str(path.relative_to(root))] = sha256(path)
            records.append({'replay': replay, 'phase': phase, 'worker': report, 'analysis': saved})
    gate = tools.reproduction_gate(records)
    require(gate['status'] == 'BF16_duplicate_call_reproduced' and gate['activation_diagnosis_admitted'] is True,
            'Stage 1 did not pass; Stage 2 is NOT_RUN')
    manifest = {'producer_commit': STAGE1_COMMIT, 'protocol_sha256': STAGE1_PROTOCOL_SHA256,
                'source_sha256': dict(STAGE1_SOURCES), 'files': files}
    return {'manifest': manifest, 'manifest_sha256': digest(manifest), 'gate': gate,
            'initial': initials[1], 'baseline': records[1]['worker']['results'][0],
            'baseline_root': _record_root(root, 1, 'continuation'),
            'parameters': records[0]['worker']['loaded_parameter_sha256_before']}


def validate_protocol(protocol, lock):
    exact_keys(protocol, ('schema_version', 'role', 'question', 'stage1_protocol_sha256',
                         'stage1_producer_commit', 'stage1_manifest_sha256', 'lock_sha256', 'fixture_sha256',
                         'controls', 'generation', 'eos_token_ids', 'common_producer_sha256',
                         'specificity_producer_sha256', 'execution_policy'), 'specificity protocol')
    require(type(protocol['schema_version']) is int and protocol['schema_version'] == 1
            and protocol['role'] == 'practice' and protocol['question'] == QUESTION,
            'Only exposed Stage-2 practice is admitted')
    require(protocol['stage1_protocol_sha256'] == STAGE1_PROTOCOL_SHA256
            and protocol['stage1_producer_commit'] == STAGE1_COMMIT and protocol['lock_sha256'] == lock['_sha256']
            and protocol['fixture_sha256'] == tools.FIXTURE_SHA256, 'Wrong parent/history/fixture')
    require(type(protocol['generation']) is dict and protocol['controls'] == list(CONTROLS) and protocol['generation'] == PROFILE
            and all(type(protocol['generation'][key]) is type(value) for key, value in PROFILE.items())
            and protocol['eos_token_ids'] == [248044, 248046]
            and all(type(x) is int for x in protocol['eos_token_ids']), 'Wrong fixed controls/profile/EOS')
    hash_string(protocol['stage1_manifest_sha256'])
    require(protocol['common_producer_sha256'] == evidence.SOURCE_AT_IMPORT
            and protocol['specificity_producer_sha256'] == SOURCE_SHA256, 'Wrong frozen Stage-2 producers')
    require(canonical(protocol['execution_policy']) == canonical(EXECUTION_POLICY), 'Frozen count/stop/resource policy differs')


def protocol_document(stage1, lock):
    """Host convenience after eligibility and source freeze; does not write files."""
    require(stage1['gate']['status'] == 'BF16_duplicate_call_reproduced', 'Stage 1 must pass first')
    return {'schema_version': 1, 'role': 'practice', 'question': QUESTION,
            'stage1_protocol_sha256': STAGE1_PROTOCOL_SHA256, 'stage1_producer_commit': STAGE1_COMMIT,
            'stage1_manifest_sha256': stage1['manifest_sha256'], 'lock_sha256': lock['_sha256'],
            'fixture_sha256': tools.FIXTURE_SHA256, 'controls': list(CONTROLS), 'generation': dict(PROFILE),
            'eos_token_ids': [248044, 248046], 'common_producer_sha256': dict(evidence.SOURCE_AT_IMPORT),
            'specificity_producer_sha256': SOURCE_SHA256, 'execution_policy': copy.deepcopy(EXECUTION_POLICY)}


def load_protocol(path, expected_sha256, lock):
    hash_string(expected_sha256)
    protocol, actual, raw = read_bound_json(Path(path))
    require(actual == expected_sha256, 'Stage-2 protocol differs from host-pinned hash')
    validate_protocol(protocol, lock)
    return protocol, raw


def messages_for(control, initial, *, padding=0):
    require(control in CONTROLS and type(padding) is int and 0 <= padding < 32, 'Invalid fixed control/padding')
    request = {**initial['request'], 'phase': 'continuation'}
    messages = tools.messages_for(request, initial)
    if control in ('C', 'N'):
        cue = COMPLETION_CUE if control == 'C' else NEUTRAL_CUE
        suffix = (' ' + ' '.join(['recorded'] * padding) + '.') if padding else ''
        messages[1]['content'] += ' ' + cue + suffix
    else:
        require(padding == 0, 'Padding only belongs to C/N')
    if control == 'S': messages[-1]['content'] = SUCCESS_RESULT
    if control == 'P': messages[1]['content'] = PENDING_PROMPT
    return messages


def render(tokenizer, messages):
    options = {'tools': tools.CASE['tools'], 'add_generation_prompt': True, 'enable_thinking': False}
    text = tokenizer.apply_chat_template(messages, tokenize=False, **options)
    ids = _ids(tokenizer.apply_chat_template(messages, tokenize=True, **options))
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = [list(pair) for pair in encoded['offset_mapping']]
    require(ids == list(encoded['input_ids']) == tokenizer.encode(text, add_special_tokens=False)
            and len(offsets) == len(ids), 'BPE/template/offset tokenization disagreement')
    require(all(type(x) is int and x >= 0 for x in ids), 'Invalid tokenizer IDs')
    require(all(len(pair) == 2 and all(type(x) is int for x in pair) and 0 <= pair[0] <= pair[1] <= len(text)
                for pair in offsets), 'Invalid character offsets')
    return {'messages': messages, 'tools': tools.CASE['tools'], 'rendered': text, 'input_ids': ids,
            'offset_mapping': offsets, 'template_sha256': sha_bytes(tokenizer.chat_template.encode()),
            'thinking': False, 'silent_truncation': False}


def _data_position(proof, span, label):
    start, end = span
    require(start < end, 'Empty observation data span')
    matches = [i for i, (left, right) in enumerate(proof['offset_mapping']) if left <= end - 1 < right]
    require(len(matches) == 1, 'Data character has no unique BPE token: ' + label)
    index = matches[0]; left, right = proof['offset_mapping'][index]
    # BPE may absorb adjacent whitespace, but never another control/data field.
    require(not proof['rendered'][left:start].strip() and not proof['rendered'][end:right].strip(),
            'Observation token crosses a non-whitespace data boundary: ' + label)
    return {'position': index, 'token_id': proof['input_ids'][index], 'offsets': [left, right],
            'data_span': [start, end], 'data_text': proof['rendered'][start:end],
            'selected_character': end - 1, 'selection': 'token covering final data character'}


def positions(proof):
    text = proof['rendered']; result = proof['messages'][-1]['content']
    require(text.count(result) == 1, 'Tool result is not uniquely rendered')
    result_start = text.index(result)
    open_result = text.rfind('<tool_response>', 0, result_start)
    close_result = text.find('</tool_response>', result_start)
    require(open_result >= 0 and close_result >= result_start + len(result), 'Tool-result framing is missing')
    # The official tools preamble itself documents <tool_call>; restrict this
    # observation to the actual completed assistant turn, never schema examples.
    assistant = text.rfind('<|im_start|>assistant\n', 0, result_start)
    require(assistant >= 0, 'Prior assistant framing is absent')
    assistant_end = text.find('<|im_end|>', assistant)
    require(assistant_end > assistant and assistant_end < result_start, 'Prior assistant turn is not complete')
    start = text.find('<tool_call>', assistant, assistant_end)
    stop = text.find('</tool_call>', start, assistant_end)
    require(start >= 0 and stop > start and text.find('<tool_call>', start + 1, assistant_end) < 0,
            'Expected one actual prior assistant call')
    target = text.rfind('parser', start, stop)
    require(target >= 0, 'Prior-call argument data is absent')
    final = len(proof['input_ids']) - 1
    require(final >= 0, 'Empty assistant prompt')
    return {'assistant_prompt': {'position': final, 'token_id': proof['input_ids'][final],
                                'offsets': proof['offset_mapping'][final], 'selection': 'last model input token'},
            'tool_result_data': _data_position(proof, [result_start, result_start + len(result)], 'tool result'),
            'prior_call_data': _data_position(proof, [target, target + len('parser')], 'prior call')}


def _compile(tokenizer, stage1, protocol, protocol_sha256, lock):
    require(stage1['manifest_sha256'] == protocol['stage1_manifest_sha256'], 'Stage-1 manifest differs')
    cache = {control: [render(tokenizer, messages_for(control, stage1['initial'], padding=count))
                       for count in range(32)] for control in ('C', 'N')}
    match = next(((c, n) for c in range(32) for n in range(32)
                  if len(cache['C'][c]['input_ids']) == len(cache['N'][n]['input_ids']) <= 512), None)
    require(match is not None, 'STOP_semantic_specificity: no fixed bounded neutral length match')
    cases = []
    for control in CONTROLS:
        padding = match[0] if control == 'C' else match[1] if control == 'N' else 0
        proof = cache[control][padding] if control in cache else render(tokenizer, messages_for(control, stage1['initial']))
        require(1 <= len(proof['input_ids']) <= 512, 'Fixed control exceeds input budget; no truncation')
        require(proof['template_sha256'] == lock['model']['files']['chat_template.jinja']['sha256'],
                'Tokenizer template differs from authenticated official template')
        altered = copy.deepcopy(proof['messages'])
        altered[-2]['tool_calls'][0]['id'] = 'call_id_invariance_control'
        altered[-1]['tool_call_id'] = 'call_id_invariance_control'
        require(render(tokenizer, altered)['input_ids'] == proof['input_ids'], 'Harness call ID changes input tokens')
        proof.update(control=control, padding=padding, positions=positions(proof),
                     harness_call_id_tokenization_invariant=True)
        cases.append(proof)
    require(cases[0]['input_ids'] == stage1['baseline']['input_ids'], 'New B does not exactly replay Stage-1 input')
    return {'schema_version': 1, 'role': 'practice', 'question': QUESTION, 'protocol_sha256': protocol_sha256,
            'stage1_manifest': stage1['manifest'], 'stage1_manifest_sha256': stage1['manifest_sha256'],
            'parent_parameter_sha256': stage1['parameters'], 'generation': dict(PROFILE),
            'specificity_producer_sha256': SOURCE_SHA256, 'common_producer_sha256': dict(evidence.SOURCE_AT_IMPORT),
            'padding_match': {'C': match[0], 'N': match[1], 'count_domain': list(range(32)),
                              'order': 'lexicographic C count then N count, first exact full-input match',
                              'lengths': {c: [len(p['input_ids']) for p in cache[c]] for c in ('C', 'N')}},
            'cases': cases, 'weights_edited': False, 'quality_evaluation': 'NOT_RUN'}


def compile_conditions(tokenizer, stage1_root, protocol_path, expected_protocol_sha256, lock):
    protocol, _ = load_protocol(protocol_path, expected_protocol_sha256, lock)
    stage1 = load_stage1(stage1_root, lock)  # MUST precede every tokenizer operation.
    return _compile(tokenizer, stage1, protocol, expected_protocol_sha256, lock)


def validate_request(request, protocol, protocol_sha256, lock):
    exact_keys(request, ('schema_version', 'role', 'question', 'control', 'lock_sha256', 'protocol_sha256',
                         'conditions_sha256', 'stage1_manifest_sha256', 'generation'), 'specificity request')
    validate_protocol(protocol, lock)
    require(type(request['schema_version']) is int and request['schema_version'] == 1
            and request['role'] == 'practice' and request['question'] == QUESTION
            and request['control'] in CONTROLS and type(request['control']) is str,
            'Invalid fixed Stage-2 practice request')
    require(request['protocol_sha256'] == protocol_sha256 and request['lock_sha256'] == lock['_sha256']
            and request['stage1_manifest_sha256'] == protocol['stage1_manifest_sha256']
            and type(request['generation']) is dict and request['generation'] == PROFILE
            and all(type(request['generation'][key]) is type(value) for key, value in PROFILE.items()),
            'Wrong specificity profile/identity')
    hash_string(request['conditions_sha256'])


def validate_conditions(conditions, request, protocol, stage1):
    exact_keys(conditions, ('schema_version', 'role', 'question', 'protocol_sha256', 'stage1_manifest',
                           'stage1_manifest_sha256', 'parent_parameter_sha256', 'generation',
                           'specificity_producer_sha256', 'common_producer_sha256', 'padding_match',
                           'cases', 'weights_edited', 'quality_evaluation'), 'prepared conditions')
    require(type(conditions['schema_version']) is int and conditions['schema_version'] == 1
            and conditions['role'] == 'practice' and conditions['question'] == QUESTION
            and conditions['protocol_sha256'] == request['protocol_sha256']
            and conditions['stage1_manifest'] == stage1['manifest']
            and conditions['stage1_manifest_sha256'] == stage1['manifest_sha256']
            and conditions['parent_parameter_sha256'] == stage1['parameters']
            and conditions['generation'] == PROFILE
            and conditions['specificity_producer_sha256'] == SOURCE_SHA256
            and conditions['common_producer_sha256'] == protocol['common_producer_sha256']
            and conditions['weights_edited'] is False and conditions['quality_evaluation'] == 'NOT_RUN',
            'Prepared conditions identity/scope differs')
    evidence.validate_parameter_map(conditions['parent_parameter_sha256'])
    exact_keys(conditions['padding_match'], ('C', 'N', 'count_domain', 'order', 'lengths'), 'padding proof')
    require(conditions['padding_match']['count_domain'] == list(range(32)), 'Padding search domain changed')
    for control in ('C', 'N'):
        count = conditions['padding_match'][control]
        require(type(count) is int and 0 <= count < 32, 'Invalid padding count')
    require(type(conditions['cases']) is list and len(conditions['cases']) == 5, 'All five prepared controls required')
    for control, case in zip(CONTROLS, conditions['cases']):
        exact_keys(case, ('messages', 'tools', 'rendered', 'input_ids', 'offset_mapping', 'template_sha256',
                          'thinking', 'silent_truncation', 'control', 'padding', 'positions',
                          'harness_call_id_tokenization_invariant'), 'prepared case')
        padding = conditions['padding_match'][control] if control in ('C', 'N') else 0
        require(case['control'] == control and case['padding'] == padding
                and case['messages'] == messages_for(control, stage1['initial'], padding=padding)
                and case['tools'] == tools.CASE['tools'] and case['thinking'] is False
                and case['silent_truncation'] is False and case['harness_call_id_tokenization_invariant'] is True,
                'Prepared control messages differ')
        require(type(case['rendered']) is str and len(case['rendered']) <= 65536
                and type(case['input_ids']) is list and 1 <= len(case['input_ids']) <= 512
                and all(type(x) is int and x >= 0 for x in case['input_ids'])
                and type(case['offset_mapping']) is list and len(case['offset_mapping']) == len(case['input_ids']),
                'Invalid prepared tokenization')
        require(all(type(pair) is list and len(pair) == 2 and all(type(x) is int for x in pair)
                    and 0 <= pair[0] <= pair[1] <= len(case['rendered']) for pair in case['offset_mapping']),
                'Invalid prepared offsets')
        require(case['positions'] == positions(case), 'Prepared observation positions differ')


def _strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result: raise ValueError('duplicate key')
            result[key] = value
        return result
    def bad(_): raise ValueError('nonfinite')
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=bad)


def _answer(value, control):
    expected = {'exit_code': 0, 'failed': 0} if control == 'S' else {'exit_code': 1, 'failed': 2}
    return type(value) is dict and set(value) == set(expected) and all(type(v) is int for v in value.values()) and value == expected


def analyze(row, control):
    require(control in CONTROLS, 'Unknown semantic control')
    base = {'role': 'practice', 'control': control, 'quality_evaluation': 'NOT_RUN', 'model_tool_calls_executed': False}
    if row.get('censored') is not False:
        return {**base, 'classification': 'STOP_censored', 'parser_agreement': True, 'passed': False, 'parsed_message': None}
    raw = row.get('text_with_special_tokens')
    require(type(raw) is str and len(raw) <= 65536, 'Missing/oversized raw response')
    if not tools.single_assistant_turn(raw):
        return {**base, 'classification': 'STOP_harness', 'parser_agreement': True, 'passed': False, 'parsed_message': None}
    try:
        message = parse_completion(raw, tools.CASE['tools'])
        calls = message.get('tool_calls', [])
        call = len(calls) == 1 and calls[0]['function'] == {'name': 'run_tests', 'arguments': canonical({'target': 'parser'})} and message.get('content') is None
        if call: primary = 'parser_call'
        elif calls: primary = 'other_tool_action'
        else:
            try: primary = 'correct_final_json' if _answer(loads(message.get('content', '')), control) else 'other_answer'
            except (ValueError, TypeError): primary = 'other_answer'
    except (ValueError, TypeError, KeyError):
        primary, message = 'STOP_parser', None
    # Existing independently implemented XML/JSON tool grammar remains unchanged.
    independent = tools.independent_classification(row, 'continuation')
    independent = 'parser_call' if independent == 'repeat_same_call' else independent
    if independent in ('correct_final_json', 'other_answer'):
        body = raw.strip()
        if body.startswith('<think>'):
            body = body.split('</think>', 1)[1].strip() if '</think>' in body else ''
        elif body.startswith('</think>'): body = body[len('</think>'):].strip()
        body = re.sub(r'<\|(?:im_end|endoftext)\|>\s*$', '', body).rstrip()
        try: independent = 'correct_final_json' if _answer(_strict_json(body), control) else 'other_answer'
        except (ValueError, TypeError): independent = 'other_answer'
    agree = primary == independent
    expected = 'correct_final_json' if control in ('S', 'C') else 'parser_call'
    return {**base, 'primary_classification': primary, 'independent_classification': independent,
            'parser_agreement': agree, 'classification': primary if agree else 'STOP_harness',
            'passed': agree and primary == expected, 'parsed_message': message,
            'call_ids': 'harness-assigned UUIDs; not model-generated'}


def _publish_arrays(root, name, arrays):
    path = root / name
    with path.open('xb') as handle: np.savez(handle, **arrays)
    reference = evidence.artifact(path)
    evidence.numeric_arrays(root, reference)
    return reference


def observe(request, protocol, conditions, stage1, lock, verified, root, protocol_sha256):
    """Trusted runtime boundary; imports Torch only inside the admitted worker."""
    import torch
    started = time.monotonic()
    model, tokenizer, loading, before = forward.load_text_model(lock, verified)
    arrays = {name: {} for name in POSITIONS}; hooks = []
    report = {'schema_version': 1, 'mode': MODE, 'role': 'practice', 'question': QUESTION,
              'control': request['control'], 'protocol_sha256': protocol_sha256, 'parent': parent_identity(lock),
              'specificity_producer_sha256': SOURCE_SHA256, 'conditions_sha256': request['conditions_sha256'],
              'stage1_manifest_sha256': stage1['manifest_sha256'], 'generation': dict(PROFILE),
              'training': False, 'weights_edited': False, 'patch_applied': False, 'quality_improvement_claimed': False,
              'quality_evaluation': 'NOT_RUN', 'complete': False, 'results': [], 'loading_info': loading,
              'loaded_parameter_sha256_before': before, 'parameter_postcheck_complete': False, **evidence.bindings(root)}
    error = None
    try:
        evidence.validate_parameter_map(before)
        require(before == stage1['parameters'], 'Runtime parent differs from admitted original Stage 1')
        publish_json({**report}, root / 'loaded-parent-verification.json')
        report['loaded_parent_proof'] = evidence.artifact(root / 'loaded-parent-verification.json')
        compiled = _compile(tokenizer, stage1, protocol, protocol_sha256, lock)
        require(compiled == conditions, 'Prepared conditions differ from fresh authenticated tokenization')
        proof = next(case for case in compiled['cases'] if case['control'] == request['control'])
        name = 'specificity-' + request['control'].lower()
        publish_json(proof, root / (name + '-prompt.json'))
        ids = torch.tensor([proof['input_ids']], dtype=torch.long)
        eos = forward.chat_stop_ids(model, tokenizer)
        require(eos == protocol['eos_token_ids'], 'Effective EOS differs from frozen chat-stop union')
        report['effective_generation_arguments'] = {**PROFILE, 'eos_token_id': eos, 'use_cache': True,
                                                    'logits_to_keep': 1, 'pad_token_id': tokenizer.eos_token_id}
        publish_json(report['effective_generation_arguments'], root / 'generation-arguments.json')
        report['generation_arguments_proof'] = evidence.artifact(root / 'generation-arguments.json')
        torch.manual_seed(PROFILE['seed'])
        def capture(layer):
            def hook(_module, args, output):
                key = f'layer_{layer}_input_bf16'
                if key in arrays['assistant_prompt']: return
                require(not torch.is_grad_enabled(), 'Read-only collection requires inference mode')
                for position_name in POSITIONS:
                    index = proof['positions'][position_name]['position']
                    for kind, value, width in (('input', args[0], 9216), ('output', output, 2560)):
                        require(value.dtype == torch.bfloat16 and value.ndim == 3 and value.shape[0] == 1
                                and value.shape[1] == ids.shape[1] and value.shape[2] == width,
                                'Unexpected prefill activation shape/dtype')
                        words = value[0, index].detach().contiguous().view(torch.uint16).numpy().copy()
                        require(not np.any((words & 0x7f80) == 0x7f80), 'Nonfinite BF16 observation')
                        arrays[position_name][f'layer_{layer}_{kind}_bf16'] = words
                # No return: the hook never changes input or output.
            return hook
        require(len(model.model.layers) == 32, 'Wrong language layer count')
        for layer, block in enumerate(model.model.layers):
            hooks.append(block.mlp.down_proj.register_forward_hook(capture(layer)))
        with torch.inference_mode(), (root / (name + '-token-journal.jsonl')).open('x') as journal:
            output = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=False,
                                    max_new_tokens=96, use_cache=True, logits_to_keep=1,
                                    pad_token_id=tokenizer.eos_token_id, eos_token_id=eos,
                                    streamer=forward.TokenJournal(journal))
        generated = output[0, ids.shape[1]:].tolist()
        row = {'id': name, 'input_ids': proof['input_ids'], 'input_tokens': ids.shape[1],
               'output_ids': generated, 'generated_tokens': len(generated),
               'text': tokenizer.decode(generated, skip_special_tokens=True),
               'text_with_special_tokens': tokenizer.decode(generated, skip_special_tokens=False),
               **forward.completion_status(generated, 96, model, tokenizer),
               'prompt_proof': evidence.artifact(root / (name + '-prompt.json'))}
        references = {position: _publish_arrays(root, name + '-' + position + '.npz', arrays[position]) for position in POSITIONS}
        row.update(observation=references['assistant_prompt']['path'],
                   observation_sha256=references['assistant_prompt']['sha256'], position_observations=references)
        if request['control'] == 'B':
            require(row['input_ids'] == stage1['baseline']['input_ids']
                    and row['output_ids'] == stage1['baseline']['output_ids'], 'Fresh B exact token replay failed')
            original = evidence.numeric_arrays(stage1['baseline_root'], {'path': stage1['baseline']['observation'],
                                                  'sha256': stage1['baseline']['observation_sha256']})
            require(all(np.array_equal(arrays['assistant_prompt'][key], value) for key, value in original.items()),
                    'Fresh B 64-vector replay differs')
            row['baseline_replay_bit_identical'] = True
        publish_json(row, root / (name + '-generation.json'))
        row['generation_sha256'] = sha256(root / (name + '-generation.json'))
        report['results'] = [row]
        analysis = analyze(row, request['control'])
        publish_json(analysis, root / 'specificity-analysis.json')
        report['specificity_analysis'] = evidence.artifact(root / 'specificity-analysis.json')
        report['complete'] = True
    except BaseException as caught:
        error = caught; report['error'] = type(caught).__name__ + ': ' + str(caught)
    finally:
        signal.alarm(450)  # Preserve the postcheck/exit reserve after the active alarm.
        for hook in hooks:
            try: hook.remove()
            except BaseException as caught:
                report['hook_cleanup_error'] = type(caught).__name__ + ': ' + str(caught)
                if error is None: error = caught
        try:
            for position, values in arrays.items():
                if values and not (root / ('specificity-' + request['control'].lower() + '-' + position + '.npz')).exists():
                    with (root / ('partial-' + position + '.npz')).open('xb') as handle: np.savez(handle, **values)
                    report.setdefault('partial_observations', {})[position] = evidence.artifact(root / ('partial-' + position + '.npz'))
        except BaseException as caught:
            report['partial_save_error'] = type(caught).__name__ + ': ' + str(caught)
            if error is None: error = caught
        try:
            after = {name: forward.parameter_sha256(model.get_parameter(name)) for name in before}
            report['loaded_parameter_sha256_after'] = after
            evidence.validate_parameter_map(after)
            require(after == before, 'Model parameters changed during read-only specificity collection')
            report['parameters_unchanged'] = True
            assert_baseline_unchanged(verified)
            report['parameter_postcheck_complete'] = True
        except BaseException as caught:
            report['postcheck_error'] = type(caught).__name__ + ': ' + str(caught)
            if error is None: error = caught
        if error is not None: report['complete'] = False
        report['seconds'] = time.monotonic() - started
        publish_json(report, root / 'observation-result.json')
    if error is not None: raise error
    return report


def verify_specificity_evidence(root, report, request, protocol, protocol_sha256, conditions, stage1, lock):
    """Host verifier; frozen sources/protocol/conditions must come from host pins."""
    validate_request(request, protocol, protocol_sha256, lock)
    validate_conditions(conditions, request, protocol, stage1)
    require(report.get('mode') == MODE and report.get('role') == 'practice' and report.get('question') == QUESTION
            and report.get('parent') == parent_identity(lock) and report.get('control') == request['control']
            and report.get('protocol_sha256') == protocol_sha256 and report.get('conditions_sha256') == request['conditions_sha256']
            and report.get('specificity_producer_sha256') == protocol['specificity_producer_sha256']
            and report.get('stage1_manifest_sha256') == protocol['stage1_manifest_sha256']
            and report.get('training') is False and report.get('weights_edited') is False
            and report.get('patch_applied') is False and report.get('quality_evaluation') == 'NOT_RUN',
            'Specificity scope/identity differs')
    require(report.get('request_sha256') == sha256(root / 'request.json')
            and sha256(root / 'conditions.json') == request['conditions_sha256']
            and sha256(root / 'protocol.json') == protocol_sha256, 'Specificity saved input bindings differ')
    arguments = evidence.bound_document(root, report['generation_arguments_proof'])
    require(arguments == report.get('effective_generation_arguments') and arguments.get('eos_token_id') == protocol['eos_token_ids']
            and all(type(arguments.get(key)) is type(value) and arguments[key] == value for key, value in PROFILE.items())
            and arguments.get('use_cache') is True and arguments.get('logits_to_keep') == 1
            and type(arguments.get('pad_token_id')) is int and arguments['pad_token_id'] in protocol['eos_token_ids'],
            'Effective generation arguments differ')
    evidence.verify_common(root, MODE, report, lock=lock, source_hashes=protocol['common_producer_sha256'])
    loaded = evidence.bound_document(root, report['loaded_parent_proof'])
    require(loaded.get('specificity_producer_sha256') == protocol['specificity_producer_sha256']
            and loaded.get('conditions_sha256') == request['conditions_sha256'], 'Loaded-parent specificity binding differs')
    control = request['control']; name = 'specificity-' + control.lower()
    evidence.verify_observation(root, report, {'generation': PROFILE, 'cases': [{'id': name}]})
    row = report['results'][0]
    require(row['eos_token_ids'] == protocol['eos_token_ids'], 'Specificity EOS differs')
    proof = evidence.bound_document(root, row['prompt_proof'])
    require(proof == next(case for case in conditions['cases'] if case['control'] == control)
            and proof['input_ids'] == row['input_ids']
            and proof['template_sha256'] == lock['model']['files']['chat_template.jinja']['sha256'],
            'Specificity prompt differs from frozen conditions or official template')
    exact_keys(row['position_observations'], POSITIONS, 'position observations')
    for reference in row['position_observations'].values(): evidence.numeric_arrays(root, reference)
    require(row['position_observations']['assistant_prompt'] == {'path': row['observation'], 'sha256': row['observation_sha256']},
            'Main observation is not the assistant decision')
    if control == 'B':
        require(row['input_ids'] == stage1['baseline']['input_ids'] and row['output_ids'] == stage1['baseline']['output_ids']
                and row.get('baseline_replay_bit_identical') is True, 'B exact replay failed')
        original = evidence.numeric_arrays(stage1['baseline_root'], {'path': stage1['baseline']['observation'],
                                               'sha256': stage1['baseline']['observation_sha256']})
        current = evidence.numeric_arrays(root, row['position_observations']['assistant_prompt'])
        require(all(np.array_equal(current[key], value) for key, value in original.items()), 'B observed vectors differ')
    saved = evidence.bound_document(root, report['specificity_analysis'])
    require(tools.semantic_analysis(saved) == tools.semantic_analysis(analyze(row, control)), 'Specificity classification differs')
    return saved


def specificity_gate(records):
    """Run all five completed semantic controls; failures irreversibly reject.

    The host stops subsequent invocations only on infrastructure, source,
    integrity, censoring or budget failure, never on an ordinary completed
    semantic failure. This frozen denominator policy is not adaptive.
    """
    require(type(records) is list and len(records) <= 5, 'Invalid specificity ledger')
    controls = [r.get('control') for r in records]
    require(len(set(controls)) == len(controls) and all(c in CONTROLS for c in controls), 'Duplicate/unknown specificity control')
    result = {'complete': len(records) == 5, 'generation_invocations': len(records), 'weights_edited': False,
              'weight_candidate_eligible': False, 'quality_evaluation': 'NOT_RUN', 'localization_admitted': False}
    for record in records:
        worker = record.get('worker')
        require(type(worker) is dict, 'Gate needs verified worker records, not cached boolean scores')
        if worker.get('complete') is not True or worker.get('parameter_postcheck_complete') is not True \
                or worker.get('parameters_unchanged') is not True:
            return {**result, 'status': 'STOP_inconclusive'}
        evidence.validate_parameter_map(worker.get('loaded_parameter_sha256_before'))
        require(worker['loaded_parameter_sha256_before'] == worker.get('loaded_parameter_sha256_after'),
                'Specificity parent changed')
        analysis = record['analysis']
        require(tools.semantic_analysis(analysis) == tools.semantic_analysis(analyze(worker['results'][0], record['control'])),
                'Cached specificity analysis differs from raw output')
        if analysis.get('classification', '').startswith('STOP') or analysis.get('parser_agreement') is not True:
            return {**result, 'status': 'STOP_harness' if analysis.get('classification') != 'STOP_censored' else 'STOP_inconclusive'}
        if analysis.get('passed') is not True: return {**result, 'status': 'STOP_semantic_specificity'}
    if len(records) != 5: return {**result, 'status': 'NOT_RUN_incomplete'}
    require(all(r['worker']['loaded_parameter_sha256_before'] == records[0]['worker']['loaded_parameter_sha256_before']
                for r in records), 'Different original parents across specificity controls')
    return {**result, 'status': 'semantic_specificity_controls_passed', 'localization_admitted': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('request', 'conditions', 'stage1', 'protocol', 'output'): parser.add_argument('--' + field, type=Path, required=True)
    parser.add_argument('--protocol-sha256', required=True)
    args = parser.parse_args()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU isolation required')
    require(not os.path.lexists(args.output), 'Refusing an existing result')
    root = args.output.parent
    require(root.is_dir(), 'Host must create fresh task-owned output directory')
    def expired(_signum, _frame): raise TimeoutError('Specificity worker active limit reached')
    signal.signal(signal.SIGALRM, expired); signal.alarm(1250)
    report = {'complete': False, 'weights_edited': False, 'quality_evaluation': 'NOT_RUN'}
    try:
        lock = load_lock(); protocol, protocol_raw = load_protocol(args.protocol, args.protocol_sha256, lock)
        request, request_hash, request_raw = read_bound_json(args.request)
        validate_request(request, protocol, args.protocol_sha256, lock)
        conditions, conditions_hash, conditions_raw = read_bound_json(args.conditions)
        require(conditions_hash == request['conditions_sha256'], 'Prepared conditions hash differs')
        stage1 = load_stage1(args.stage1, lock)
        require(stage1['manifest_sha256'] == request['stage1_manifest_sha256'], 'Admitted Stage-1 evidence changed')
        validate_conditions(conditions, request, protocol, stage1)
        for name, raw in (('request.json', request_raw), ('conditions.json', conditions_raw), ('protocol.json', protocol_raw)):
            with (root / name).open('xb') as handle: handle.write(raw)
        verified = verify_baseline(lock, Path('/model'))
        publish_json(verified, root / 'baseline-verification.json')
        report = observe(request, protocol, conditions, stage1, lock, verified, root, args.protocol_sha256)
        report['request_sha256'] = request_hash
    except BaseException as error:
        if (root / 'observation-result.json').is_file(): report.update(read_json(root / 'observation-result.json'))
        report.update(complete=False, error=type(error).__name__ + ': ' + str(error))
        raise
    finally:
        publish_json(report, args.output); signal.alarm(0)


if __name__ == '__main__': main()
