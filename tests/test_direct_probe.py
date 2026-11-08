"""Decision probe contracts. No real model, GPU, grader or holdout is used."""
from __future__ import annotations

import copy
from collections import UserDict
import importlib.util
import math
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import direct_probe as p
from direct_weights import ValidationError, load_lock, read_json


class Tokenizer:
    """Reversible tokenizer with the real leading-space operator merge hazard."""
    merges = {' >=': 4001, ' >': 4002, ' end:': 4003}
    inverse = {value: key for key, value in merges.items()}
    pad_token_id = 0
    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        result = []
        while text:
            for phrase, token in self.merges.items():
                if text.startswith(phrase):
                    result.append(token); text = text[len(phrase):]
                    break
            else:
                result.append(ord(text[0]) + 100); text = text[1:]
        return result

    def decode(self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        return ''.join(self.inverse[token] if token in self.inverse else chr(token - 100)
                       for token in ids if not (skip_special_tokens and token == 0))

    def apply_chat_template(self, messages, tokenize, **kwargs):
        text = '<user>' + messages[0]['content'] + '</user><assistant>'
        return self.encode(text) if tokenize else text


def fixture():
    tokenizer = Tokenizer()
    prompt = 'Implement half-open intersection. Empty intervals are permitted.'
    text = ('```python\ndef wsp_intersection(a, b):\n'
            '    """Reversed intervals: start >= end."""\n'
            '    def validate_interval(interval):\n'
            '        start, end = interval\n'
            '        if start >= end:\n'
            '            raise ValueError()\n```')
    prefix = text[:text.rindex('>=')]
    patient = {'id': 'half_open_overlap-r1', 'text': text, 'hit_token_limit': False,
               'input_ids': tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=True),
               'output_ids': tokenizer.encode(text)}
    patient['generated_tokens'] = len(patient['output_ids'])
    lock = load_lock()
    source = {'role': 'practice', 'lock_sha256': lock['_sha256'],
              'cases': [{'id': patient['id'], 'messages': [{'role': 'user', 'content': prompt}]}]}
    request = {'schema_version': 1, 'role': 'practice', 'question': p.QUESTION,
               'lock_sha256': lock['_sha256'], 'protocol_sha256': 'a' * 64,
               'patient_sha256': 'b' * 64, 'source_request_sha256': 'c' * 64,
               'original_prompt': prompt, 'assistant_prefix': prefix,
               'semantic_cue': 'Equal endpoints mean valid empty intervals; only start > end is reversed.',
               'operators': dict(p.OPERATORS), 'profile': dict(p.PROFILE), 'arms': [],
               'budget': {'max_forward_calls': 2, 'max_wall_seconds': 2300,
                          'postcheck_reserve_seconds': 120, 'controls_admitted': False}}
    return tokenizer, request, patient, source, lock


def first_assertion_fixture():
    tokenizer, request, patient, source, lock = fixture()
    text = ('```python\ndef wsp_intersection(a, b):\n'
            '    """\n    Raises:\n'
            '        ValueError: If any input interval is reversed (start >= end).\n'
            '    """\n    start, end = a\n'
            '    if start >= end:\n        raise ValueError()\n```')
    patient.update(text=text, output_ids=tokenizer.encode(text))
    patient['generated_tokens'] = len(patient['output_ids'])
    cue = 'Equal endpoints are valid empty intervals; reversed input means strictly larger start.'
    cue += ' ' * (len(p.LENGTH_CONTROL) - len(cue))
    request.update(schema_version=2, question=p.FIRST_ASSERTION_QUESTION,
                   assistant_prefix=text[:text.index('>=')], semantic_cue=cue,
                   length_control=p.LENGTH_CONTROL, operators=dict(p.FIRST_ASSERTION_OPERATORS))
    request['budget'].update(max_forward_calls=3, max_wall_seconds=1700)
    return tokenizer, request, patient, source, lock


class ProbeContracts(unittest.TestCase):
    def setUp(self):
        self.tokenizer, self.request, self.patient, self.source, self.lock = fixture()

    def validate(self, request=None, patient=None, source=None):
        p.validate_request(self.request if request is None else request, self.lock,
                           self.patient if patient is None else patient,
                           self.source if source is None else source,
                           patient_sha256='b' * 64, source_request_sha256='c' * 64)

    def test_initial_profile_and_empty_controls_are_accepted(self):
        self.validate()

    def test_tokenizer_mapping_wrappers_are_supported_and_require_input_ids(self):
        self.assertEqual(p._ids(UserDict({'input_ids': [11, 22]})), [11, 22])
        self.assertEqual(p._ids(UserDict({'input_ids': np.asarray([[11, 22]], dtype=np.int64)})), [11, 22])
        with self.assertRaisesRegex(ValidationError, 'lacks input_ids'):
            p._ids(UserDict({'attention_mask': [1, 1]}))
        with self.assertRaises(ValidationError):
            p._ids(UserDict({'input_ids': [True]}))

    def test_prepare_context_accepts_batchencoding_style_mapping(self):
        class MappingTokenizer(Tokenizer):
            def apply_chat_template(self, messages, tokenize, **kwargs):
                result = super().apply_chat_template(messages, tokenize=tokenize, **kwargs)
                return UserDict({'input_ids': result}) if tokenize else result
        plan = p.prepare_context(MappingTokenizer(), self.request, self.patient, 'baseline')
        self.assertEqual(plan['candidate_ids']['incorrect'], [4001, 4003])

    def test_parent_source_and_protocol_identities_are_strict(self):
        for field, value in [('role', 'holdout'), ('question', 'other'), ('schema_version', True),
                             ('lock_sha256', '0' * 64), ('patient_sha256', 'd' * 64),
                             ('source_request_sha256', 'd' * 64), ('protocol_sha256', 'bad')]:
            request = copy.deepcopy(self.request); request[field] = value
            with self.subTest(field=field), self.assertRaises(ValidationError): self.validate(request)

    def test_only_original_completed_exposed_patient_is_accepted(self):
        changes = [lambda x: x.update(hit_token_limit=True), lambda x: x.update(id='other'),
                   lambda x: x.update(generated_tokens=1), lambda x: x.update(input_ids=[True]),
                   lambda x: x.update(output_ids=[-1])]
        for change in changes:
            patient = copy.deepcopy(self.patient); change(patient)
            with self.subTest(change=change), self.assertRaises(ValidationError): self.validate(patient=patient)
        request = copy.deepcopy(self.request); request['original_prompt'] += 'changed'
        with self.assertRaises(ValidationError): self.validate(request)

    def test_actual_guard_is_required_instead_of_docstring_comparison(self):
        request = copy.deepcopy(self.request)
        request['assistant_prefix'] = self.patient['text'][:self.patient['text'].index('>=')]
        with self.assertRaisesRegex(ValidationError, 'actual'): self.validate(request)

    def test_unknown_executable_patch_and_arbitrary_profiles_are_rejected(self):
        for field in ('code', 'patch', 'training', 'python', 'holdout'):
            request = copy.deepcopy(self.request); request[field] = 'data'
            with self.subTest(field=field), self.assertRaises(ValidationError): self.validate(request)
        for field, value in [('max_input_tokens', 512.0), ('threads', 2), ('thinking', 0), ('seed', True)]:
            request = copy.deepcopy(self.request); request['profile'][field] = value
            with self.subTest(field=field), self.assertRaises(ValidationError): self.validate(request)
        request = copy.deepcopy(self.request); request['operators']['correct'] = '<'
        with self.assertRaises(ValidationError): self.validate(request)

    def test_only_frozen_four_groups_and_strengths_are_accepted(self):
        request = copy.deepcopy(self.request)
        request['arms'] = [{'id': 'group0-half', 'source': 'cue', 'group': 0, 'strength': 0.5}]
        request['budget'].update(controls_admitted=True, max_forward_calls=3)
        self.validate(request)
        for field, value in [('group', 4), ('group', True), ('strength', 0.25), ('strength', True),
                             ('strength', math.nan), ('source', 'fitted-direction'), ('id', '../other')]:
            changed = copy.deepcopy(request); changed['arms'][0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValidationError): self.validate(changed)
        for field, value in [('controls_admitted', False), ('max_forward_calls', 2)]:
            changed = copy.deepcopy(request); changed['budget'][field] = value
            with self.subTest(field=field), self.assertRaises(ValidationError): self.validate(changed)
        changed = copy.deepcopy(request); changed['arms'] *= 2; changed['budget']['max_forward_calls'] = 4
        with self.assertRaises(ValidationError): self.validate(changed)

    def test_budget_types_and_upper_bounds_are_enforced(self):
        for field, value in [('max_forward_calls', True), ('max_forward_calls', 19),
                             ('max_wall_seconds', 2400), ('postcheck_reserve_seconds', 0),
                             ('controls_admitted', 1)]:
            request = copy.deepcopy(self.request); request['budget'][field] = value
            with self.subTest(field=field), self.assertRaises(ValidationError): self.validate(request)

    def test_bpe_boundary_preserves_actual_patient_ids_and_records_space(self):
        plan = p.prepare_context(self.tokenizer, self.request, self.patient, 'baseline')
        boundary = plan['tokenization']['patient_output_prefix_tokens']
        self.assertEqual(plan['prefix_ids'], self.patient['input_ids'] + self.patient['output_ids'][:boundary])
        self.assertEqual(plan['tokenization']['prefix_suffix_in_scored_continuation'], ' ')
        self.assertEqual(plan['candidate_ids']['incorrect'], [4001, 4003])
        self.assertEqual(plan['candidate_ids']['correct'], [4002, 4003])
        self.assertEqual(plan['decision_position'], len(plan['prefix_ids']) - 1)
        self.assertTrue(plan['tokenization']['original_prompt_ids_match_patient'])

    def test_cue_changes_only_prompt_and_preserves_assistant_prefix_ids(self):
        base = p.prepare_context(self.tokenizer, self.request, self.patient, 'baseline')
        cue = p.prepare_context(self.tokenizer, self.request, self.patient, 'cue')
        count = base['tokenization']['patient_output_prefix_tokens']
        self.assertEqual(base['prefix_ids'][-count:], cue['prefix_ids'][-count:])
        self.assertEqual(base['candidate_ids'], cue['candidate_ids'])
        self.assertNotEqual(base['prefix_ids'], cue['prefix_ids'])

    def test_changed_patient_prompt_ids_or_text_are_rejected(self):
        patient = copy.deepcopy(self.patient); patient['input_ids'][0] += 1
        with self.assertRaisesRegex(ValidationError, 'prompt token IDs'):
            p.prepare_context(self.tokenizer, self.request, patient, 'baseline')
        patient = copy.deepcopy(self.patient); patient['output_ids'][0] += 1
        with self.assertRaisesRegex(ValidationError, 'actual generated'):
            p.prepare_context(self.tokenizer, self.request, patient, 'baseline')

    def test_no_silent_truncation_for_oversized_cue_context(self):
        request = copy.deepcopy(self.request); request['semantic_cue'] = 'x' * 1000
        with self.assertRaisesRegex(ValidationError, 'no silent truncation'):
            p.prepare_context(self.tokenizer, request, self.patient, 'cue')

    def test_canonical_retokenization_difference_is_reported_not_substituted(self):
        class DifferentCanonical(Tokenizer):
            def encode(self, text, add_special_tokens=False):
                # Alternative tokenization with identical text only in a full chat.
                ids = super().encode(text, add_special_tokens=add_special_tokens)
                if '<assistant>```' in text:
                    for index, token in enumerate(ids):
                        if token == 4001:
                            return ids[:index] + [ord(c) + 100 for c in ' >='] + ids[index + 1:]
                return ids
        tokenizer = DifferentCanonical()
        plan = p.prepare_context(tokenizer, self.request, self.patient, 'baseline')
        self.assertFalse(plan['tokenization']['branches']['incorrect']['canonical_matches_replay'])
        self.assertEqual(plan['candidate_ids']['incorrect'], [4001, 4003])

    def test_logprob_margin_is_stable_and_uses_all_sequence_tokens(self):
        logits = np.zeros((2, 2, 6), dtype=np.float32)
        logits[:, 0, 1] = 3
        logits[0, 1, 3] = 2
        logits[1, 1, 4] = -2
        score = p.score_logits(logits, {'correct': [1, 3], 'incorrect': [2, 4]})
        self.assertGreater(score['correct_minus_incorrect_logprob'], 3)
        self.assertEqual(score['next_token_id'], 1)
        self.assertEqual(score['preferred_scored_branch'], 'correct')
        np.testing.assert_allclose(
            p.score_logits(logits + 10000, {'correct': [1, 3], 'incorrect': [2, 4]})['correct_minus_incorrect_logprob'],
            score['correct_minus_incorrect_logprob'], rtol=1e-9)

    def test_nonfinite_logits_invalid_ids_or_broken_shared_prefix_are_rejected(self):
        for mode in ('nonfinite', 'bounds', 'different-prefix'):
            logits = np.zeros((2, 2, 6), dtype=np.float32)
            candidates = {'correct': [1, 3], 'incorrect': [2, 4]}
            if mode == 'nonfinite': logits[0, 1, 0] = math.nan
            if mode == 'bounds': candidates['correct'][0] = 6
            if mode == 'different-prefix': logits[1, 0, 0] = 1
            with self.subTest(mode=mode), self.assertRaises(ValidationError): p.score_logits(logits, candidates)

    def test_forward_admission_reserves_postcheck_and_uses_actual_timing(self):
        budget = self.request['budget']
        self.assertTrue(p.forward_admitted(budget, 300, [], 0))
        self.assertTrue(p.forward_admitted(budget, 800, [500], 1))
        self.assertFalse(p.forward_admitted(budget, 1600, [500], 1))
        self.assertFalse(p.forward_admitted(budget, 100, [1, 1], 2))

    def test_numeric_npz_publication_is_non_overwriting_and_no_pickle(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'obs.npz'
            arrays = {'layer_0_input_bf16': np.asarray([0x3f80], dtype='<u2')}
            p.publish_arrays(arrays, output)
            with np.load(output, allow_pickle=False) as archive:
                np.testing.assert_array_equal(archive['layer_0_input_bf16'], arrays['layer_0_input_bf16'])
            with self.assertRaises(FileExistsError): p.publish_arrays(arrays, output)
            arrays['layer_0_input_bf16'] = np.asarray(['code'], dtype=object)
            with self.assertRaises(ValidationError): p.publish_arrays(arrays, Path(directory) / 'bad.npz')

    def test_complete_run_persists_bound_result_and_all_parameter_postcheck(self):
        model = SimpleNamespace(get_parameter=lambda name: name)
        before = {'model.parameter-a': 'd' * 64, 'protected.parameter-b': 'e' * 64}
        def forward(_model, _tokenizer, plan, outdir, name, **kwargs):
            return {'id': name, 'status': 'completed', 'seconds': 0.01}, {}
        with tempfile.TemporaryDirectory() as directory, \
             mock.patch.object(p, 'load_text_model', return_value=(model, self.tokenizer, {}, before)), \
             mock.patch.object(p, 'parameter_sha256', side_effect=lambda name: before[name]), \
             mock.patch.object(p, 'run_forward', side_effect=forward):
            outdir = Path(directory)
            result = p.run_probe(self.request, self.lock, {}, outdir, self.patient, self.source,
                                 request_sha256='f' * 64, patient_sha256='b' * 64,
                                 source_request_sha256='c' * 64)
            stored = read_json(outdir / 'result.json')
            self.assertEqual(stored, result)
            self.assertTrue(result['complete'])
            self.assertTrue(result['parameters_unchanged'])
            self.assertTrue(result['parameter_postcheck_complete'])
            self.assertEqual(result['loaded_parameter_sha256_before'], result['loaded_parameter_sha256_after'])
            self.assertEqual(result['quality_evaluation'], 'NOT_RUN')
            self.assertFalse(result['weights_edited'])
            self.assertEqual(result['patient_sha256'], 'b' * 64)
            self.assertEqual(result['source_request_sha256'], 'c' * 64)

    def test_interruption_still_rechecks_parameters_and_preserves_incomplete_result(self):
        model = SimpleNamespace(get_parameter=lambda name: name)
        before = {'parameter': 'd' * 64}
        with tempfile.TemporaryDirectory() as directory, \
             mock.patch.object(p, 'load_text_model', return_value=(model, self.tokenizer, {}, before)), \
             mock.patch.object(p, 'parameter_sha256', return_value='d' * 64) as postcheck, \
             mock.patch.object(p, 'run_forward', side_effect=TimeoutError('own time budget')):
            outdir = Path(directory)
            with self.assertRaises(TimeoutError):
                p.run_probe(self.request, self.lock, {}, outdir, self.patient, self.source,
                            request_sha256='f' * 64, patient_sha256='b' * 64,
                            source_request_sha256='c' * 64)
            result = read_json(outdir / 'result.json')
            self.assertFalse(result['complete'])
            self.assertTrue(result['parameters_unchanged'])
            self.assertIn('TimeoutError', result['error'])
            postcheck.assert_called_once_with('parameter')

    def test_parameter_mutation_fails_closed_even_when_forward_returns(self):
        model = SimpleNamespace(get_parameter=lambda name: name)
        before = {'parameter': 'd' * 64}
        with tempfile.TemporaryDirectory() as directory, \
             mock.patch.object(p, 'load_text_model', return_value=(model, self.tokenizer, {}, before)), \
             mock.patch.object(p, 'parameter_sha256', return_value='0' * 64), \
             mock.patch.object(p, 'run_forward', return_value=({'id': 'baseline', 'status': 'completed',
                                                              'seconds': 0.01}, {})):
            with self.assertRaisesRegex(ValidationError, 'parameter bytes changed'):
                p.run_probe(self.request, self.lock, {}, Path(directory), self.patient, self.source,
                            request_sha256='f' * 64, patient_sha256='b' * 64,
                            source_request_sha256='c' * 64)
            result = read_json(Path(directory) / 'result.json')
            self.assertFalse(result['complete'])
            self.assertFalse(result['parameters_unchanged'])
            self.assertFalse(result['parameter_postcheck_complete'])

    def test_inadmissible_forward_budget_is_incomplete_and_does_not_run(self):
        model = SimpleNamespace(get_parameter=lambda name: name)
        before = {'parameter': 'd' * 64}
        with tempfile.TemporaryDirectory() as directory, \
             mock.patch.object(p, 'load_text_model', return_value=(model, self.tokenizer, {}, before)), \
             mock.patch.object(p, 'parameter_sha256', return_value='d' * 64), \
             mock.patch.object(p, 'forward_admitted', return_value=False), \
             mock.patch.object(p, 'run_forward') as forward:
            result = p.run_probe(self.request, self.lock, {}, Path(directory), self.patient, self.source,
                                 request_sha256='f' * 64, patient_sha256='b' * 64,
                                 source_request_sha256='c' * 64)
            forward.assert_not_called()
            self.assertFalse(result['complete'])
            self.assertTrue(all(row['status'] == 'NOT_RUN_budget' for row in result['arms']))
            self.assertTrue(result['parameters_unchanged'])


class FirstAssertionContracts(unittest.TestCase):
    def setUp(self):
        self.tokenizer, self.request, self.patient, self.source, self.lock = first_assertion_fixture()

    def validate(self, request=None, patient=None, source=None):
        p.validate_request(self.request if request is None else request, self.lock,
                           self.patient if patient is None else patient,
                           self.source if source is None else source,
                           patient_sha256='b' * 64, source_request_sha256='c' * 64)

    def test_v2_exact_schema_first_assertion_and_three_contexts(self):
        self.validate()
        self.assertEqual(p.required_contexts(self.request), ('baseline', 'cue', 'length_control'))
        self.assertTrue(self.patient['text'].startswith(self.request['assistant_prefix'] + '>= end).'))

    def test_v2_rejects_later_guard_other_question_suffix_or_new_control(self):
        for field, value in [('question', p.QUESTION), ('length_control', 'another note'),
                             ('assistant_prefix', self.patient['text'][:self.patient['text'].rindex('>=')]),
                             ('operators', dict(p.OPERATORS))]:
            request = copy.deepcopy(self.request); request[field] = value
            with self.subTest(field=field), self.assertRaises(ValidationError): self.validate(request)
        request = copy.deepcopy(self.request); request.pop('length_control')
        with self.assertRaises(ValidationError): self.validate(request)

    def test_v2_requires_exactly_three_calls_no_arms_and_1700_seconds(self):
        for field, value in [('max_forward_calls', 2), ('max_forward_calls', 4),
                             ('max_wall_seconds', 1701), ('controls_admitted', True)]:
            request = copy.deepcopy(self.request); request['budget'][field] = value
            with self.subTest(field=field), self.assertRaises(ValidationError): self.validate(request)
        request = copy.deepcopy(self.request)
        request['arms'] = [{'id': 'not-yet', 'source': 'cue', 'group': 0, 'strength': 1}]
        with self.assertRaises(ValidationError): self.validate(request)

    def test_v2_source_binding_remains_required(self):
        source = copy.deepcopy(self.source); source['cases'][0]['messages'][0]['content'] += 'changed'
        with self.assertRaises(ValidationError): self.validate(source=source)
        request = copy.deepcopy(self.request); request['source_request_sha256'] = '0' * 64
        with self.assertRaises(ValidationError): self.validate(request)

    def test_v2_plans_preserve_first_assertion_tokens_and_length_matching(self):
        plans = p.prepare_plans(self.tokenizer, self.request, self.patient)
        self.assertEqual(set(plans), {'baseline', 'cue', 'length_control'})
        cue, neutral = plans['cue'], plans['length_control']
        self.assertEqual(cue['decision_position'], neutral['decision_position'])
        self.assertEqual(cue['max_forward_input_tokens'], neutral['max_forward_input_tokens'])
        self.assertEqual(cue['candidate_ids'], neutral['candidate_ids'])
        boundary = plans['baseline']['tokenization']['patient_output_prefix_tokens']
        self.assertEqual(plans['baseline']['prefix_ids'],
                         self.patient['input_ids'] + self.patient['output_ids'][:boundary])

    def test_v2_token_length_mismatch_fails_before_any_forward(self):
        request = copy.deepcopy(self.request); request['semantic_cue'] = 'Short semantic cue.'
        self.validate(request)
        with self.assertRaisesRegex(ValidationError, 'exact token length'):
            p.prepare_plans(self.tokenizer, request, self.patient)

    @staticmethod
    def scores(plan, name, *, baseline_correct=False):
        correct = name == 'cue' or (name == 'baseline' and baseline_correct)
        return {'id': name, 'status': 'completed', 'seconds': 0.01,
                'branches': {label: {'ids': plan['candidate_ids'][label]} for label in p.LABELS},
                'next_token_id': plan['candidate_ids']['correct' if correct else 'incorrect'][0],
                'correct_minus_incorrect_logprob': 0.4 if correct else -5.5 if name == 'baseline' else -5.4}

    def run_fake(self, directory, *, baseline_correct=False, admitted=None):
        model = SimpleNamespace(get_parameter=lambda name: name)
        before = {'protected.parameter': 'd' * 64}
        def forward(_model, _tokenizer, plan, outdir, name, **kwargs):
            return self.scores(plan, name, baseline_correct=baseline_correct), {}
        with mock.patch.object(p, 'load_text_model', return_value=(model, self.tokenizer, {}, before)), \
             mock.patch.object(p, 'parameter_sha256', return_value='d' * 64), \
             mock.patch.object(p, 'run_forward', side_effect=forward) as called:
            if admitted is None:
                result = p.run_probe(self.request, self.lock, {}, directory, self.patient, self.source,
                                     request_sha256='f' * 64, patient_sha256='b' * 64,
                                     source_request_sha256='c' * 64)
            else:
                with mock.patch.object(p, 'forward_admitted', side_effect=admitted):
                    result = p.run_probe(self.request, self.lock, {}, directory, self.patient, self.source,
                                         request_sha256='f' * 64, patient_sha256='b' * 64,
                                         source_request_sha256='c' * 64)
            return result, called.call_count

    def test_v2_complete_requires_three_contexts_and_persists_question_and_stop_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            result, calls = self.run_fake(Path(directory))
            self.assertEqual(calls, 3)
            self.assertTrue(result['complete'])
            self.assertTrue(result['parameters_unchanged'])
            self.assertEqual(result['schema_version'], 2)
            self.assertEqual(result['question'], p.FIRST_ASSERTION_QUESTION)
            self.assertEqual(result['diagnostic_gate']['status'], 'diagnostic_gate_passed')
            self.assertFalse(result['diagnostic_gate']['weight_candidate_eligible'])
            self.assertFalse(result['diagnostic_gate']['activation_screening_admitted'])
            self.assertEqual(read_json(Path(directory) / 'result.json'), result)

    def test_v2_correct_baseline_stops_without_contrast_or_weight_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            result, calls = self.run_fake(Path(directory), baseline_correct=True)
            self.assertEqual(calls, 1)
            self.assertFalse(result['complete'])
            self.assertTrue(result['parameters_unchanged'])
            self.assertEqual(result['diagnostic_gate']['status'], 'STOP_localization')
            self.assertIn('arithmetic replay', result['diagnostic_gate']['reason'])
            self.assertTrue(all(row['status'] == 'NOT_RUN_replay_mismatch' for row in result['arms'][1:]))

    def test_v2_missing_length_control_result_is_incomplete(self):
        with tempfile.TemporaryDirectory() as directory:
            result, calls = self.run_fake(Path(directory), admitted=[True, True, False])
            self.assertEqual(calls, 2)
            self.assertFalse(result['complete'])
            self.assertEqual(result['diagnostic_gate']['status'], 'NOT_RUN_incomplete')

    def test_v2_neutral_explained_or_absent_semantic_effect_stops_screening(self):
        plans = p.prepare_plans(self.tokenizer, self.request, self.patient)
        rows = [self.scores(plans[name], name) for name in p.required_contexts(self.request)]
        for field, value in [('correct_minus_incorrect_logprob', -2.0),
                             ('next_token_id', plans['length_control']['candidate_ids']['correct'][0])]:
            changed = copy.deepcopy(rows); changed[2][field] = value
            with self.subTest(field=field):
                self.assertEqual(p.first_assertion_gate(changed)['status'], 'STOP_localization')
        rows[1]['correct_minus_incorrect_logprob'] = -1
        self.assertEqual(p.first_assertion_gate(rows)['status'], 'STOP_localization')


TORCH_ENABLED = os.environ.get('WSP_TEST_DIRECT_PROBE_TORCH') == '1' and importlib.util.find_spec('torch')


@unittest.skipUnless(TORCH_ENABLED, 'explicit tiny Torch container verification required')
class TinyForwardControls(unittest.TestCase):
    """Small 32-layer CPU model; no pretrained weights or actual 4B execution."""
    def test_hooks_controls_and_parameter_integrity(self):
        import torch
        torch.set_num_threads(1)
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(16, 2)
                self.model = torch.nn.Module()
                self.model.layers = torch.nn.ModuleList()
                for _ in range(32):
                    block = torch.nn.Module(); block.mlp = torch.nn.Module()
                    block.mlp.down_proj = torch.nn.Linear(3, 2, bias=False)
                    self.model.layers.append(block)
                self.head = torch.nn.Linear(2, 16, bias=False)
            def forward(self, input_ids, attention_mask, use_cache, logits_to_keep, return_dict):
                self.assertions = (not torch.is_grad_enabled(), use_cache, logits_to_keep)
                hidden = self.embedding(input_ids)
                for layer in self.model.layers:
                    features = torch.cat((hidden, torch.ones_like(hidden[..., :1])), dim=-1)
                    hidden = hidden + layer.mlp.down_proj(features) * 0.05
                return SimpleNamespace(logits=self.head(hidden)[:, -logits_to_keep:, :].float())
        torch.manual_seed(1)
        model = Tiny().to(torch.bfloat16).eval()
        for parameter in model.parameters(): parameter.requires_grad_(False)
        before = {name: p.parameter_sha256(value) for name, value in model.named_parameters()}
        plan = {'prefix_ids': [1, 2], 'candidate_ids': {'correct': [3, 4], 'incorrect': [5, 4]},
                'decision_position': 1, 'context': 'baseline'}
        tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=0,
                                    decode=lambda ids, **kwargs: 'token:' + str(ids[0]))
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(p, 'INPUT_WIDTH', 3), \
             mock.patch.object(p, 'OUTPUT_WIDTH', 2):
            outdir = Path(directory)
            base, arrays = p.run_forward(model, tokenizer, plan, outdir, 'baseline')
            for context in ('cue', 'length_control'):
                other = {**plan, 'context': context}
                score, _ = p.run_forward(model, tokenizer, other, outdir, context)
                self.assertEqual(score['context'], context)
                self.assertEqual(score['branches'], base['branches'])
                self.assertEqual(read_json(outdir / (context + '-score.json'))['context'], context)
            donors = {'self': arrays, 'cue': {key: value.copy() for key, value in arrays.items()}}
            # Change cue's declared output activations only; model parameters remain untouched.
            for key in donors['cue']:
                if '_output_' in key:
                    donors['cue'][key] = np.full(2, 0x3f80, dtype='<u2')
            for source, strength in [('self', 1), ('cue', 0), ('cue', 0.5), ('cue', 1)]:
                name = source + '-' + str(strength).replace('.', '_')
                arm = {'id': name, 'source': source, 'group': 0, 'strength': strength}
                score, observed = p.run_forward(model, tokenizer, plan, outdir, name,
                                               arm=arm, donors=donors)
                self.assertEqual(len(score['intervention']), 8)
                if source == 'self' or strength == 0:
                    self.assertEqual(score['branches'], base['branches'])
                if strength == 1 and source == 'cue':
                    np.testing.assert_array_equal(observed['layer_0_applied_output_bf16'],
                                                  donors['cue']['layer_0_output_bf16'])
            self.assertEqual(model.assertions, (True, False, 2))
            self.assertTrue(all(not layer.mlp.down_proj._forward_hooks for layer in model.model.layers))
        after = {name: p.parameter_sha256(value) for name, value in model.named_parameters()}
        self.assertEqual(before, after)


if __name__ == '__main__':
    unittest.main()
