"""Finite activation experiment controls; no pretrained model is executed."""
from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'tests')]
import direct_causal as c
import direct_probe as p
from direct_weights import ValidationError, read_json
from test_direct_probe import first_assertion_fixture


def stats(group=0, norm=1.0, count=100):
    return [{'layer': layer, 'delta_l2': norm, 'changed_entries': count, 'max_absolute_delta': 0.1}
            for layer in range(group * 8, group * 8 + 8)]


def score(name, plan, value, *, correct=None, group=0):
    good = value > 0 if correct is None else correct
    branches = {}
    for label in p.LABELS:
        ids = plan['candidate_ids'][label]
        total = -10 + value if label == 'correct' else -10
        branches[label] = {'ids': ids, 'token_logprobs': [total / len(ids)] * len(ids),
                           'sequence_logprob': total}
    return {'id': name, 'status': 'completed', 'seconds': 0.01, 'branches': branches,
            'next_token_id': plan['candidate_ids']['correct' if good else 'incorrect'][0],
            'correct_minus_incorrect_logprob': value, 'actual_interventions': stats(group)}


def bundle(root):
    tokenizer, original, patient, source, lock = first_assertion_fixture()
    plans = p.prepare_plans(tokenizer, original, patient)
    arrays = {context: {name: np.full(9216 if '_input_' in name else 2560,
                                     0x3f81 if context == 'cue' else 0x3f80, dtype='<u2')
                        for name in c.ARRAY_NAMES} for context in ('cue', 'length_control')}
    for context, values in arrays.items():
        with (root / (context + '-decision.npz')).open('xb') as handle:
            np.savez(handle, **values)
    parameters = {name: 'd' * 64 for name in c.evidence.expected_parameter_names()}
    runtime = {'torch': 'test-torch', 'transformers': 'test-transformers', 'numpy': 'test-numpy',
               'cpu_reference_kernels': {'kind': 'test-pinned-reference'},
               'probe_sha256': c.sha256(Path(p.__file__)),
               'direct_forward_sha256': c.sha256(Path(c.direct_forward.__file__)),
               'provenance': {'torch_observation_sha256': 'e' * 64, 'environment_lock_sha256': 'f' * 64,
                              'scope': 'Historical Torch observed separately; Transformers/NumPy controller pins; '
                                       'no historical binary closure'}}
    rows = [score('baseline', plans['baseline'], -4), score('cue', plans['cue'], 6),
            score('length_control', plans['length_control'], -3.8)]
    for row in rows[1:]:
        row['observation_sha256'] = c.sha256(root / (row['id'] + '-decision.npz'))
    result = {'schema_version': 2, 'question': p.FIRST_ASSERTION_QUESTION, 'role': 'practice',
              'status': 'practice_only', 'complete': True, 'parameters_unchanged': True,
              'parameter_postcheck_complete': True, 'weights_edited': False, 'quality_evaluation': 'NOT_RUN',
              'parent': c.parent_identity(lock), 'lock_sha256': lock['_sha256'],
              'patient_sha256': 'b' * 64, 'source_request_sha256': 'c' * 64,
              'protocol_sha256': original['protocol_sha256'], 'request_sha256': 'a' * 64,
              'loaded_parameter_sha256_before': parameters, 'loaded_parameter_sha256_after': parameters,
              'loading_info': {'cpu_reference_kernels': runtime['cpu_reference_kernels']},
              'arms': rows, 'diagnostic_gate': p.first_assertion_gate(rows)}
    documents = {'result.json': result, 'runtime-identity.json': runtime, 'tokenization-proof.json': plans,
                 'request-proof.json': {'request': original, 'request_sha256': 'a' * 64,
                                        'patient_sha256': 'b' * 64, 'source_request_sha256': 'c' * 64}}
    for name, document in documents.items():
        (root / name).write_text(json.dumps(document))
    request = {'schema_version': 1, 'role': 'practice', 'question': c.QUESTION, 'stage': 1,
               'lock_sha256': lock['_sha256'], 'protocol_sha256': '9' * 64,
               'patient_sha256': 'b' * 64, 'source_request_sha256': 'c' * 64,
               'donor_files': {name: c.sha256(root / name) for name in c.DONOR_NAMES},
               'winner_group': None, 'stage1_result_sha256': None,
               'budget': {'max_forward_calls': 6, 'max_wall_seconds': 1700, 'postcheck_reserve_seconds': 120}}
    return tokenizer, request, patient, source, lock, result, arrays, runtime


class CausalContracts(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(); self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name); self.donors = self.root / 'donors'; self.donors.mkdir()
        (self.tokenizer, self.request, self.patient, self.source, self.lock, self.previous,
         self.arrays, self.runtime) = bundle(self.donors)

    def validate(self, request=None):
        c.validate_request(self.request if request is None else request, self.lock,
                           patient_sha256='b' * 64, source_request_sha256='c' * 64)

    def test_request_has_exact_finite_stages_and_no_external_vectors_or_programs(self):
        self.validate()
        for key, value in [('role', 'holdout'), ('stage', True), ('winner_group', 0),
                           ('patient_sha256', '0' * 64), ('code', 'print(1)'), ('delta', [1])]:
            request = copy.deepcopy(self.request); request[key] = value
            with self.subTest(key=key), self.assertRaises(ValidationError): self.validate(request)
        for key, value in [('max_forward_calls', 7), ('max_wall_seconds', 1800), ('postcheck_reserve_seconds', 0)]:
            request = copy.deepcopy(self.request); request['budget'][key] = value
            with self.subTest(key=key), self.assertRaises(ValidationError): self.validate(request)

    def test_numeric_bundle_authentication_and_parameter_ancestry(self):
        loaded = c.load_donors(self.request, self.lock, self.donors, self.patient, self.source)
        self.assertEqual(len(loaded['cue']), 64)
        self.assertEqual(len(loaded['result']['loaded_parameter_sha256_before']), 426)
        path = self.donors / 'result.json'; path.write_text('{}')
        with self.assertRaisesRegex(ValidationError, 'hash mismatch'):
            c.load_donors(self.request, self.lock, self.donors, self.patient, self.source)

    def test_changed_donor_parameters_rejected_even_with_updated_artifact_hash(self):
        result = copy.deepcopy(self.previous); result['loaded_parameter_sha256_after'] = {}
        path = self.donors / 'result.json'; path.write_text(json.dumps(result))
        self.request['donor_files']['result.json'] = c.sha256(path)
        with self.assertRaisesRegex(ValidationError, 'parameters were changed'):
            c.load_donors(self.request, self.lock, self.donors, self.patient, self.source)

    def test_npz_rejects_pickle_duplicates_compression_shape_and_nonfinite_words(self):
        for mode in ('pickle', 'compressed', 'shape', 'nonfinite', 'duplicate'):
            path = self.root / (mode + '.npz')
            arrays = {key: value.copy() for key, value in self.arrays['cue'].items()}
            if mode == 'pickle': arrays['layer_0_output_bf16'] = np.asarray([object()], dtype=object)
            if mode == 'shape': arrays['layer_0_output_bf16'] = np.zeros((1, 2560), dtype='<u2')
            if mode == 'nonfinite': arrays['layer_0_output_bf16'][0] = 0x7f80
            with path.open('xb') as handle:
                (np.savez_compressed if mode == 'compressed' else np.savez)(handle, **arrays)
            if mode == 'duplicate':
                with zipfile.ZipFile(path, 'a') as archive:
                    archive.writestr('layer_0_output_bf16.npy', b'invalid')
            with self.subTest(mode=mode), self.assertRaises((ValidationError, ValueError)):
                c.load_numeric_npz(path)

    def test_frozen_delta_and_two_sign_controls_preserve_precast_magnitudes(self):
        deltas = c.frozen_deltas(self.arrays['cue'], self.arrays['length_control'])
        np.testing.assert_array_equal(deltas[0], np.full(2560, 0.0078125, dtype=np.float32))
        first = c.sign_randomized(deltas, 20261108)
        second = c.sign_randomized(deltas, 20261109)
        for layer in range(32):
            np.testing.assert_array_equal(np.abs(first[layer]), np.abs(deltas[layer]))
        self.assertFalse(np.array_equal(first[0], second[0]))
        with self.assertRaises(ValidationError): c.sign_randomized(deltas, 123)

    def test_norm_and_support_matching_is_per_layer_and_cannot_accept_empty_controls(self):
        self.assertTrue(c.matched(stats(), stats()))
        self.assertFalse(c.matched([], []))
        self.assertFalse(c.matched(stats(norm=1.051), stats()))
        self.assertFalse(c.matched(stats(count=106), stats()))
        self.assertFalse(c.matched(stats(group=1), stats()))
        before = np.array([0x3f80, 0x3f80], dtype='<u2')
        after = np.array([0x3f81, 0x3f80], dtype='<u2')
        self.assertEqual(c.delta_stats(before, after)['changed_entries'], 1)

    def test_replay_requires_per_token_scores_and_all_vectors_bit_identical(self):
        row = self.previous['arms'][1]
        c.check_replay(row, self.arrays['cue'], row, self.arrays['cue'])
        changed = {key: value.copy() for key, value in self.arrays['cue'].items()}
        changed['layer_31_input_bf16'][0] += 1
        with self.assertRaisesRegex(ValidationError, 'bit-identical'):
            c.check_replay(row, changed, row, self.arrays['cue'])
        other = copy.deepcopy(row); other['branches']['correct']['token_logprobs'][0] += 0.01
        with self.assertRaisesRegex(ValidationError, 'log probabilities'):
            c.check_replay(other, self.arrays['cue'], row, self.arrays['cue'])

    def test_fixed_call_lists_no_adaptive_layers_or_seeds(self):
        self.assertEqual(len(c.specifications(1)), 6)
        controls = c.specifications(2, 2)
        self.assertEqual(len(controls), 8)
        self.assertEqual([x['seed'] for x in controls if x.get('mode') == 'random'], [20261108, 20261109])
        self.assertEqual([x['group'] for x in controls if 'group' in x], [2] * 6)

    def screen_rows(self):
        plan = p.prepare_plans(self.tokenizer, read_json(self.donors / 'request-proof.json')['request'],
                               self.patient)['length_control']
        return [score('neutral-replay', plan, -3.8), score('no-op', plan, -3.8)] + [
            score(f'group-{g}', plan, 1.0 if g in (1, 2) else -2, group=g) for g in range(4)]

    def test_screen_selects_one_qualified_group_with_frozen_tie_break_and_can_stop(self):
        rows = self.screen_rows()
        self.assertEqual(c.diagnostic_gate(1, rows)['winner_group'], 1)
        rows[4]['correct_minus_incorrect_logprob'] += 0.0000005
        self.assertEqual(c.diagnostic_gate(1, rows)['winner_group'], 1)
        for row in rows[2:]: row['next_token_id'] = row['branches']['incorrect']['ids'][0]
        gate = c.diagnostic_gate(1, rows)
        self.assertEqual(gate['status'], 'STOP_no_group')
        self.assertFalse(gate['weight_candidate_eligible'])

    def test_complete_controls_fail_on_unmatched_random_offtarget_reverse_or_dose(self):
        plan = p.prepare_plans(self.tokenizer, read_json(self.donors / 'request-proof.json')['request'],
                               self.patient)['length_control']
        values = [-3.8, 1, -1, -3.7, -3.6, -3.5, 6, -1]
        rows = [score(spec['id'], plan, value, group=1) for spec, value in zip(c.specifications(2, 1), values)]
        self.assertEqual(c.diagnostic_gate(2, rows, 1)['status'], 'activation_controls_passed')
        for index, field, value in [(3, 'actual_interventions', stats(group=1, norm=2)),
                                    (5, 'correct_minus_incorrect_logprob', 0),
                                    (7, 'correct_minus_incorrect_logprob', 1),
                                    (2, 'correct_minus_incorrect_logprob', 2)]:
            changed = copy.deepcopy(rows); changed[index][field] = value
            with self.subTest(index=index):
                self.assertEqual(c.diagnostic_gate(2, changed, 1)['status'], 'STOP_controls_failed')
        self.assertEqual(c.diagnostic_gate(2, rows[:-1], 1)['status'], 'NOT_RUN_incomplete')

    def run_fake(self, output, *, mismatch=False, failure=None, mutation=False, admitted=True):
        loaded = c.load_donors(self.request, self.lock, self.donors, self.patient, self.source)
        model = SimpleNamespace(get_parameter=lambda name: name)
        before = loaded['result']['loaded_parameter_sha256_before']
        plans = loaded['plans']
        def forward(_model, _tokenizer, plan, outdir, spec, deltas, off_position):
            if failure: raise failure
            name = spec['id']
            if name in ('neutral-replay', 'no-op'):
                result = copy.deepcopy(loaded['scores']['length_control']); result['id'] = name
            elif name == 'semantic-replay':
                result = copy.deepcopy(loaded['scores']['cue']); result['id'] = name
            elif self.request['stage'] == 2:
                value = {'full': 1, 'half': -1, 'off-target': -3.7, 'reverse': -1}.get(name, -3.7)
                result = score(name, plan, value, group=self.request['winner_group'])
            else:
                result = score(name, plan, 1 if spec['group'] == 1 else -2, group=spec['group'])
            arrays = {key: value.copy() for key, value in loaded[spec['context']].items()}
            if mismatch: arrays['layer_0_input_bf16'][0] += 1
            return result, arrays
        with mock.patch.object(c.direct_forward, 'load_text_model', return_value=(model, self.tokenizer, loaded['result']['loading_info'], before)), \
             mock.patch.object(c.direct_forward, 'parameter_sha256', return_value='0' * 64 if mutation else 'd' * 64), \
             mock.patch.object(c, 'runtime_identity', return_value={k: v for k, v in self.runtime.items() if k != 'provenance'}), \
             mock.patch.object(c, 'off_target_position', return_value=1), \
             mock.patch.object(c.probe, 'forward_admitted', return_value=admitted), \
             mock.patch.object(c, 'run_addition', side_effect=forward) as called:
            result = c.run_causal(self.request, self.lock, {}, self.donors, output, self.patient, self.source,
                                  bindings={'request_sha256': '7' * 64, 'patient_sha256': 'b' * 64,
                                            'source_request_sha256': 'c' * 64})
            return result, called.call_count

    def test_orchestration_complete_and_postcheck_are_separate_from_quality(self):
        output = self.root / 'out'; output.mkdir()
        result, calls = self.run_fake(output)
        self.assertEqual(calls, 6)
        self.assertTrue(result['complete'])
        self.assertTrue(result['parameters_unchanged'])
        self.assertEqual(result['diagnostic_gate']['winner_group'], 1)
        self.assertEqual(result['quality_evaluation'], 'NOT_RUN')
        self.assertFalse(result['weights_edited'])
        self.assertEqual(read_json(output / 'result.json'), result)

    def test_replay_mismatch_interruptions_and_mutations_preserve_negative_proof(self):
        for mode in ('replay', 'interrupt', 'mutation'):
            output = self.root / mode; output.mkdir()
            with self.subTest(mode=mode), self.assertRaises((ValidationError, TimeoutError)):
                self.run_fake(output, mismatch=mode == 'replay', mutation=mode == 'mutation',
                              failure=TimeoutError('own deadline') if mode == 'interrupt' else None)
            result = read_json(output / 'result.json')
            self.assertFalse(result['complete'])
            self.assertEqual(result['diagnostic_gate']['status'], 'STOP_integrity_or_replay')
            self.assertEqual(result['parameters_unchanged'], mode != 'mutation')

    def test_time_admission_cannot_turn_missing_controls_into_a_pass(self):
        output = self.root / 'budget'; output.mkdir()
        result, calls = self.run_fake(output, admitted=False)
        self.assertEqual(calls, 0)
        self.assertFalse(result['complete'])
        self.assertEqual(result['diagnostic_gate']['status'], 'NOT_RUN_incomplete')

    def stage2_request(self):
        screen = self.root / 'screen'; screen.mkdir()
        result, _ = self.run_fake(screen)
        path = self.donors / 'stage1-result.json'; path.write_text(json.dumps(result))
        request = copy.deepcopy(self.request)
        request.update(stage=2, winner_group=1, stage1_result_sha256=c.sha256(path))
        request['donor_files']['stage1-result.json'] = c.sha256(path)
        request['budget']['max_forward_calls'] = 8
        return request

    def test_stage2_binds_stage1_hash_winner_donors_and_fixed_selection_rule(self):
        request = self.stage2_request(); self.validate(request)
        loaded = c.load_donors(request, self.lock, self.donors, self.patient, self.source)
        self.assertEqual(loaded['stage1']['diagnostic_gate']['winner_group'], 1)
        for field, value in [('winner_group', 0), ('stage1_result_sha256', '0' * 64)]:
            changed = copy.deepcopy(request); changed[field] = value
            with self.subTest(field=field), self.assertRaises(ValidationError):
                self.validate(changed)
                c.load_donors(changed, self.lock, self.donors, self.patient, self.source)
        stage1 = loaded['stage1']; stage1['diagnostic_gate']['winner_group'] = 0
        path = self.donors / 'stage1-result.json'; path.write_text(json.dumps(stage1))
        changed = copy.deepcopy(request)
        changed.update(winner_group=0, stage1_result_sha256=c.sha256(path))
        changed['donor_files']['stage1-result.json'] = c.sha256(path)
        with self.assertRaisesRegex(ValidationError, 'frozen selection'):
            c.load_donors(changed, self.lock, self.donors, self.patient, self.source)

    def test_stage2_orchestration_runs_exactly_eight_and_never_grants_weight_eligibility(self):
        self.request = self.stage2_request()
        output = self.root / 'controls'; output.mkdir()
        result, calls = self.run_fake(output)
        self.assertEqual(calls, 8)
        self.assertTrue(result['complete'])
        self.assertTrue(result['parameters_unchanged'])
        self.assertEqual(result['diagnostic_gate']['status'], 'activation_controls_passed')
        self.assertFalse(result['weight_candidate_eligible'])
        self.assertFalse(result['diagnostic_gate']['weight_candidate_eligible'])


TORCH_ENABLED = os.environ.get('WSP_TEST_DIRECT_CAUSAL_TORCH') == '1' and importlib.util.find_spec('torch')


@unittest.skipUnless(TORCH_ENABLED, 'explicit small Torch container required')
class TinyAdditionTests(unittest.TestCase):
    def test_addition_zero_identity_random_reverse_and_parameter_integrity(self):
        import torch
        torch.set_num_threads(1)
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.embedding = torch.nn.Embedding(16, 2)
                self.model = torch.nn.Module(); self.model.layers = torch.nn.ModuleList()
                for _ in range(32):
                    block = torch.nn.Module(); block.mlp = torch.nn.Module()
                    block.mlp.down_proj = torch.nn.Linear(3, 2, bias=False); self.model.layers.append(block)
                self.head = torch.nn.Linear(2, 16, bias=False)
            def forward(self, input_ids, attention_mask, use_cache, logits_to_keep, return_dict):
                hidden = self.embedding(input_ids)
                for block in self.model.layers:
                    features = torch.cat((hidden, torch.ones_like(hidden[..., :1])), dim=-1)
                    hidden = hidden + block.mlp.down_proj(features) * 0.05
                return SimpleNamespace(logits=self.head(hidden)[:, -logits_to_keep:, :].float())
        torch.manual_seed(4); model = Tiny().to(torch.bfloat16).eval()
        for value in model.parameters(): value.requires_grad_(False)
        before = {name: c.direct_forward.parameter_sha256(value) for name, value in model.named_parameters()}
        tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=0, decode=lambda ids, **kwargs: str(ids))
        plan = {'prefix_ids': [1, 2], 'candidate_ids': {'correct': [3, 4], 'incorrect': [5, 4]},
                'decision_position': 1, 'context': 'length_control'}
        deltas = {layer: np.array([0.25, -0.125], dtype=np.float32) for layer in range(32)}
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(p, 'INPUT_WIDTH', 3), \
             mock.patch.object(p, 'OUTPUT_WIDTH', 2):
            root = Path(directory)
            base, arrays = c.run_addition(model, tokenizer, plan, root,
                                         {'id': 'base', 'context': 'length_control'}, deltas, 0)
            zero, unchanged = c.run_addition(model, tokenizer, plan, root,
                {'id': 'zero', 'context': 'length_control', 'group': 0, 'strength': 0, 'mode': 'semantic'}, deltas, 0)
            c.check_replay(zero, unchanged, base, arrays, tolerance=0)
            for mode, strength, seed in [('semantic', 1, None), ('semantic', -1, None), ('random', 1, 20261108)]:
                spec = {'id': mode + str(strength), 'context': 'length_control', 'group': 0,
                        'strength': strength, 'mode': mode}
                if seed: spec['seed'] = seed
                result, after = c.run_addition(model, tokenizer, plan, root, spec, deltas, 0)
                expected_delta = c.sign_randomized(deltas, seed)[0] if seed else deltas[0]
                expected = (torch.from_numpy(arrays['layer_0_output_bf16'].copy()).view(torch.bfloat16).float()
                            + strength * torch.from_numpy(expected_delta)).to(torch.bfloat16)
                np.testing.assert_array_equal(after['layer_0_output_bf16'], expected.view(torch.uint16).numpy())
                self.assertEqual(len(result['actual_interventions']), 8)
            self.assertTrue(all(not block.mlp.down_proj._forward_hooks for block in model.model.layers))
        self.assertEqual(before, {name: c.direct_forward.parameter_sha256(value)
                                  for name, value in model.named_parameters()})


if __name__ == '__main__':
    unittest.main()
