"""Frozen full-episode contracts and independent parser controls; no Qwen load."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'tests')]
import direct_tools as t
import direct_evidence as e
from direct_weights import load_lock
from ops import direct_edit54 as host
from pipeline import ValidationError, sha256
from direct_evidence_fixtures import make_worker, arrays, write

XML = '<tool_call>\n<function=run_tests>\n<parameter=target>\nparser\n</parameter>\n</function>\n</tool_call><|im_end|>'
CALL = '<tool_call>{"name":"run_tests","arguments":{"target":"parser"}}</tool_call><|im_end|>'
FINAL = '{"exit_code":1,"failed":2}<|im_end|>'


def request(phase='initial', replay=1, episode_files=None):
    return {'schema_version': 1, 'role': 'practice', 'question': t.QUESTION,
            'lock_sha256': load_lock()['_sha256'], 'protocol_sha256': sha256(t.PROTOCOL_FILE),
            'fixture_sha256': t.FIXTURE_SHA256, 'phase': phase, 'replay': replay,
            'episode_files': episode_files or {}, 'generation': dict(t.PROFILE)}


def worker(output, value, raw=CALL, capsule_source=None):
    _, base, path = make_worker(output, 'verify')
    lock = load_lock(); name = t.case_id(value['replay'], value['phase'])
    write(output / 'request.json', value); request_hash = sha256(output / 'request.json')
    capsule = None
    if value['phase'] == 'continuation':
        t.copy_capsule(capsule_source, output / 'episode', value)
        capsule = t.load_capsule(output / 'episode', value, lock)
    parameters = {name: 'd' * 64 for name in e.expected_parameter_names()}
    report = {**base, 'mode': t.MODE, 'question': t.QUESTION, 'fixture_sha256': t.FIXTURE_SHA256,
              'phase': value['phase'], 'replay': value['replay'], 'episode_files': value['episode_files'],
              'request_sha256': request_hash, 'protocol_sha256': value['protocol_sha256'],
              'loaded_parameter_sha256_before': parameters, 'loaded_parameter_sha256_after': dict(parameters),
              'parameters_unchanged': True, 'parameter_postcheck_complete': True, 'loading_info': {}}
    report['loaded_parent_proof'] = write(output / 'loaded-parent-verification.json', report)
    proof = {'messages': t.messages_for(value, capsule), 'tools': t.CASE['tools'], 'input_ids': [1, 2],
             'rendered': 'synthetic fixture, not a model invocation',
             'template_sha256': lock['model']['files']['chat_template.jinja']['sha256'],
             'harness_call_id_tokenization_invariant': True, 'silent_truncation': False, 'thinking': False}
    row = {'id': name, 'input_ids': [1, 2], 'input_tokens': 2, 'output_ids': [3, 2], 'generated_tokens': 2,
           'text': raw, 'text_with_special_tokens': raw,
           'eos_token_ids': [2], 'finish_reason': 'eos', 'reached_length_cap': False,
           'hit_token_limit': False, 'censored': False,
           'prompt_proof': write(output / (name + '-prompt.json'), proof),
           'observation': name + '-prefill.npz'}
    arrays(output / row['observation']); row['observation_sha256'] = sha256(output / row['observation'])
    row['generation_sha256'] = write(output / (name + '-generation.json'), row)['sha256']
    report['results'] = [row]
    report['tool_analysis'] = write(output / 'tool-analysis.json', t.analyze(row, value['phase']))
    write(path, report)
    return report, path


class ToolReproductionContracts(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(); self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_request_is_fixed_practice_only_no_candidate_and_no_profile_tuning(self):
        t.validate_request(request(), load_lock())
        bad = [('role', 'independent-final'), ('question', 'half_open_overlap'), ('replay', True),
               ('replay', 3), ('fixture_sha256', '0' * 64), ('protocol_sha256', '0' * 64),
               ('episode_files', []), ('generation', {**t.PROFILE, 'max_new_tokens': 192})]
        for key, value in bad:
            with self.subTest(key=key), self.assertRaises(ValidationError):
                t.validate_request({**request(), key: value}, load_lock())
        with self.assertRaises(ValidationError): t.validate_request({**request(), 'patch': {}}, load_lock())

    def test_fixed_messages_match_actual_exposed_evaluator_system_and_prompt(self):
        from evaluate_coder import evaluate_tools
        captured = []
        class Client:
            def complete(self, messages, tools=None):
                captured.append(copy.deepcopy(messages))
                if len(captured) == 1:
                    return t.analyze({'censored': False, 'text_with_special_tokens': CALL}, 'initial')['parsed_message'], {}
                return {'role': 'assistant', 'content': '{"exit_code":1,"failed":2}'}, {}
        evaluate_tools(Client(), t.CASE, [])
        self.assertEqual(t.messages_for(request()), captured[0])
        self.assertEqual(captured[1][-1]['content'], t.CASE['steps'][0][2])

    def test_xml_json_and_final_json_agree_between_independent_parsers(self):
        for raw, phase, expected in [(CALL, 'initial', 'expected_initial_call'),
                (XML, 'initial', 'expected_initial_call'), (XML, 'continuation', 'repeat_same_call'),
                (FINAL, 'continuation', 'correct_final_json')]:
            value = t.analyze({'censored': False, 'text_with_special_tokens': raw}, phase)
            with self.subTest(raw=raw, phase=phase):
                self.assertTrue(value['parser_agreement']); self.assertEqual(value['classification'], expected)

    def test_wrong_arguments_duplicates_bad_json_and_censoring_never_become_target_repeats(self):
        samples = [CALL.replace('parser', 'other'), CALL + CALL,
                   '<tool_call>{"name":"run_tests","arguments":{}}</tool_call>',
                   '{"exit_code":true,"failed":2}', '{"exit_code":1,"failed":2,"extra":0}',
                   '<tool_call>{"name":"run_tests"}', '<tool_call>{"name":"run_tests","arguments":{"target":"parser","target":"other"}}</tool_call>']
        for raw in samples:
            with self.subTest(raw=raw):
                self.assertNotEqual(t.analyze({'censored': False, 'text_with_special_tokens': raw}, 'continuation')['classification'], 'repeat_same_call')
        self.assertEqual(t.analyze({'censored': True, 'text_with_special_tokens': CALL}, 'continuation')['classification'], 'STOP_censored')

    def test_disagreement_is_harness_stop_not_a_weight_defect(self):
        with mock.patch.object(t, 'independent_classification', return_value='other_answer'):
            result = t.analyze({'censored': False, 'text_with_special_tokens': CALL}, 'continuation')
        self.assertFalse(result['parser_agreement']); self.assertEqual(result['classification'], 'STOP_harness')

    def test_fast_tokenizer_mapping_is_unwrapped_for_id_invariance(self):
        from collections import UserDict
        class FastMapping(UserDict):
            def __getitem__(self, key):
                if key == 0: return object()  # Fast BatchEncoding exposes an Encoding row here.
                return super().__getitem__(key)
        class Tokenizer:
            chat_template = 'synthetic tokenizer source'
            def apply_chat_template(self, messages, tokenize, **options):
                if not tokenize: return 'fixed synthetic rendering'
                return np.array([[1, 2]]) if options.get('return_tensors') else FastMapping(input_ids=[[1, 2]])
            def encode(self, text, add_special_tokens=False): return [1, 2]
        analysis = t.analyze({'censored': False, 'text_with_special_tokens': CALL}, 'initial')
        ids, proof = t.prepare_prompt(Tokenizer(), {'phase': 'continuation'}, {'analysis': analysis})
        self.assertEqual(ids.tolist(), [[1, 2]])
        self.assertTrue(proof['harness_call_id_tokenization_invariant'])

    def test_call_uuid_is_preserved_and_wire_arguments_normalized_without_repair(self):
        analysis = t.analyze({'censored': False, 'text_with_special_tokens': CALL}, 'initial')
        value = request('continuation', episode_files={name: 'a' * 64 for name in t.capsule_names(1)})
        messages = t.messages_for(value, {'analysis': analysis})
        self.assertEqual(messages[-2]['tool_calls'][0]['id'], analysis['parsed_message']['tool_calls'][0]['id'])
        self.assertEqual(messages[-1]['tool_call_id'], messages[-2]['tool_calls'][0]['id'])
        self.assertEqual(messages[-2]['tool_calls'][0]['function']['arguments'], {'target': 'parser'})
        self.assertEqual(analysis['parsed_message']['tool_calls'][0]['function']['arguments'], '{"target":"parser"}')

    def test_capsule_authenticates_actual_initial_evidence_before_continuation(self):
        initial = self.root / 'initial'; value = request(); report, path = worker(initial, value)
        self.assertTrue(host.worker_evidence(initial, t.MODE, value, report['request_sha256'], value['lock_sha256'])['worker_complete'])
        digests = {name: sha256(initial / name) for name in t.capsule_names(1)}
        continuation = request('continuation', episode_files=digests)
        loaded = t.load_capsule(initial, continuation, load_lock())
        self.assertEqual(loaded['analysis']['classification'], 'expected_initial_call')
        output = self.root / 'continuation'
        result, path = worker(output, continuation, FINAL, initial)
        self.assertTrue(host.worker_evidence(output, t.MODE, continuation, result['request_sha256'], continuation['lock_sha256'])['worker_complete'])
        (initial / 'tool-analysis.json').write_text('{}')
        with self.assertRaises(ValidationError): t.load_capsule(initial, continuation, load_lock())

    def test_correct_initial_answer_cannot_be_replaced_with_a_synthetic_tool_call(self):
        initial = self.root / 'initial'; value = request(); worker(initial, value, FINAL)
        continuation = request('continuation', episode_files={name: sha256(initial / name) for name in t.capsule_names(1)})
        with self.assertRaisesRegex(ValidationError, 'spontaneously'):
            t.load_capsule(initial, continuation, load_lock())

    def test_broker_keeps_tool_worker_in_shared_cpu_sandbox_and_freezes_capsule(self):
        initial = request(); host.validate_host_request(initial, t.MODE, initial['lock_sha256'])
        command = host.container_command('wsp-direct-tool-contract', Path('/source'), Path('/deps'), [],
                    Path('/out'), Path('/request'), 10 * host.GIB, t.MODE, episode=Path('/initial-frozen'))
        self.assertIn('scripts/direct_tools.py', command)
        self.assertIn('type=bind,source=/initial-frozen,target=/episode,readonly', command)
        self.assertNotIn('--gpus', command); self.assertNotIn('--rm', command)
        self.assertIn('CUDA_VISIBLE_DEVICES=', command)

    def records(self, continuation=CALL):
        records = []
        for replay in (1, 2):
            for phase, raw in [('initial', CALL), ('continuation', continuation)]:
                params = {name: 'd' * 64 for name in e.expected_parameter_names()}
                records.append({'phase': phase, 'replay': replay,
                    'analysis': t.analyze({'censored': False, 'text_with_special_tokens': raw}, phase),
                    'worker': {'complete': True, 'parameter_postcheck_complete': True, 'parameters_unchanged': True,
                        'loaded_parameter_sha256_before': params, 'loaded_parameter_sha256_after': dict(params),
                        'results': [{'input_ids': [1, 2], 'output_ids': [3, 2]}]}})
        return records

    def test_frozen_gate_requires_both_full_exact_episodes_and_never_admits_weights(self):
        result = t.reproduction_gate(self.records())
        self.assertEqual(result['status'], 'BF16_duplicate_call_reproduced')
        self.assertTrue(result['activation_diagnosis_admitted']); self.assertFalse(result['weight_candidate_eligible'])
        self.assertEqual(result['quality_evaluation'], 'NOT_RUN')
        self.assertEqual(t.reproduction_gate(self.records()[:-1])['status'], 'NOT_RUN_incomplete')

    def test_either_correct_json_irreversibly_stops_hypothesis_with_full_denominator(self):
        values = self.records(); values[1]['analysis'] = t.analyze({'censored': False, 'text_with_special_tokens': FINAL}, 'continuation')
        gate = t.reproduction_gate(values)
        self.assertEqual(gate['status'], 'STOP_BF16_target_not_reproduced'); self.assertEqual(gate['generation_invocations'], 4)
        self.assertFalse(gate['activation_diagnosis_admitted'])

    def test_nondeterministic_episode_tokens_or_integrity_cannot_admit_localization(self):
        values = self.records(); values[-1]['worker']['results'][0]['output_ids'] = [4, 2]
        self.assertEqual(t.reproduction_gate(values)['status'], 'STOP_inconclusive')
        values = self.records(); values[0]['worker']['parameter_postcheck_complete'] = False
        self.assertEqual(t.reproduction_gate(values)['status'], 'STOP_inconclusive')


if __name__ == '__main__': unittest.main()

# Explicit admitted CPU runtime only; random/stub fixtures, never official Qwen weights.
import importlib.util
import os
from types import SimpleNamespace


@unittest.skipUnless(os.environ.get('WSP_TEST_DIRECT_TOOL_TORCH') == '1' and importlib.util.find_spec('torch'),
                     'explicit tiny tool observer CPU container required')
class TinyToolObserver(unittest.TestCase):
    def execute(self, outdir, *, interrupted=False):
        import torch
        from direct_forward import observe, parameter_sha256
        class Down(torch.nn.Module):
            def forward(self, value):
                return torch.zeros((*value.shape[:-1], 2560), dtype=torch.bfloat16)
        layers = [SimpleNamespace(mlp=SimpleNamespace(down_proj=Down())) for _ in range(32)]
        protected = torch.zeros(1, dtype=torch.bfloat16)
        class Model:
            model = SimpleNamespace(layers=layers)
            generation_config = SimpleNamespace(eos_token_id=2)
            def get_parameter(self, name): return protected
            def generate(self, input_ids, **kwargs):
                self.generation_calls = getattr(self, 'generation_calls', 0) + 1
                for block in self.model.layers:
                    block.mlp.down_proj(torch.zeros((1, 2, 9216), dtype=torch.bfloat16))
                return torch.tensor([[1, 2, 3, 2]], dtype=torch.long)
        model = Model()
        tokenizer = SimpleNamespace(eos_token_id=2, decode=lambda ids, skip_special_tokens: CALL)
        before = {name: parameter_sha256(protected) for name in e.expected_parameter_names()}
        def builder(tokenizer, case):
            if interrupted: raise TimeoutError('synthetic prompt preparation interruption')
            return torch.tensor([[1, 2]]), {'fixture': 'synthetic-tool-prompt', 'input_ids': [1, 2]}
        loading = {'official_dtype_restorations': [], 'cpu_reference_kernels': {'kind': 'synthetic-runtime'}}
        with mock.patch('direct_forward.load_text_model', return_value=(model, tokenizer, loading, before)), mock.patch('builtins.print'):
            result = observe(t.observer_request(request()), load_lock(), {}, outdir, prompt_builder=builder)
        return model, layers, result

    def test_real_torch_observer_callback_raw_markers_hooks_and_postcheck(self):
        with tempfile.TemporaryDirectory() as directory:
            model, layers, report = self.execute(Path(directory))
            self.assertEqual(model.generation_calls, 1)
            self.assertTrue(report['parameter_postcheck_complete']); self.assertTrue(report['parameters_unchanged'])
            self.assertEqual(len(report['loaded_parameter_sha256_after']), 426)
            self.assertEqual(report['results'][0]['text_with_special_tokens'], CALL)
            self.assertEqual(t.analyze(report['results'][0], 'initial')['classification'], 'expected_initial_call')
            self.assertTrue(all(not layer.mlp.down_proj._forward_hooks for layer in layers))

    def test_real_torch_prompt_interruption_preserves_incomplete_426_postcheck(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with self.assertRaises(TimeoutError): self.execute(output, interrupted=True)
            report = json.loads((output / 'observation-result.json').read_text())
            self.assertFalse(report['complete']); self.assertTrue(report['parameter_postcheck_complete'])
            self.assertEqual(len(report['loaded_parameter_sha256_after']), 426)
