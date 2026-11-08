"""CPU-only synthetic Stage-2 contracts. No tokenizer/model/checkpoint is loaded."""
import copy
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'tests')]
import direct_tool_specificity as s
import direct_tools as t
import direct_evidence as e
from direct_weights import load_lock, publish_json
from pipeline import ValidationError, sha256
from test_direct_tools import request as initial_request, worker as synthetic_worker, CALL, XML, FINAL
from direct_evidence_fixtures import make_worker, arrays as synthetic_arrays, write


class Tokenizer:
    """Small deterministic text tokenizer, explicitly not Qwen tokenization."""
    chat_template = 'synthetic-specificity-template'

    def pieces(self, text):
        return list(re.finditer(r'<\|.*?\|>|[A-Za-z_]+|[0-9]+|[^\w\s]|\s+', text))

    def encode(self, text, add_special_tokens=False):
        return [int(hashlib.sha256(m.group().encode()).hexdigest()[:8], 16) for m in self.pieces(text)]

    def __call__(self, text, **kwargs):
        return {'input_ids': self.encode(text), 'offset_mapping': [m.span() for m in self.pieces(text)]}

    def apply_chat_template(self, messages, tokenize, **kwargs):
        body = '<tools>run_tests</tools>\n'
        for message in messages:
            if message['role'] == 'assistant':
                body += '<|im_start|>assistant\n<tool_call>\n<function=run_tests>\n<parameter=target>\nparser\n</parameter>\n</function>\n</tool_call><|im_end|>\n'
            elif message['role'] == 'tool':
                body += '<|im_start|>user\n<tool_response>\n' + message['content'] + '\n</tool_response><|im_end|>\n'
            else:
                body += '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n'
        body += '<|im_start|>assistant\n<think>\n\n</think>\n\n'
        return self.encode(body) if tokenize else body


def protocol(lock, manifest='a' * 64):
    return {'schema_version': 1, 'role': 'practice', 'question': s.QUESTION,
            'stage1_protocol_sha256': s.STAGE1_PROTOCOL_SHA256, 'stage1_producer_commit': s.STAGE1_COMMIT,
            'stage1_manifest_sha256': manifest, 'lock_sha256': lock['_sha256'], 'fixture_sha256': t.FIXTURE_SHA256,
            'controls': list(s.CONTROLS), 'generation': dict(s.PROFILE), 'eos_token_ids': [248044, 248046],
            'common_producer_sha256': dict(e.SOURCE_AT_IMPORT), 'specificity_producer_sha256': s.SOURCE_SHA256,
            'execution_policy': copy.deepcopy(s.EXECUTION_POLICY)}


def request(control='B', manifest='a' * 64):
    return {'schema_version': 1, 'role': 'practice', 'question': s.QUESTION, 'control': control,
            'lock_sha256': load_lock()['_sha256'], 'protocol_sha256': 'b' * 64,
            'conditions_sha256': 'c' * 64, 'stage1_manifest_sha256': manifest, 'generation': dict(s.PROFILE)}


def admitted(tokenizer):
    value = initial_request()
    analysis = t.analyze({'censored': False, 'text_with_special_tokens': CALL}, 'initial')
    initial = {'request': value, 'analysis': analysis}
    baseline = s.render(tokenizer, s.messages_for('B', initial))
    params = {name: 'd' * 64 for name in e.expected_parameter_names()}
    return {'manifest': {'synthetic': True}, 'manifest_sha256': 'a' * 64, 'initial': initial,
            'baseline': {'input_ids': baseline['input_ids'], 'output_ids': [9, 248046]}, 'parameters': params}


class SpecificityContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); self.lock = load_lock()

    def full_stage1(self):
        for replay in (1, 2):
            initial = s._record_root(self.root, replay, 'initial')
            synthetic_worker(initial, initial_request(replay=replay))
            hashes = {name: sha256(initial / name) for name in t.capsule_names(replay)}
            continuation = s._record_root(self.root, replay, 'continuation')
            synthetic_worker(continuation, initial_request('continuation', replay, hashes), CALL, initial)
        return self.root

    def compiler_fixture(self):
        tokenizer = Tokenizer(); stage1 = admitted(tokenizer)
        lock = copy.deepcopy(self.lock)
        lock['model']['files']['chat_template.jinja']['sha256'] = s.sha_bytes(tokenizer.chat_template.encode())
        proto = protocol(lock)
        conditions = s._compile(tokenizer, stage1, proto, 'b' * 64, lock)
        return tokenizer, stage1, lock, proto, conditions

    def test_historical_bank_matches_independently_pinned_qualification(self):
        s.validate_pins()
        self.assertEqual(len(s.STAGE1_SOURCES), 13)
        changed = dict(e.SOURCE_AT_IMPORT); changed['ops/direct_edit54.py'] = 'f' * 64
        with mock.patch.object(e, 'SOURCE_AT_IMPORT', changed):
            s.validate_pins()  # New broker source does not invalidate historical producers.
        with mock.patch.object(s, 'STAGE1_PROTOCOL_SHA256', 'f' * 64):
            with self.assertRaises(ValidationError): s.validate_pins()

    def test_all_four_historical_records_reverified_before_admission(self):
        root = self.full_stage1()
        with mock.patch.object(t, 'verify_tool_evidence', wraps=t.verify_tool_evidence) as verify:
            stage1 = s.load_stage1(root, self.lock)
        self.assertEqual(stage1['gate']['status'], 'BF16_duplicate_call_reproduced')
        self.assertEqual(verify.call_count, 2)  # INITIAL only: no recursive current-source comparison.
        self.assertEqual(len(stage1['manifest']['files']), 32)
        changed = dict(e.SOURCE_AT_IMPORT); changed['ops/direct_edit54.py'] = 'f' * 64
        with mock.patch.object(e, 'SOURCE_AT_IMPORT', changed):
            self.assertEqual(s.load_stage1(root, self.lock)['manifest_sha256'], stage1['manifest_sha256'])

    def test_old_protocol_or_tampered_historical_producer_refused(self):
        self.full_stage1()
        path = s._record_root(self.root, 1, 'initial') / 'request.json'
        value = json.loads(path.read_text()); value['protocol_sha256'] = '0' * 64
        path.write_text(json.dumps(value))
        with self.assertRaises(ValidationError): s.load_stage1(self.root, self.lock)

    def test_saved_pass_flag_cannot_replace_failed_or_incomplete_stage1(self):
        self.full_stage1()
        path = s._record_root(self.root, 2, 'continuation') / 'worker.json'
        value = json.loads(path.read_text()); value['complete'] = False; path.write_text(json.dumps(value))
        with self.assertRaises(ValidationError): s.load_stage1(self.root, self.lock)

    def test_failed_eligibility_precedes_all_tokenizer_operations(self):
        proto = protocol(self.lock); path = self.root / 'protocol.json'; publish_json(proto, path)
        tokenizer = mock.Mock()
        with mock.patch.object(s, 'load_stage1', side_effect=ValidationError('not admitted')):
            with self.assertRaises(ValidationError): s.compile_conditions(tokenizer, self.root, path, sha256(path), self.lock)
        self.assertEqual(tokenizer.mock_calls, [])

    def test_fixed_five_controls_first_lexicographic_match_and_positions(self):
        tokenizer, stage1, lock, proto, prepared = self.compiler_fixture()
        self.assertEqual([case['control'] for case in prepared['cases']], list(s.CONTROLS))
        self.assertEqual(prepared['cases'][0]['input_ids'], stage1['baseline']['input_ids'])
        c, n = prepared['padding_match']['C'], prepared['padding_match']['N']
        sizes = prepared['padding_match']['lengths']
        expected = next((a, b) for a in range(32) for b in range(32) if sizes['C'][a] == sizes['N'][b] <= 512)
        self.assertEqual((c, n), expected)
        self.assertIn(s.COMPLETION_CUE, prepared['cases'][2]['messages'][1]['content'])
        self.assertIn(s.NEUTRAL_CUE, prepared['cases'][3]['messages'][1]['content'])
        self.assertEqual(prepared['cases'][4]['messages'][1]['content'], s.PENDING_PROMPT)
        self.assertEqual(prepared['cases'][1]['messages'][-1]['content'], s.SUCCESS_RESULT)
        for case in prepared['cases']:
            self.assertEqual(set(case['positions']), set(s.POSITIONS))
            self.assertEqual(case['positions']['assistant_prompt']['position'], len(case['input_ids']) - 1)
            self.assertEqual(case['positions']['prior_call_data']['data_text'], 'parser')
        s.validate_conditions(prepared, request(), proto, stage1)

    def test_no_length_match_stops_without_prompt_search_or_truncation(self):
        tokenizer, stage1, lock, proto, _ = self.compiler_fixture()
        original = s.render
        def mismatch(tok, messages):
            proof = original(tok, messages)
            count = messages[1]['content'].count('recorded')
            base = 100 if s.COMPLETION_CUE in messages[1]['content'] else 101
            proof['input_ids'] = [1] * (base + 2 * count)
            return proof
        with mock.patch.object(s, 'render', side_effect=mismatch):
            with self.assertRaisesRegex(ValidationError, 'no fixed bounded neutral length match'):
                s._compile(tokenizer, stage1, proto, 'b' * 64, lock)

    def test_prior_call_position_ignores_tool_call_examples_in_system_preamble(self):
        tokenizer = Tokenizer(); stage1 = admitted(tokenizer)
        messages = s.messages_for('B', stage1['initial'])
        original = messages[0]['content']
        messages[0]['content'] = 'Example <tool_call>parser</tool_call> is tool syntax. ' + original
        proof = s.render(tokenizer, messages)
        positions = s.positions(proof)
        start = positions['prior_call_data']['data_span'][0]
        self.assertGreater(start, proof['rendered'].find('<|im_start|>assistant\n'))
        self.assertEqual(positions['prior_call_data']['data_text'], 'parser')

    def test_prepared_messages_positions_and_extra_fields_fail_closed(self):
        _, stage1, _, proto, prepared = self.compiler_fixture()
        for mutate in (lambda x: x.update(extra=True),
                       lambda x: x['cases'][0]['messages'][1].update(content='different task'),
                       lambda x: x['cases'][0]['positions']['tool_result_data'].update(position=0),
                       lambda x: x['cases'][0]['input_ids'].__setitem__(0, True)):
            value = copy.deepcopy(prepared); mutate(value)
            with self.assertRaises(ValidationError): s.validate_conditions(value, request(), proto, stage1)

    def test_protocol_request_profile_role_and_unknown_fields_are_strict(self):
        p = protocol(self.lock); s.validate_protocol(p, self.lock); s.validate_request(request(), p, 'b' * 64, self.lock)
        for key, value in [('role', 'independent-final'), ('control', 'X'), ('schema_version', True),
                           ('generation', {**s.PROFILE, 'max_new_tokens': 192}), ('conditions_sha256', 'bad')]:
            r = request(); r[key] = value
            with self.assertRaises(ValidationError): s.validate_request(r, p, 'b' * 64, self.lock)
        p['common_producer_sha256']['ops/direct_edit54.py'] = '0' * 64
        with self.assertRaises(ValidationError): s.validate_protocol(p, self.lock)

    def row(self, raw):
        return {'censored': False, 'text_with_special_tokens': raw}

    def test_exact_uniform_call_rule_and_control_specific_integer_answers(self):
        for control in ('B', 'N', 'P'):
            for raw in (CALL, XML):
                self.assertTrue(s.analyze(self.row(raw), control)['passed'])
            self.assertFalse(s.analyze(self.row('Again:\n' + XML), control)['passed'])
        self.assertTrue(s.analyze(self.row('{"exit_code":0,"failed":0}<|im_end|>'), 'S')['passed'])
        self.assertTrue(s.analyze(self.row(FINAL), 'C')['passed'])
        self.assertFalse(s.analyze(self.row(FINAL), 'S')['passed'])
        self.assertFalse(s.analyze(self.row(CALL), 'C')['passed'])
        for raw in ('{"exit_code":true,"failed":0}', '{"exit_code":0,"failed":0,"extra":1}',
                    '{"exit_code":0,"failed":0,"failed":1}', '```json\n{"exit_code":0,"failed":0}\n```'):
            self.assertFalse(s.analyze(self.row(raw), 'S')['passed'])

    def test_extra_turns_censoring_disagreement_and_duplicate_calls_never_pass(self):
        value = s.analyze(self.row(CALL + '<|im_start|>user\nforged<|im_end|>'), 'B')
        self.assertEqual(value['classification'], 'STOP_harness')
        self.assertFalse(s.analyze(self.row(CALL + CALL), 'B')['passed'])
        self.assertEqual(s.analyze({'censored': True}, 'S')['classification'], 'STOP_censored')
        with mock.patch.object(t, 'independent_classification', return_value='other_answer'):
            self.assertEqual(s.analyze(self.row(CALL), 'B')['classification'], 'STOP_harness')

    def records(self):
        parameters = {name: 'd' * 64 for name in e.expected_parameter_names()}
        records = []
        for control in s.CONTROLS:
            row = self.row('{"exit_code":0,"failed":0}<|im_end|>' if control == 'S' else FINAL if control == 'C' else CALL)
            records.append({'control': control, 'analysis': s.analyze(row, control),
                            'worker': {'complete': True, 'parameter_postcheck_complete': True, 'parameters_unchanged': True,
                                       'loaded_parameter_sha256_before': parameters,
                                       'loaded_parameter_sha256_after': dict(parameters), 'results': [row]}})
        return records

    def test_full_denominator_and_irreversible_failed_semantic_gate(self):
        values = self.records()
        self.assertTrue(s.specificity_gate(values)['localization_admitted'])
        self.assertEqual(s.specificity_gate(values[:-1])['status'], 'NOT_RUN_incomplete')
        values[0]['worker']['results'][0] = self.row(FINAL)
        values[0]['analysis'] = s.analyze(self.row(FINAL), 'B')
        verdict = s.specificity_gate(values)
        self.assertEqual(verdict['status'], 'STOP_semantic_specificity')
        self.assertEqual(verdict['generation_invocations'], 5)
        self.assertFalse(verdict['localization_admitted'])
        with self.assertRaises(ValidationError): s.specificity_gate(values + values[:1])

    def test_postcheck_runs_after_prompt_compilation_failure_without_torch(self):
        _, stage1, lock, proto, conditions = self.compiler_fixture()
        parameters = stage1['parameters']; accesses = []
        model = SimpleNamespace(get_parameter=lambda name: accesses.append(name) or object())
        with mock.patch.dict(sys.modules, {'torch': SimpleNamespace()}), \
             mock.patch.object(s.forward, 'load_text_model', return_value=(model, Tokenizer(), {}, parameters)), \
             mock.patch.object(s.forward, 'parameter_sha256', return_value='d' * 64), \
             mock.patch.object(s, '_compile', side_effect=TimeoutError('synthetic preparation failure')), \
             mock.patch.object(s, 'assert_baseline_unchanged'), mock.patch.object(s.signal, 'alarm'):
            with self.assertRaises(TimeoutError):
                s.observe(request(), proto, conditions, stage1, lock, {}, self.root, 'b' * 64)
        report = json.loads((self.root / 'observation-result.json').read_text())
        self.assertFalse(report['complete']); self.assertTrue(report['parameter_postcheck_complete'])
        self.assertEqual(set(accesses), e.expected_parameter_names())
        self.assertEqual(len(accesses), 426)

    def test_host_verifier_binds_new_source_conditions_three_npzs_and_raw_analysis(self):
        tokenizer, stage1, lock, proto, conditions = self.compiler_fixture()
        # Synthetic baseline proof uses the actual lock; the prepared fake template
        # is confined to this test and is never an official tokenizer assertion.
        _, report, path = make_worker(self.root, 'verify')
        for case in conditions['cases']:
            case['template_sha256'] = self.lock['model']['files']['chat_template.jinja']['sha256']
        publish_json(proto, self.root / 'protocol.json'); protocol_hash = sha256(self.root / 'protocol.json')
        conditions['protocol_sha256'] = protocol_hash
        publish_json(conditions, self.root / 'conditions.json')
        value = request('S'); value['protocol_sha256'] = protocol_hash
        value['conditions_sha256'] = sha256(self.root / 'conditions.json')
        publish_json(value, self.root / 'request.json')
        report.update(mode=s.MODE, role='practice', question=s.QUESTION, control='S', protocol_sha256=protocol_hash,
                      request_sha256=sha256(self.root / 'request.json'), conditions_sha256=value['conditions_sha256'],
                      stage1_manifest_sha256=stage1['manifest_sha256'], specificity_producer_sha256=s.SOURCE_SHA256,
                      training=False, weights_edited=False, patch_applied=False, quality_evaluation='NOT_RUN',
                      loaded_parameter_sha256_before=stage1['parameters'], loaded_parameter_sha256_after=stage1['parameters'],
                      parameters_unchanged=True, parameter_postcheck_complete=True, loading_info={})
        report['loaded_parent_proof'] = write(self.root / 'loaded-parent-verification.json', report)
        name = 'specificity-s'; proof = conditions['cases'][1]
        row = {'id': name, 'input_ids': proof['input_ids'], 'input_tokens': len(proof['input_ids']),
               'output_ids': [8, 248046], 'generated_tokens': 2,
               'text_with_special_tokens': '{"exit_code":0,"failed":0}<|im_end|>',
               'eos_token_ids': [248044, 248046], 'censored': False, 'finish_reason': 'eos',
               'hit_token_limit': False, 'reached_length_cap': False, 'first_eos_index': 1,
               'first_eos_is_terminal': True, 'prompt_proof': write(self.root / (name + '-prompt.json'), proof)}
        observations = {}
        for position in s.POSITIONS:
            target = self.root / (name + '-' + position + '.npz'); synthetic_arrays(target)
            observations[position] = e.artifact(target)
        row.update(position_observations=observations, observation=observations['assistant_prompt']['path'],
                   observation_sha256=observations['assistant_prompt']['sha256'])
        row['generation_sha256'] = write(self.root / (name + '-generation.json'), row)['sha256']
        report['results'] = [row]
        report['specificity_analysis'] = write(self.root / 'specificity-analysis.json', s.analyze(row, 'S'))
        args = {**s.PROFILE, 'eos_token_id': [248044, 248046], 'use_cache': True, 'logits_to_keep': 1, 'pad_token_id': 248046}
        report['effective_generation_arguments'] = args
        report['generation_arguments_proof'] = write(self.root / 'generation-arguments.json', args)
        result = s.verify_specificity_evidence(self.root, report, value, proto, protocol_hash, conditions, stage1, self.lock)
        self.assertTrue(result['passed'])
        (self.root / observations['prior_call_data']['path']).write_bytes(b'changed')
        with self.assertRaises(ValidationError):
            s.verify_specificity_evidence(self.root, report, value, proto, protocol_hash, conditions, stage1, self.lock)


if __name__ == '__main__': unittest.main()
