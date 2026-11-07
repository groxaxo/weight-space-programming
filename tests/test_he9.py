"""HE9 protocol integrity and promotion controls; all fixtures are synthetic."""
import copy
from pathlib import Path
import sys
import unittest
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
from coder_common import CoderError, fingerprint
from compare_he9 import compare_he9, TOTALS
from ops.replay_he9 import TASKS, LIMITS, PRIVATE_SHA256, GRADER_IMAGE, actor_events, opencode_config
from ops.replay_he9_vm54 import record_progress
from coder_sampling import QWEN35_INSTRUCT


class He9Tests(unittest.TestCase):
    def fixture(self, solved, adapter=False):
        h = 'a'*64
        identity = {'adapter': None, 'base_model': 'synthetic', 'revision': 'b'*40,
            'model_proof_sha256': h, 'artifact_sha256': h, 'quantization': 'Q8_0',
            'runtime': {'launch_options': LIMITS,'sampling':QWEN35_INSTRUCT}, 'protocol_sha256': h,
            'chat_stop_token_ids': [1, 2], 'limits': {'context': 262144, 'output': 8192}}
        if adapter:
            identity.update(adapter='/synthetic/adapter', adapter_application='native-f32-lora-on-q8_0',
                adapter_sha256=h, adapter_config_sha256=h, lora_artifact_sha256=h)
        report = {'schema_version': 1, 'suite': 'experimentos-selected-he9', 'status': 'complete',
            'tasks': TASKS, 'suite_sha256': PRIVATE_SHA256, 'limits': LIMITS,
            'grader_image': GRADER_IMAGE, 'protocol_sha256': {'ops/client_proxy.py': h},
            'vm': {'personal_vms_modified': False, 'vm': 'synthetic-clean'},
            'virgin_guest_proof': {k: True for k in ('no_prior_opencode_binary', 'no_prior_opencode_state',
                'no_prior_hermes_state', 'no_prior_harness_workspace')}, 'results': []}
        for i, task in enumerate(TASKS):
            passed = i < solved
            report['results'].append({'task_id': task, 'identity': identity, 'instance_id': str(i),
                'limits': LIMITS, 'suite_sha256': PRIVATE_SHA256, 'grader_image': GRADER_IMAGE,
                'grading': 'scored', 'actor_errors': [], 'actor_status': 'finished', 'actor_exit': 0,
                'final_step_observed': True, 'base_pass': passed, 'plus_pass': passed,
                'plus_total': TOTALS[task], 'plus_passed': TOTALS[task] if passed else 0,
                'error': None if passed else 'synthetic failure',
                'preflight': {'empty_client_state': True, 'interfaces': ['lo'],
                    'docker_socket_mounted': False, 'python': '3.11.17'},
                'isolation': {'verified': True, 'network': 'none'}, 'remaining_actor_containers': [],
                'actor_image': 'sha256:'+h, 'opencode_version': '1.19.4', 'opencode_sha256': h,
                'harness_sha256': {'ops/client_proxy.py': h}, 'reference_selftest_sha256': h,
                'project_source_sha256': h, 'model_attestation': {'verified': True, 'requests': 2,
                    'instance_id': str(i), 'identity_sha256': fingerprint(identity), 'proxy_sha256': h}})
        return report

    def test_only_a_strict_gain_qualifies_and_never_automatically_promotes(self):
        result = compare_he9(self.fixture(3), self.fixture(4, True))
        self.assertEqual(result['status'], 'he9_qualified_pending_compatibility')
        self.assertFalse(result['adapter_promoted'])
        for solved in (2, 3):
            self.assertEqual(compare_he9(self.fixture(3), self.fixture(solved, True))['status'], 'not_promoted')

    def test_an_aggregate_gain_cannot_hide_a_previously_solved_task_regression(self):
        candidate = self.fixture(5, True)
        row = candidate['results'][0]
        row.update(base_pass=False, plus_pass=False, plus_passed=0, error='synthetic regression')
        result = compare_he9(self.fixture(3), candidate)
        self.assertTrue(result['strict_improvement'])
        self.assertEqual(result['status'], 'not_promoted')
        self.assertEqual(result['regressions'], [TASKS[0]])

    def test_partial_duplicate_unscored_and_unattributed_results_fail_closed(self):
        for change in ('partial', 'duplicate', 'unscored', 'no_requests', 'kv_drift', 'state_reused', 'different_source'):
            candidate = self.fixture(4, True)
            if change == 'partial': candidate['results'].pop()
            if change == 'duplicate': candidate['results'][1]['task_id'] = TASKS[0]
            if change == 'unscored': candidate['results'][0]['grading'] = 'grader_error_unscored'
            if change == 'no_requests': candidate['results'][0]['model_attestation']['requests'] = 0
            if change == 'kv_drift':
                candidate = copy.deepcopy(candidate)
                candidate['limits']['cache_type_v'] = 'f16'
            if change == 'state_reused': candidate['results'][0]['preflight']['empty_client_state'] = False
            if change == 'different_source': candidate['results'][0]['project_source_sha256'] = 'b'*64
            with self.subTest(change=change), self.assertRaises(CoderError):
                compare_he9(self.fixture(3), candidate)

    def test_declaring_max_context_reserves_the_original_output_allowance(self):
        config = opencode_config('synthetic')
        provider = config['provider']['local']
        self.assertEqual(provider['models']['synthetic']['limit'],
            {'context': 262144, 'input': 253952, 'output': 8192})
        self.assertEqual(config['mcp'], {})
        tools = config['agent']['build']['tools']
        self.assertFalse(tools['*'])
        self.assertNotIn('task', tools)
        self.assertEqual(provider['options']['baseURL'], 'http://127.0.0.1:18084/v1')
        self.assertFalse(provider['models']['synthetic']['reasoning'])
        self.assertEqual(provider['models']['synthetic']['options'], {})

    def test_budget_artifact_is_scored_but_does_not_count_as_agent_completion(self):
        candidate = self.fixture(4, True)
        candidate['results'][0].update(actor_status='budget_limit', actor_exit=124,
            grading='scored_artifact_at_budget_limit', final_step_observed=False)
        result = compare_he9(self.fixture(3), candidate)
        self.assertEqual(result['candidate']['plus'], 4)
        self.assertEqual(result['candidate']['completed_actors'], 8)
        self.assertEqual(result['status'], 'not_promoted')

    def test_task_progress_preserves_each_checkpoint_without_overwriting_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            record_progress(output, {'suite': 'synthetic'}, [{'task_id': 'one'}])
            original = (output/'progress-01.json').read_bytes()
            record_progress(output, {'suite': 'synthetic'}, [{'task_id': 'one'}, {'task_id': 'two'}])
            self.assertEqual((output/'progress-01.json').read_bytes(), original)
            self.assertTrue((output/'progress-02.json').is_file())
            self.assertFalse((output/'replay.json').exists())
            with self.assertRaises(FileExistsError):
                record_progress(output, {}, [{'task_id': 'one'}])


if __name__ == '__main__': unittest.main()
