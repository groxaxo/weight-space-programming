"""Accidental legacy CLI reuse must fail before model work or service control."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from method_policy import MethodPolicyError, blocked_legacy_operation


class MethodPolicyTests(unittest.TestCase):
    def test_no_runtime_override(self):
        with self.assertRaisesRegex(MethodPolicyError, 'no CLI/environment override'):
            blocked_legacy_operation('training')

    def test_checked_in_policy_matches_the_user_boundary(self):
        policy = json.loads((ROOT / 'research-policy.json').read_text())
        for name in ('training_allowed', 'lora_allowed', 'backward_allowed',
                     'optimizers_allowed', 'iterative_loss_fitting_allowed',
                     'service_preemption_allowed', 'cleanup_allowed',
                     'reuse_08b_coordinates_allowed'):
            self.assertIs(policy[name], False, name)
        self.assertEqual(policy['surgeon'], {'model': 'gpt-6.1-sol', 'reasoning_effort': 'xhigh'})

    def test_legacy_clis_refuse_before_creating_outputs(self):
        entries = ('scripts/train_coder.py', 'scripts/compile_coder.py',
                   'scripts/import_coder_corpora.py', 'scripts/prepare_coder_data.py',
                   'scripts/build_corrective_corpus.py', 'scripts/build_semantic_repair_corpus.py',
                   'scripts/build_formatted_repair_corpus.py', 'scripts/build_agent_flow_corpus.py',
                   'scripts/build_agent_continuation_corpus.py',
                   'scripts/export_coder_lora_gguf.py', 'scripts/convert_coder_gguf.py',
                   'scripts/select_coder_checkpoint.py', 'ops/verify_pipeline54.py',
                   'ops/replay_he9_vm54.py',
                   'scripts/serve_coder.py', 'scripts/serve_coder_gguf.py',
                   'ops/run54_gguf_phase.py')
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'never-created'
            for entry in entries:
                with self.subTest(entry=entry):
                    result = subprocess.run([sys.executable, str(ROOT / entry),
                                             '--output', str(output)],
                                            capture_output=True, text=True, timeout=15)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn('is disabled: the active objective', result.stderr)
                    self.assertFalse(output.exists())

    def test_legacy_service_launcher_cannot_start_or_stop_a_job(self):
        for args in (['start', '--base', '--allow-pause-bonsai'], ['stop']):
            with self.subTest(args=args):
                result = subprocess.run([sys.executable, str(ROOT / 'ops/local_coder54.py'), *args],
                                        capture_output=True, text=True, timeout=15)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('is disabled: the active objective', result.stderr)

    def test_training_supervisor_refuses_before_output_or_service_actions(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'never-created'
            result = subprocess.run([sys.executable, str(ROOT / 'ops/run54_gpu_phase.py'),
                                     '--phase', 'train', '--output', str(output),
                                     '--allow-pause-bonsai'], capture_output=True,
                                    text=True, timeout=15)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('is disabled: the active objective', result.stderr)
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
