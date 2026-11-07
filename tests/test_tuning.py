"""Promotion and repeatable tuning controls; no model execution on the CPU."""
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import shutil
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import test_coder
from compare_coder import acceptance
from ops import run54_gpu_phase as phase
import train_coder
from build_corrective_corpus import code_examples, tool_examples
from coder_common import normalize_record
from evaluate_coder import run_python
from prepare_coder_data import prepare, read_dataset


class TuningContracts(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('WSP_TEST_DOCKER_IMAGE') and shutil.which('docker'), 'Pinned Docker controls require the Linux test environment')
    def test_authored_code_targets_pass_independent_executable_controls(self):
        rows, solutions, checks = code_examples(2)
        self.assertEqual(len(rows), 16)
        result = run_python(solutions, checks, os.environ['WSP_TEST_DOCKER_IMAGE'])
        self.assertTrue(result['passed'])
        self.assertTrue(result['tests_completed'])

    @unittest.skipUnless(sys.platform.startswith('linux'), 'Atomic dataset publication requires Linux')
    def test_authored_variants_are_family_disjoint_between_splits(self):
        rows = code_examples(2)[0] + tool_examples(2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root/'authored.jsonl'
            path.write_text(''.join(json.dumps(normalize_record(row)) + '\n' for row in rows))
            prepare([path], [{'name':'authored','revision':'fixture','license':'owner-authored'}], root/'prepared', seed=20261106)
            _, splits = read_dataset(root/'prepared')
            families = [{row['task_id'] for row in splits[name]} for name in ('train','validation')]
            self.assertFalse(families[0] & families[1])
            self.assertEqual(families[0] | families[1], {row['task_id'] for row in rows})

    def test_authored_tool_summaries_copy_results_and_stop(self):
        for row in tool_examples(2):
            value = normalize_record(row)
            if row['task_id'].endswith(('observed_failure', 'observed_success')):
                observed = json.loads(value['messages'][-2]['content'])
                self.assertEqual(json.loads(value['messages'][-1]['content']), {k: observed[k] for k in ('returncode','fail_count')})
                self.assertEqual(sum(bool(m.get('tool_calls')) for m in value['messages']), 1)

    def test_authored_file_edits_ground_paths_and_replacements_in_observed_content(self):
        rows = [normalize_record(r) for r in tool_examples(2) if r['task_id'].endswith(('grounded_file_edit','recover_denied_path'))]
        self.assertEqual(len(rows), 4)
        for row in rows:
            calls = [call for m in row['messages'] for call in m.get('tool_calls',[])]
            read, edit = calls
            arguments = edit['function']['arguments']
            self.assertEqual(read['function']['arguments']['filePath'], arguments['filePath'])
            self.assertIn(arguments['filePath'], row['messages'][1]['content'])
            observed = next(m['content'] for m in row['messages'] if m['role']=='tool')
            self.assertIn(arguments['oldString'], observed)
            self.assertNotIn('tests passed', row['messages'][-1]['content'])

    def test_equal_humaneval_never_promotes_even_with_better_development(self):
        values = test_coder.PromotionTests().complete_evidence()
        values[3]['results'][0]['passed'] = False
        values[3]['passed'] = 163
        self.assertEqual(acceptance(*values)['status'], 'not_promoted')

    def test_strictly_better_humaneval_can_pass_with_all_other_evidence(self):
        values = test_coder.PromotionTests().complete_evidence()
        values[3]['results'][0]['passed'] = True
        values[3]['passed'] = 164
        self.assertEqual(acceptance(*values)['status'], 'accepted_for_local_use')

    def test_mlp_scope_preserves_attention_vision_and_output_layers(self):
        class Linear: pass
        names = ['model.language_model.layers.0.mlp.gate_proj',
                 'model.language_model.layers.0.mlp.up_proj',
                 'model.language_model.layers.0.mlp.down_proj',
                 'model.language_model.layers.0.self_attn.q_proj',
                 'model.visual.layers.0.mlp.up_proj', 'lm_head']
        modules = [(name, Linear()) for name in names]
        self.assertEqual(train_coder.adapter_targets(modules, Linear, 'mlp'), names[:3])
        self.assertEqual(train_coder.adapter_targets(modules, Linear, 'all-language'), names[:4])
        for scope in ('', 'attention', None):
            with self.subTest(scope=scope), self.assertRaises(ValueError):
                train_coder.adapter_targets(modules, Linear, scope)
        with self.assertRaises(ValueError): train_coder.adapter_targets([], Linear, 'mlp')

    def test_project_clients_are_sequential_inside_the_owned_server_window(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(python='fixture-python', config=root/'config.json', adapter=None,
                model_proof=root/'proof.json', port=18084, phase='evaluate', clients=True, projects=True,
                humaneval=True, image='pinned-image')
            info = {'wsp': {'revision': 'pinned-revision', 'adapter': None}}
            process = SimpleNamespace(poll=lambda: None)
            with patch.object(phase.subprocess, 'Popen', return_value=process), \
                 patch.object(phase, 'urlopen', side_effect=lambda *a, **k: io.BytesIO(json.dumps(info).encode())), \
                 patch.object(phase, 'run_logged') as run, patch.object(phase, 'stop_child'):
                phase.serve_and_evaluate(args, root, {'revision': 'pinned-revision', 'served_model_name': 'fixture-model'}, '/model')
            scripts = [call.args[0][2] for call in run.call_args_list]
            self.assertEqual(scripts, ['scripts/evaluate_coder.py', 'ops/smoke_clients.py',
                                      'ops/project_clients.py', 'scripts/evaluate_humaneval.py'])
            self.assertEqual(run.call_args_list[2].kwargs['accepted_codes'], (0, 2))


if __name__ == '__main__': unittest.main()
