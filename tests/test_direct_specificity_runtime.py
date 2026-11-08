"""Real Torch hooks on synthetic arrays; no official checkpoint or quality score."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'tests')]
import direct_tool_specificity as s
import test_direct_tool_specificity as contracts
from direct_weights import load_lock


@unittest.skipUnless(os.environ.get('WSP_TEST_DIRECT_TOOL_TORCH') == '1'
                     and importlib.util.find_spec('torch'), 'Explicit bounded Torch runtime required')
class SpecificityRuntime(unittest.TestCase):
    def execute(self, output, *, interrupted=False):
        import torch
        torch.set_num_threads(1)
        fixture = contracts.SpecificityContracts('runTest'); fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        _, stage1, lock, protocol, conditions = fixture.compiler_fixture()
        proof = conditions['cases'][1]  # S, so no baseline replay is simulated.
        ids = proof['input_ids']; count = len(ids)
        parameter = torch.tensor([0.125], dtype=torch.bfloat16)
        before = {name: s.forward.parameter_sha256(parameter) for name in s.evidence.expected_parameter_names()}
        stage1['parameters'] = before
        class Down(torch.nn.Module):
            def forward(self, value): return value[..., :2560].clone()
        blocks = [SimpleNamespace(mlp=SimpleNamespace(down_proj=Down())) for _ in range(32)]
        class Model:
            model = SimpleNamespace(layers=blocks)
            generation_config = SimpleNamespace(eos_token_id=248044)
            calls = 0
            def get_parameter(self, name): return parameter
            def generate(self, input_ids, **kwargs):
                self.calls += 1
                if interrupted: raise TimeoutError('synthetic generation interruption')
                for block in blocks:
                    block.mlp.down_proj(torch.arange(count, dtype=torch.bfloat16).reshape(1, count, 1).expand(1, count, 9216))
                    # Decode/cache callback must not overwrite prefill arrays.
                    block.mlp.down_proj(torch.full((1, 1, 9216), -7, dtype=torch.bfloat16))
                return torch.tensor([ids + [9, 248046]])
        model = Model()
        tokenizer = SimpleNamespace(eos_token_id=248046,
                                   decode=lambda *args, **kw: '{"exit_code":0,"failed":0}<|im_end|>')
        request = {'control': 'S', 'conditions_sha256': 'a' * 64}
        with mock.patch.object(s.forward, 'load_text_model', return_value=(model, tokenizer, {}, before)), \
                mock.patch.object(s, '_compile', return_value=conditions), \
                mock.patch.object(s, 'assert_baseline_unchanged'), mock.patch.object(s.signal, 'alarm'):
            result = s.observe(request, protocol, conditions, stage1, lock, {}, output, 'b' * 64)
        return result, proof, blocks, model

    def test_three_distinct_positions_all_layers_and_cache_callbacks_preserve_prefill(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); report, proof, blocks, model = self.execute(root)
            self.assertEqual(model.calls, 1)
            self.assertTrue(report['complete']); self.assertTrue(report['parameters_unchanged'])
            self.assertEqual(len(report['loaded_parameter_sha256_after']), 426)
            self.assertTrue(all(not b.mlp.down_proj._forward_hooks for b in blocks))
            for position, reference in report['results'][0]['position_observations'].items():
                arrays = s.evidence.numeric_arrays(root, reference)
                self.assertEqual(len(arrays), 64)
                expected = proof['positions'][position]['position']
                import torch
                word = torch.tensor(expected, dtype=torch.bfloat16).view(torch.uint16).item()
                self.assertTrue(all((value == word).all() for value in arrays.values()))
            self.assertTrue(s.analyze(report['results'][0], 'S')['passed'])

    def test_generation_interruption_preserves_all_426_postchecks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(TimeoutError, 'synthetic generation'): self.execute(root, interrupted=True)
            report = json.loads((root / 'observation-result.json').read_text())
            self.assertFalse(report['complete']); self.assertTrue(report['parameter_postcheck_complete'])
            self.assertEqual(len(report['loaded_parameter_sha256_after']), 426)


if __name__ == '__main__': unittest.main()
