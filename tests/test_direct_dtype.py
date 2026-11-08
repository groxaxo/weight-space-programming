"""Exact inference dtype restoration, never a training-library optimizer probe."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import direct_forward
from pipeline import ValidationError


@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('safetensors')
                     and importlib.util.find_spec('transformers'),
                     'Run separately in the bounded CPU inference container')
class OriginalDtypeTests(unittest.TestCase):
    def test_installed_tiny_model_loader_metadata_can_be_preserved(self):
        import torch
        from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM
        config = Qwen3_5TextConfig(vocab_size=64, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
            head_dim=8, layer_types=['full_attention'], tie_word_embeddings=False)
        with tempfile.TemporaryDirectory() as directory:
            # Synthetic random fixture, never optimized and never a research candidate.
            model = Qwen3_5ForCausalLM(config).eval()
            model.save_pretrained(directory)
            loaded, info = Qwen3_5ForCausalLM.from_pretrained(directory, dtype=torch.bfloat16,
                local_files_only=True, output_loading_info=True)
            result = direct_forward.loading_metadata(info)
            import json
            json.dumps(result, allow_nan=False)
            self.assertFalse(result['missing_keys'])
            self.assertFalse(result['mismatched_keys'])

    def test_cpu_reference_chunk_and_recurrence_agree_without_gpu_dispatch(self):
        import torch
        from transformers.models.qwen3_5 import modeling_qwen3_5 as module
        identity = direct_forward.select_cpu_reference_kernels()
        self.assertEqual(len(identity['functions_sha256']), 4)
        torch.manual_seed(20261108)
        q = torch.randn(1, 5, 2, 4)
        k = torch.randn_like(q)
        v = torch.randn(1, 5, 2, 3)
        g = -torch.rand(1, 5, 2)
        beta = torch.rand(1, 5, 2)
        with torch.inference_mode():
            chunk, chunk_state = module.torch_chunk_gated_delta_rule(q, k, v, g, beta,
                output_final_state=True, use_qk_l2norm_in_kernel=True)
            recurrent, recurrent_state = module.torch_recurrent_gated_delta_rule(q, k, v, g, beta,
                output_final_state=True, use_qk_l2norm_in_kernel=True)
        self.assertTrue(torch.allclose(chunk, recurrent, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(chunk_state, recurrent_state, atol=1e-5, rtol=1e-5))
        self.assertFalse(chunk.requires_grad)

    def test_fp32_auxiliary_is_reloaded_from_original_words_not_cast_back(self):
        import torch
        from safetensors.torch import save_file
        original = torch.tensor([0.123456789, 1.23456789, -0.987654321], dtype=torch.float32)
        rounded = original.to(torch.bfloat16)
        self.assertFalse(torch.equal(original, rounded.float()))
        model = torch.nn.Module(); model.block = torch.nn.Module()
        model.block.register_parameter('A_log', torch.nn.Parameter(rounded, requires_grad=False))
        before = model.block.A_log.detach().clone()
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'parent.safetensors'
            save_file({'official.A_log': original}, source)
            parameter, restored = direct_forward.restore_parent_dtype(model, 'block.A_log',
                model.block.A_log, 'official.A_log', source, {'dtype': 'F32', 'shape': [3]})
            self.assertTrue(restored)
            self.assertIs(parameter, model.block.A_log)
            self.assertEqual(parameter.dtype, torch.float32)
            self.assertFalse(parameter.requires_grad)
            self.assertTrue(torch.equal(parameter, original))
            self.assertTrue(torch.equal(before, rounded))
            self.assertEqual(direct_forward.parameter_sha256(parameter),
                             direct_forward.parameter_sha256(original))

    def test_original_bf16_is_never_rounded_or_replaced(self):
        import torch
        model = torch.nn.Module(); model.block = torch.nn.Module()
        parameter = torch.nn.Parameter(torch.tensor([0.25, -0.5], dtype=torch.bfloat16), requires_grad=False)
        model.block.register_parameter('weight', parameter)
        result, restored = direct_forward.restore_parent_dtype(model, 'block.weight', parameter,
                            'official.weight', Path('/unused'), {'dtype': 'BF16', 'shape': [2]})
        self.assertFalse(restored)
        self.assertIs(result, parameter)

    def test_wrong_shape_and_unsupported_dtype_are_rejected(self):
        import torch
        model = torch.nn.Module()
        parameter = torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16), requires_grad=False)
        for spec in ({'dtype': 'F32', 'shape': [3]}, {'dtype': 'I64', 'shape': [2]}):
            with self.assertRaises(ValidationError):
                direct_forward.restore_parent_dtype(model, 'block.weight', parameter,
                                                     'official.weight', Path('/unused'), spec)


if __name__ == '__main__':
    unittest.main()
