"""Reject altered parents; prove continuation loads real PEFT weights on CPU."""
import copy
from contextlib import nullcontext
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from coder_common import CoderError, digest
from coder_initialization import initialize_adapter, verify_initial_adapter


def write(path, value):
    path.write_text(json.dumps(value))


def parent(root, adapter=None, targets=None):
    training = root / 'training'
    training.mkdir()
    destination = training / 'adapter'
    cfg = json.loads((ROOT / 'configs/qwen35-4b-corrective-mlp.json').read_text())
    targets = targets or ['model.language_model.layers.0.mlp.down_proj']
    if adapter is not None:
        adapter.save_pretrained(destination)
    else:
        destination.mkdir()
        (destination / 'adapter_model.safetensors').write_bytes(b'authentication-only fixture, not real weights')
        write(destination / 'adapter_config.json', {'peft_type': 'LORA', 'bias': 'none', 'r': 8,
              'lora_alpha': 16, 'lora_dropout': 0.05, 'task_type': 'CAUSAL_LM', 'target_modules': targets})
    (training / 'source').mkdir()
    (training / 'source/recipe.py').write_text('# frozen fixture\n')
    write(training / 'run-manifest.json', {'model_id': cfg['model_id'], 'model_revision': cfg['revision'],
          'config': cfg, 'recipe_sha256': {'recipe.py': digest(training / 'source/recipe.py')}})
    write(training / 'completed-run-manifest.json', {'status': 'complete', 'exit_code': 0})
    write(training / 'adapter-targets.json', targets)
    result = {'status': 'trained_unpromoted', 'smoke_only': False,
              'adapter_files': {p.name: digest(p) for p in destination.iterdir()}}
    write(training / 'training-result.json', result)
    return training, cfg, targets


class AdapterAuthenticationTests(unittest.TestCase):
    def test_compact_peft_summary_covers_exact_full_recorded_targets(self):
        targets = [f'model.language_model.layers.{layer}.mlp.{name}' for layer in range(2)
                   for name in ('gate_proj', 'up_proj', 'down_proj')]
        with tempfile.TemporaryDirectory() as directory:
            training, cfg, _ = parent(Path(directory), targets=targets)
            path = training / 'adapter/adapter_config.json'
            config = json.loads(path.read_text())
            def configure(patterns):
                config['target_modules'] = patterns; write(path, config)
                result = json.loads((training / 'training-result.json').read_text())
                result['adapter_files'][path.name] = digest(path); write(training / 'training-result.json', result)
            configure(['gate_proj', 'up_proj', 'down_proj'])
            self.assertEqual(verify_initial_adapter(training, cfg, targets)['target_count'], 6)
            for patterns in (['down_proj'], ['gate_proj', 'up_proj', 'down_proj', 'q_proj']):
                configure(patterns)
                with self.assertRaises(CoderError): verify_initial_adapter(training, cfg, targets)

    def test_supervisor_forwards_the_parent_to_training_and_rejects_it_for_inference(self):
        # Historical command construction only: both the policy and subprocess
        # execution are mocked. Real CLI refusal is covered by test_method_policy.
        sys.path.insert(0, str(ROOT))
        from ops import run54_gpu_phase as phase
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); proof = root / 'proof.json'; write(proof, {'snapshot': '/pinned/model'})
            cfg = {'revision': 'pinned'}
            command = ['phase', '--phase', 'train', '--output', str(root / 'run'),
                       '--model-proof', str(proof), '--init-training', str(root / 'parent')]
            with patch.object(sys, 'argv', command), patch.object(phase, 'ROOT', root), \
                 patch.object(phase, 'load_config', return_value=cfg), \
                 patch.object(phase, 'gpu_window', return_value=nullcontext()), \
                 patch.object(phase.signal, 'signal'), patch.dict(os.environ), \
                 patch('method_policy.blocked_legacy_operation'), \
                 patch.object(phase, 'run_logged') as run:
                phase.main()
            actual = run.call_args.args[0]
            self.assertEqual(actual[actual.index('--init-training') + 1], str(root / 'parent'))
            self.assertEqual(json.loads((root / 'run/phase.json').read_text())['init_training'], str(root / 'parent'))
            command[command.index('train')] = 'evaluate'
            command[command.index(str(root / 'run'))] = str(root / 'inference')
            with patch.object(sys, 'argv', command), patch('method_policy.blocked_legacy_operation'), \
                 self.assertRaisesRegex(CoderError, 'only valid for training'):
                phase.main()
            self.assertFalse((root / 'inference').exists())

    def test_completed_parent_has_an_explicit_new_optimizer_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            training, cfg, targets = parent(Path(directory))
            proof = verify_initial_adapter(training, cfg, targets)
            self.assertFalse(proof['optimizer_state_resumed'])
            self.assertEqual(proof['adapter_sha256'], digest(training / 'adapter/adapter_model.safetensors'))

    def test_changed_weight_or_recipe_is_rejected(self):
        for name in ('adapter/adapter_model.safetensors', 'source/recipe.py'):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                training, cfg, _ = parent(Path(directory))
                (training / name).write_bytes(b'changed')
                with self.assertRaises(CoderError): verify_initial_adapter(training, cfg)

    def test_unverified_extra_adapter_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            training, cfg, _ = parent(Path(directory))
            (training / 'adapter/extra.json').write_text('{}')
            with self.assertRaises(CoderError): verify_initial_adapter(training, cfg)

    def test_smoke_and_incomplete_parents_are_rejected(self):
        for name, update in [('training-result.json', {'smoke_only': True}),
                             ('completed-run-manifest.json', {'status': 'running'})]:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                training, cfg, _ = parent(Path(directory))
                record = json.loads((training / name).read_text()); record.update(update)
                write(training / name, record)
                with self.assertRaises(CoderError): verify_initial_adapter(training, cfg)

    def test_changed_base_rank_scale_or_scope_is_rejected(self):
        for key, value in [('model_id', 'other/model'), ('revision', '0' * 40),
                           ('lora_rank', 4), ('lora_alpha', 8), ('lora_target_scope', 'all-language')]:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                training, cfg, _ = parent(Path(directory))
                cfg = copy.deepcopy(cfg); cfg[key] = value
                with self.assertRaises(CoderError): verify_initial_adapter(training, cfg)

    def test_loaded_model_targets_must_match_the_authenticated_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            training, cfg, _ = parent(Path(directory))
            with self.assertRaises(CoderError): verify_initial_adapter(training, cfg, ['other.target'])


@unittest.skipUnless(importlib.util.find_spec('peft') and importlib.util.find_spec('torch'),
                     'Pinned training environment required for real CPU PEFT controls')
class AdapterLoadingTests(unittest.TestCase):
    def model(self):
        from peft import get_peft_model, LoraConfig
        from transformers import LlamaConfig, LlamaForCausalLM
        import torch
        torch.manual_seed(7)
        targets = ['model.layers.0.mlp.down_proj']
        model = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=24,
                    num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2))
        return get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                              bias='none', task_type='CAUSAL_LM', target_modules=targets)), targets

    def test_saved_weights_restore_identical_logits_and_remain_trainable(self):
        import torch
        from peft import get_peft_model_state_dict
        original, targets = self.model()
        with torch.no_grad():
            for name, parameter in original.named_parameters():
                if 'lora_' in name: parameter.copy_(torch.randn_like(parameter) * 0.02)
        with tempfile.TemporaryDirectory() as directory:
            training, cfg, _ = parent(Path(directory), original, targets)
            resumed, _ = self.model()
            proof = initialize_adapter(resumed, training, cfg, targets)
            self.assertTrue(proof['exact_loaded_tensor_equality'])
            tokens = torch.tensor([[1, 2, 3, 4]])
            original.eval(); resumed.eval()
            self.assertTrue(torch.equal(original(tokens).logits, resumed(tokens).logits))
            before = {k: v.detach().clone() for k, v in get_peft_model_state_dict(resumed).items()}
            resumed.train()
            optimizer = torch.optim.SGD((p for p in resumed.parameters() if p.requires_grad), lr=0.01)
            loss = resumed(input_ids=tokens, labels=tokens).loss
            self.assertTrue(torch.isfinite(loss).item())
            loss.backward()
            gradients = [p.grad for p in resumed.parameters() if p.requires_grad and p.grad is not None]
            self.assertTrue(gradients and all(torch.isfinite(gradient).all().item() for gradient in gradients))
            optimizer.step()
            after = get_peft_model_state_dict(resumed)
            self.assertTrue(any(not torch.equal(before[k], after[k]) for k in before))

    def test_authenticated_but_incomplete_tensor_set_is_rejected(self):
        from safetensors.torch import load_file, save_file
        model, targets = self.model()
        with tempfile.TemporaryDirectory() as directory:
            training, cfg, _ = parent(Path(directory), model, targets)
            weights = training / 'adapter/adapter_model.safetensors'
            tensors = load_file(str(weights)); tensors.pop(next(iter(tensors))); save_file(tensors, str(weights))
            result = json.loads((training / 'training-result.json').read_text())
            result['adapter_files'][weights.name] = digest(weights); write(training / 'training-result.json', result)
            with self.assertRaises(CoderError): initialize_adapter(model, training, cfg, targets)


if __name__ == '__main__': unittest.main()
