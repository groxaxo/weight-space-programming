"""Authenticate and load a completed LoRA as the start of a new optimizer run."""
from pathlib import Path

from coder_artifacts import verify_recipe
from coder_common import adapter_identity, check, digest, loads


def verify_initial_adapter(training, cfg, expected_targets=None):
    training = Path(training)
    run = verify_recipe(training)
    completed = loads((training / 'completed-run-manifest.json').read_text())
    result = loads((training / 'training-result.json').read_text())
    check(completed.get('status') == 'complete' and completed.get('exit_code') == 0
          and result.get('status') == 'trained_unpromoted' and result.get('smoke_only') is False,
          'Initial adapter must come from a completed non-smoke training run')
    check(run.get('model_id') == cfg['model_id'] and run.get('model_revision') == cfg['revision'],
          'Initial adapter base identity differs from this recipe')
    previous = run.get('config', {})
    for key in ('lora_rank', 'lora_alpha', 'lora_target_scope'):
        check(previous.get(key, 'all-language' if key == 'lora_target_scope' else None)
              == cfg.get(key, 'all-language' if key == 'lora_target_scope' else None),
              'Initial adapter recipe differs: ' + key)
    adapter = training / 'adapter'
    files = result.get('adapter_files', {})
    identity = adapter_identity(result)
    check(set(files) == {p.name for p in adapter.iterdir()}, 'Initial adapter has missing or unverified files')
    for name, expected in files.items():
        path = adapter / name
        check(Path(name).name == name and path.is_file() and not path.is_symlink()
              and digest(path) == expected, 'Initial adapter artifact changed: ' + name)
    config = loads((adapter / 'adapter_config.json').read_text())
    from compile_coder import validate_adapter
    rank, _ = validate_adapter(config)
    check(rank == cfg['lora_rank'] and config['lora_alpha'] == cfg['lora_alpha']
          and config.get('lora_dropout') == 0.05 and config.get('task_type') == 'CAUSAL_LM',
          'Initial adapter configuration differs from the training construction')
    targets = loads((training / 'adapter-targets.json').read_text())
    patterns = config.get('target_modules')
    check(isinstance(targets, list) and targets and all(isinstance(t, str) and t for t in targets)
          and len(targets) == len(set(targets)), 'Invalid full initial adapter targets')
    # PEFT minimizes a long explicit target list to suffixes when saving.
    # Those suffixes are a summary, not the authority used for injection:
    # construct from this run's full targets and require exact tensor keys below.
    matches = lambda target, pattern: target == pattern or target.endswith('.' + pattern)
    check(isinstance(patterns, list) and patterns and all(isinstance(p, str) and p for p in patterns)
          and all(any(matches(t, p) for p in patterns) for t in targets)
          and all(any(matches(t, p) for t in targets) for p in patterns),
          'Initial adapter targets differ from its recorded configuration')
    if expected_targets is not None:
        check(set(targets) == set(expected_targets), 'Initial adapter targets differ from the loaded model')
    return {'mode': 'authenticated_adapter_new_optimizer', 'training': str(training.resolve()),
            **identity, 'model_id': cfg['model_id'], 'model_revision': cfg['revision'],
            'run_manifest_sha256': digest(training / 'run-manifest.json'),
            'completed_run_manifest_sha256': digest(training / 'completed-run-manifest.json'),
            'adapter_targets_sha256': digest(training / 'adapter-targets.json'),
            'recipe_sha256': run['recipe_sha256'], 'target_count': len(targets),
            'configuration_target_patterns': patterns,
            'optimizer_state_resumed': False}


def initialize_adapter(model, training, cfg, targets):
    """Require exact saved keys, shapes, dtypes and values after PEFT loading."""
    import torch
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    from safetensors.torch import load_file

    proof = verify_initial_adapter(training, cfg, targets)
    saved = load_file(str(Path(training) / 'adapter/adapter_model.safetensors'), device='cpu')
    current = get_peft_model_state_dict(model, save_embedding_layers=False)
    check(set(saved) == set(current) and len(saved) == 2 * len(targets),
          'Initial adapter tensor keys differ from the constructed trainable adapter')
    for name, value in saved.items():
        check(value.dtype == torch.float32 and value.shape == current[name].shape
              and value.dtype == current[name].dtype and torch.isfinite(value).all().item(),
              'Initial adapter tensor shape, dtype or values are invalid: ' + name)
    outcome = set_peft_model_state_dict(model, saved, ignore_mismatched_sizes=False)
    check(not outcome.unexpected_keys and not any('lora_' in key for key in outcome.missing_keys),
          'PEFT did not load the complete initial adapter')
    observed = get_peft_model_state_dict(model, save_embedding_layers=False)
    check(all(torch.equal(observed[name].detach().cpu(), value) for name, value in saved.items()),
          'Loaded initial adapter differs from its authenticated saved tensors')
    trainable = [parameter for name, parameter in model.named_parameters() if 'lora_' in name]
    check(len(trainable) == len(saved) and all(p.requires_grad for p in trainable),
          'Initial adapter is not entirely trainable')
    return {**proof, 'tensor_count': len(saved), 'exact_loaded_tensor_equality': True}
