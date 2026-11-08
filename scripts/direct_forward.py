#!/usr/bin/env python3
"""Trusted, bounded, CPU-only BF16 practice observer. No patch selection or training.

Run only through ops/direct_edit54.py's no-network container. Arbitrary numerical
programs, generated code and independent final cases are not accepted here.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import signal
import time

from direct_weights import load_lock, read_json, verify_baseline, publish_json
from pipeline import require, sha256, read_header, CHUNK


def validate_request(request, lock):
    require(set(request) == {'schema_version', 'role', 'lock_sha256', 'protocol_sha256',
                            'cases', 'generation'}, 'Unexpected practice request fields')
    require(type(request['schema_version']) is int and request['schema_version'] == 1,
            'Unsupported request schema')
    require(request['role'] == 'practice', 'Only exposed practice is admitted by this worker')
    require(request['lock_sha256'] == lock['_sha256'], 'Wrong parent lock identity')
    require(isinstance(request['protocol_sha256'], str) and len(request['protocol_sha256']) == 64
            and all(c in '0123456789abcdef' for c in request['protocol_sha256']), 'Invalid protocol hash')
    gen = request['generation']
    require(type(gen) is dict and all(type(gen.get(k)) is int for k in
            ('max_new_tokens', 'max_input_tokens', 'seed', 'threads'))
            and all(type(gen.get(k)) is bool for k in ('do_sample', 'thinking')),
            'Diagnostic generation settings must have exact numeric/boolean types')
    require(gen['max_new_tokens'] in (192, 384)
            and gen == {'max_new_tokens': gen['max_new_tokens'], 'max_input_tokens': 256, 'do_sample': False,
                        'thinking': False, 'seed': 20261108, 'threads': 1},
            'Only the frozen short CPU diagnostic profile is supported')
    cases = request['cases']
    require(isinstance(cases, list) and 1 <= len(cases) <= 2, 'Practice pilot supports one or two cases')
    names = set()
    for case in cases:
        require(set(case) == {'id', 'messages'}, 'Unexpected case fields')
        name = case['id']
        require(isinstance(name, str) and 1 <= len(name) <= 80
                and all(c in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for c in name)
                and name not in names, 'Invalid/duplicate case identity')
        names.add(name)
        messages = case['messages']
        require(isinstance(messages, list) and 1 <= len(messages) <= 6, 'Invalid conversation length')
        for message in messages:
            require(set(message) == {'role', 'content'} and message['role'] in ('system', 'user', 'assistant')
                    and isinstance(message['content'], str) and len(message['content']) <= 4000,
                    'Invalid practice message')


def write_new(path, value):
    # Serialize fully before publication; retain staging and never a partial final.
    publish_json(value, path)


def loading_metadata(info):
    """Installed Transformers returns sets in loading_info; preserve them as lists."""
    result = {}
    for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs'):
        values = info.get(key, [])
        require(isinstance(values, (list, tuple, set))
                and all(isinstance(value, str) for value in values), 'Unexpected loader metadata: ' + key)
        result[key] = sorted(values)
    result['original_field_types'] = {key: type(value).__name__ for key, value in info.items()}
    return result


class TokenJournal:
    """Preserve actual prompt/generated token IDs even if generation is interrupted."""
    def __init__(self, handle):
        self.handle = handle
        self.first = True

    def put(self, value):
        json.dump({'stage': 'prompt' if self.first else 'generated', 'ids': value.tolist()}, self.handle)
        self.handle.write('\n')
        self.handle.flush()
        self.first = False

    def end(self):
        self.handle.flush()


def parameter_sha256(parameter):
    import torch
    require(parameter.dtype in (torch.bfloat16, torch.float32) and parameter.is_contiguous(),
            'Expected contiguous official BF16/FP32 parameter')
    return hashlib.sha256(memoryview(parameter.detach().reshape(-1).view(torch.uint8).numpy()).cast('B')).hexdigest()


def select_cpu_reference_kernels():
    """Use installed Transformers' own Torch references, not GPU-only FLA dispatch.

    Selection is local to this isolated worker, recorded in its runtime identity,
    and changes no dependency files or model parameters.
    """
    from transformers.models.qwen3_5 import modeling_qwen3_5 as module
    selected = {}
    source_file = Path(module.__file__).resolve()
    for name in ('causal_conv1d_fn', 'causal_conv1d_update',
                 'torch_chunk_gated_delta_rule', 'torch_recurrent_gated_delta_rule'):
        function = inspect.unwrap(getattr(module, name))
        require(function.__module__ == module.__name__
                and Path(inspect.getsourcefile(function)).resolve() == source_file,
                'CPU reference did not resolve to installed Qwen implementation: ' + name)
        setattr(module, name, function)
        selected[name] = hashlib.sha256(inspect.getsource(function).encode()).hexdigest()
    return {'kind': 'installed-transformers-pure-torch-cpu-reference',
            'module_sha256': sha256(source_file), 'functions_sha256': selected}


def restore_parent_dtype(model, name, parameter, source, path, spec):
    """Undo loader dtype coercion; restore exact official auxiliaries, never fit them."""
    import torch
    from safetensors import safe_open
    require(spec['dtype'] in ('BF16', 'F32'), 'Unsupported official language parameter dtype')
    target_dtype = {'BF16': torch.bfloat16, 'F32': torch.float32}[spec['dtype']]
    require(list(parameter.shape) == spec['shape'], 'Loaded parameter shape differs from parent: ' + source)
    if parameter.dtype != target_dtype:
        with safe_open(str(path), framework='pt', device='cpu') as handle:
            exact = handle.get_tensor(source)
            require(exact.dtype == target_dtype and list(exact.shape) == spec['shape'],
                    'Original parameter dtype/shape differs from header')
            # Never cast back a rounded value. Reload the exact original words.
            replacement = torch.nn.Parameter(exact.clone(), requires_grad=False)
        module_name, attribute = name.rsplit('.', 1)
        model.get_submodule(module_name).register_parameter(attribute, replacement)
        return replacement, True
    return parameter, False


def load_text_model(lock, verified):
    import numpy as np
    import torch
    from transformers import AutoConfig, AutoTokenizer, Qwen3_5ForCausalLM
    require(not torch.cuda.is_available(), 'CPU observer must not have CUDA access')
    reference = select_cpu_reference_kernels()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cfg = AutoConfig.from_pretrained('/model', local_files_only=True, trust_remote_code=False)
    model, info = Qwen3_5ForCausalLM.from_pretrained(
        '/model', config=cfg.text_config, dtype=torch.bfloat16, device_map='cpu',
        attn_implementation='eager', local_files_only=True, trust_remote_code=False,
        key_mapping={r'^model\.language_model\.': 'model.'}, output_loading_info=True)
    require(not info.get('missing_keys') and not info.get('mismatched_keys')
            and not info.get('error_msgs'), 'Incomplete or mismatched official text model load')
    require(all(k.startswith(('model.visual.', 'mtp.')) for k in info.get('unexpected_keys', [])),
            'Unexpected non-vision parent parameters during text-only loading')
    info = loading_metadata(info)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        require(parameter.device.type == 'cpu' and parameter.dtype in (torch.bfloat16, torch.float32),
                'Unsupported or non-CPU model parameter')
    # Authenticate every loaded language parameter, including protected tensors,
    # against exact safetensor bytes, without copying a full embedding matrix.
    checked, dtype_restorations = {}, []
    index = read_json(Path('/model/model.safetensors.index.json'))['weight_map']
    headers = {shard: read_header(Path(spec['path'])) for shard, spec in verified['files'].items()
               if shard.endswith('.safetensors')}
    for name, parameter in model.named_parameters():
        source = name.replace('model.', 'model.language_model.', 1) if name.startswith('model.') else name
        require(source in index, 'Loaded parameter absent from the official parent: ' + source)
        shard = index[source]
        header, start = headers[shard]
        spec = header[source]
        parameter, restored = restore_parent_dtype(model, name, parameter, source,
                                                   Path(verified['files'][shard]['path']), spec)
        if restored:
            dtype_restorations.append({'name': source, 'restored_original_dtype': spec['dtype']})
        begin, end = spec['data_offsets']
        expected = hashlib.sha256()
        with Path(verified['files'][shard]['path']).open('rb') as handle:
            handle.seek(start + begin)
            remaining = end - begin
            while remaining:
                block = handle.read(min(CHUNK, remaining))
                require(bool(block), 'Truncated official parent tensor')
                expected.update(block)
                remaining -= len(block)
        actual = parameter_sha256(parameter)
        require(actual == expected.hexdigest(), 'Loaded parameter differs from official BF16 parent: ' + source)
        checked[name] = actual
    tokenizer = AutoTokenizer.from_pretrained('/model', local_files_only=True, trust_remote_code=False)
    info['official_dtype_restorations'] = dtype_restorations
    info['cpu_reference_kernels'] = reference
    return model, tokenizer, info, checked


def observe(request, lock, verified, outdir):
    import numpy as np
    import torch
    started = time.monotonic()
    model, tokenizer, loading, before = load_text_model(lock, verified)
    write_new(outdir / 'loaded-parent-verification.json', {
        'loaded_parameter_sha256': before, 'loading_info': loading,
        'quality_evaluation': 'NOT_RUN', 'scope': 'Exact official language parameter authentication before inference'})
    print(json.dumps({'loaded_original_parameters_verified': len(before),
                      'original_dtype_restorations': len(loading['official_dtype_restorations']),
                      'cpu_reference': loading['cpu_reference_kernels']['kind']}), flush=True)
    torch.manual_seed(request['generation']['seed'])
    results = []
    for case in request['cases']:
        ids = tokenizer.apply_chat_template(case['messages'], tokenize=True, add_generation_prompt=True,
                                            enable_thinking=False, return_tensors='pt')
        # New tokenizer APIs may return a BatchEncoding rather than the tensor.
        if not isinstance(ids, torch.Tensor):
            ids = ids['input_ids']
        require(ids.ndim == 2 and ids.shape[0] == 1 and ids.shape[1] <= 256,
                'Practice input exceeds the frozen CPU diagnostic token limit')
        arrays, hooks = {}, []
        def capture(layer):
            def hook(module, args, output):
                # First (prefill) call only: all layers refer to the same last prompt token.
                key = f'layer_{layer}_input_bf16'
                if key not in arrays:
                    require(not torch.is_grad_enabled(), 'Observation must run in inference mode')
                    arrays[key] = args[0][0, -1].detach().contiguous().view(torch.uint16).numpy().copy()
                    arrays[f'layer_{layer}_output_bf16'] = output[0, -1].detach().contiguous().view(torch.uint16).numpy().copy()
            return hook
        for layer, block in enumerate(model.model.layers):
            hooks.append(block.mlp.down_proj.register_forward_hook(capture(layer)))
        begin = time.monotonic()
        try:
            with torch.inference_mode(), (outdir / (case['id'] + '-token-journal.jsonl')).open('x') as journal:
                output = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                        do_sample=False, max_new_tokens=request['generation']['max_new_tokens'], use_cache=True,
                                        logits_to_keep=1, pad_token_id=tokenizer.eos_token_id,
                                        streamer=TokenJournal(journal))
        except BaseException as error:
            if arrays:
                partial = outdir / (case['id'] + '-interrupted-prefill.npz')
                with partial.open('xb') as handle:
                    np.savez(handle, **arrays)
                write_new(outdir / (case['id'] + '-interruption.json'), {
                    'error': type(error).__name__ + ': ' + str(error), 'observed_arrays': len(arrays),
                    'observation_sha256': sha256(partial), 'generation_complete': False})
            raise
        finally:
            for hook in hooks:
                hook.remove()
        require(len(arrays) == 64, 'Missing 4B language MLP observations')
        for layer in range(32):
            require(arrays[f'layer_{layer}_input_bf16'].shape == (9216,)
                    and arrays[f'layer_{layer}_output_bf16'].shape == (2560,), 'Hook shape mismatch')
        observation = outdir / (case['id'] + '-prefill.npz')
        # NumPy's writer otherwise replaces an existing path. Own exclusive handle instead.
        with observation.open('xb') as handle:
            np.savez(handle, **arrays)
        generated = output[0, ids.shape[1]:].tolist()
        results.append({'id': case['id'], 'input_ids': ids[0].tolist(),
                        'output_ids': generated, 'text': tokenizer.decode(generated, skip_special_tokens=True),
                        'input_tokens': ids.shape[1], 'generated_tokens': len(generated),
                        'hit_token_limit': len(generated) == request['generation']['max_new_tokens'],
                        'seconds': time.monotonic() - begin,
                        'observation_sha256': sha256(observation), 'observation': observation.name,
                        'observation_position': 'last prompt token, first prefill; all 32 down_proj inputs/outputs'})
        write_new(outdir / (case['id'] + '-generation.json'), results[-1])
        print(json.dumps({'case': case['id'], 'tokens': len(generated), 'seconds': results[-1]['seconds']}), flush=True)
    after = {name: parameter_sha256(model.get_parameter(name)) for name in before}
    require(after == before, 'Observed model weights changed during inference')
    return {'results': results, 'loaded_parameter_sha256_before': before,
            'loaded_parameter_sha256_after': after, 'loading_info': loading,
            'torch': torch.__version__, 'seconds': time.monotonic() - started,
            'scope': 'Short CPU BF16 practice diagnosis. Not a paired improvement result or full-context/client qualification.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('verify', 'observe'), required=True)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    def expired(_signum, _frame):
        raise TimeoutError('Own CPU worker reached its internal wall-time limit')
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(1700 if args.mode == 'observe' else 840)
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU isolation required')
    require(not args.output.exists(), 'Refusing an existing worker result')
    lock = load_lock()
    request = read_json(args.request)
    validate_request(request, lock)
    report = {'schema_version': 1, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'mode': args.mode, 'request_sha256': sha256(args.request),
              'protocol_sha256': request['protocol_sha256'], 'parent': None,
              'training': False, 'patch_applied': False, 'quality_improvement_claimed': False}
    verified = verify_baseline(lock, Path('/model'))
    report['parent'] = verified['parent']
    write_new(args.output.parent / 'baseline-verification.json', verified)
    print('Full immutable BF16 baseline authentication passed', flush=True)
    if args.mode == 'observe':
        report.update(observe(request, lock, verified, args.output.parent))
    write_new(args.output, report)


if __name__ == '__main__':
    main()
