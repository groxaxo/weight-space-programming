#!/usr/bin/env python3
"""Compile a standard language-only LoRA into a copied BF16 checkpoint on CPU.

Only one dense tensor update is materialized at a time. Original safetensors
headers, file layout, metadata and all unedited byte regions are preserved.
The compiled format is a new inference profile and requires fresh evaluation.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
import numpy as np

from coder_common import check, digest, load_config, loads, verify_model_proof, write_json
from pipeline import read_header, read_json, leaf, digest_regions, publish_directory, sync_directory


def to_bf16(values):
    values = np.ascontiguousarray(values, dtype='<f4')
    check(np.isfinite(values).all(), 'Non-finite merge result')
    bits = values.view('<u4')
    # IEEE round-to-nearest, ties-to-even. Truncating the low 16 bits is incorrect.
    rounded = ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) >> 16).astype('<u2')
    decoded = (rounded.astype('<u4') << 16).view('<f4')
    check(np.isfinite(decoded).all(), 'BF16 merge overflow')
    return rounded.tobytes()


def read_array(path, info, start, dtype):
    a, b = info['data_offsets']
    with path.open('rb') as f:
        f.seek(start+a)
        raw = f.read(b-a)
    check(len(raw) == b-a, 'Truncated tensor data')
    values = np.frombuffer(raw, dtype=dtype).reshape(info['shape'])
    if dtype == '<u2':
        values = (values.astype('<u4') << 16).view('<f4')
    check(np.isfinite(values).all(), 'Non-finite merge input')
    return values, raw


def validate_adapter(config):
    check(config.get('peft_type') == 'LORA' and config.get('bias') == 'none', 'Only ordinary bias-free LoRA is supported')
    check(not any(config.get(k) for k in ['use_dora', 'use_rslora', 'fan_in_fan_out', 'rank_pattern',
          'alpha_pattern', 'modules_to_save', 'layer_replication', 'target_parameters', 'alora_invocation_tokens']),
          'Unsupported LoRA variant')
    rank, alpha = config['r'], config['lora_alpha']
    check(type(rank) is int and 0 < rank <= 256, 'Invalid LoRA rank')
    check(isinstance(alpha, (int, float)) and not isinstance(alpha, bool) and math.isfinite(alpha) and alpha > 0,
          'Invalid LoRA alpha')
    return rank, np.float32(alpha/rank)


def verify_compiled(base, output, record):
    """Validate written tensors, headers and every unedited byte against the source."""
    grouped = {}
    index = read_json(base/'model.safetensors.index.json')['weight_map']
    seen = set()
    for row in record['tensor_checks']:
        name = row['name']
        check(name not in seen and index[name] == row['shard'], 'Invalid or duplicate compiled tensor')
        seen.add(name)
        header, start = read_header(base/row['shard'])
        a, b = header[name]['data_offsets']
        check((row['offset'], row['bytes']) == (start+a, b-a), 'Incorrect compiled tensor span')
        path = output/row['shard']
        check(path.is_file() and not path.is_symlink(), 'Missing regular compiled shard')
        with path.open('rb') as f:
            f.seek(start+a)
            check(hashlib.sha256(f.read(b-a)).hexdigest() == row['merged_sha256'], 'Compiled tensor changed')
        grouped.setdefault(row['shard'], []).append((start+a, start+b))
    for name, expected in record['output_file_sha256'].items():
        path = output/leaf(name)
        check(path.is_file() and not path.is_symlink() and digest(path) == expected, 'Compiled file changed: '+name)
        spans = grouped.get(name, [])
        if spans:
            check(digest_regions(base/name, spans) == digest_regions(path, spans), 'Unedited bytes changed: '+name)
        else:
            check(digest(base/name) == expected, 'Unedited file changed: '+name)
    return {'status': 'compiled_output_verified', 'edited_tensors': len(seen), 'all_unedited_bytes_preserved': True}


def compile_checkpoint(base, adapter, targets, files, output, provenance=None):
    base, adapter, output = Path(base), Path(adapter), Path(output).absolute()
    check(not os.path.lexists(output), 'Refusing existing compiled output')
    for source in (base, adapter):
        check(not output.resolve().is_relative_to(source.resolve()) and not source.resolve().is_relative_to(output.resolve()),
              'Compiled output overlaps an immutable input')
    check(targets and len(set(targets)) == len(targets) and all(isinstance(t, str) and re.fullmatch(
          r'model\.language_model\.layers\.\d+\..+', t) for t in targets), 'Invalid language-layer targets')
    cfg = read_json(adapter/'adapter_config.json')
    rank, scale = validate_adapter(cfg)
    weights = adapter/'adapter_model.safetensors'
    check(weights.is_file() and not weights.is_symlink(), 'Missing regular LoRA weights')
    lora_header, lora_start = read_header(weights)
    expected = {f'base_model.model.{t}.lora_{part}.weight' for t in targets for part in ('A', 'B')}
    check(set(lora_header)-{'__metadata__'} == expected, 'Incomplete or unexpected LoRA tensors')
    index = read_json(base/'model.safetensors.index.json')['weight_map']
    shards = {leaf(v) for v in index.values()}
    check(shards == {n for n in files if n.endswith('.safetensors')}, 'Checkpoint shard/index mismatch')
    headers = {name: read_header(base/name) for name in shards}
    for name, (header, _) in headers.items():
        check(set(header)-{'__metadata__'} == {k for k,v in index.items() if v == name}, 'Checkpoint header/index mismatch')
    for name, spec in files.items():
        path = base/leaf(name)
        check(path.is_file() and path.stat().st_size == spec['size'] and digest(path) == spec['sha256'], 'Base authentication failed: '+name)
    output.parent.mkdir(parents=True, exist_ok=True)
    check(shutil.disk_usage(output.parent).free >= sum(s['size'] for s in files.values()) + 64*1024**2,
          'Insufficient free space for copied checkpoint')
    staged = Path(tempfile.mkdtemp(prefix='.'+output.name+'.building-', dir=output.parent))
    try:
        for name in files:
            shutil.copyfile(base/name, staged/name)
            check(digest(staged/name) == files[name]['sha256'], 'Copied base authentication failed: '+name)
        checks = []
        for target in targets:
            name = target+'.weight'
            check(name in index, 'LoRA target is absent from base: '+name)
            shard = index[name]; header, start = headers[shard]; info = header[name]
            check(info['dtype'] == 'BF16' and len(info['shape']) == 2, 'Expected a BF16 matrix target')
            original, raw = read_array(staged/shard, info, start, '<u2')
            matrices = []
            for part in ('A', 'B'):
                entry = lora_header[f'base_model.model.{target}.lora_{part}.weight']
                check(entry['dtype'] == 'F32' and len(entry['shape']) == 2, 'Expected FP32 LoRA matrix')
                values, _ = read_array(weights, entry, lora_start, '<f4')
                matrices.append(values)
            a, b = matrices
            check(a.shape == (rank, original.shape[1]) and b.shape == (original.shape[0], rank), 'LoRA matrix shape mismatch')
            with np.errstate(over='raise', invalid='raise'):
                merged = to_bf16(original + (b @ a)*scale)
            offset = start+info['data_offsets'][0]
            with (staged/shard).open('r+b') as f:
                f.seek(offset); check(f.write(merged) == len(merged), 'Short compiled tensor write')
            checks.append({'name': name, 'shard': shard, 'offset': offset, 'bytes': len(merged),
                'base_sha256': hashlib.sha256(raw).hexdigest(), 'merged_sha256': hashlib.sha256(merged).hexdigest()})
            # No dense update is retained across tensors.
            del original, raw, matrices, a, b, merged
        record = {'schema_version': 1, 'status': 'compiled_unpromoted',
            'created_at_utc': datetime.now(timezone.utc).isoformat(), 'numpy': np.__version__,
            'arithmetic': 'FP32 NumPy B@A, alpha/r scaling, BF16 round-to-nearest ties-to-even',
            'rank': rank, 'scale': float(scale), 'tensor_checks': checks,
            'adapter_files': {n: digest(adapter/n) for n in ['adapter_config.json','adapter_model.safetensors']},
            'source_files': files, 'source_base': str(base.resolve()), 'source_adapter': str(adapter.resolve()),
            'provenance': provenance or {},
            'compiler_sha256': digest(Path(__file__)), 'output_file_sha256': {n: digest(staged/n) for n in files},
            'scope': 'CPU weight compilation only. This new inference profile requires independent evaluation.'}
        record['verification'] = verify_compiled(base, staged, record)
        write_json(staged/'compile.json', record)
        for p in staged.iterdir():
            with p.open('rb') as f: os.fsync(f.fileno())
        sync_directory(staged); publish_directory(staged, output); sync_directory(output.parent)
        return record
    finally:
        if staged.exists(): shutil.rmtree(staged)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--model-proof', type=Path, required=True)
    parser.add_argument('--training', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    proof = loads(args.model_proof.read_text()); base = Path(proof['snapshot'])
    verify_model_proof(cfg, base, args.model_proof)
    training = loads((args.training/'training-result.json').read_text())
    manifest = loads((args.training/'run-manifest.json').read_text())
    check(training['status'] == 'trained_unpromoted' and training['smoke_only'] is False, 'Need a complete training candidate')
    check(manifest['model_id'] == cfg['model_id'] and manifest['model_revision'] == cfg['revision'], 'Different training base')
    adapter = args.training/'adapter'
    for name, expected in training['adapter_files'].items():
        path = adapter/leaf(name)
        check(path.is_file() and not path.is_symlink() and digest(path) == expected, 'Adapter artifact changed')
    targets = loads((args.training/'adapter-targets.json').read_text())
    provenance = {'model_id': cfg['model_id'], 'revision': cfg['revision'],
        'model_proof_sha256': digest(args.model_proof),
        'training_result_sha256': digest(args.training/'training-result.json'),
        'run_manifest_sha256': digest(args.training/'run-manifest.json'),
        'adapter_targets_sha256': digest(args.training/'adapter-targets.json')}
    result = compile_checkpoint(base, adapter, targets, proof['files'], args.output, provenance)
    print({'status': result['status'], 'output': str(args.output), 'edited_tensors': len(result['tensor_checks']),
           'all_unedited_bytes_preserved': result['verification']['all_unedited_bytes_preserved']})


if __name__ == '__main__': main()
