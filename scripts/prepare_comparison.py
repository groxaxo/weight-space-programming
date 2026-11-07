#!/usr/bin/env python3
"""Freeze a CPU-prepared three-arm MATH-500 comparison; does not allocate GPUs."""
import argparse
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import sys
import urllib.request

import compare_gpu as c

DATASET_REPO = 'HuggingFaceH4/MATH-500'
DATASET_REV = '6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--repaired', type=Path, required=True)
    p.add_argument('--verification-python', type=Path, required=True)
    p.add_argument('--run-id', required=True)
    a = p.parse_args()
    if not a.run_id or any(x not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for x in a.run_id):
        p.error('run-id must contain only letters, numbers, dashes, underscores')
    project = Path(__file__).resolve().parents[1]
    run = project / 'runs' / a.run_id
    if run.exists():
        raise FileExistsError('Run already exists; inspect it instead of overwriting/resubmitting')
    for source in (a.base, a.repaired):
        if not (source / 'model.safetensors.index.json').is_file():
            raise FileNotFoundError('Checkpoint index missing: ' + str(source))
    if not a.verification_python.is_file():
        raise FileNotFoundError('Verification Python missing')
    url = f'https://huggingface.co/datasets/{DATASET_REPO}/resolve/{DATASET_REV}/test.jsonl'
    data_dir = project / 'data' / 'evaluation'
    data_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    ds = data_dir / ('math500-' + DATASET_REV + '.jsonl')
    if not ds.exists():
        with urllib.request.urlopen(url, timeout=60) as response:
            content = response.read(10 * 1024 * 1024 + 1)
        if len(content) > 10 * 1024 * 1024:
            raise ValueError('Dataset response exceeds safety bound')
        with ds.open('xb') as f:
            f.write(content)
        ds.chmod(0o600)
    blob = ds.read_bytes()
    meta_url = f'https://huggingface.co/api/datasets/{DATASET_REPO}/revision/{DATASET_REV}?blobs=true'
    with urllib.request.urlopen(meta_url, timeout=60) as response:
        info = json.load(response)
    if info['sha'] != DATASET_REV:
        raise ValueError('Dataset revision changed unexpectedly')
    file = next(x for x in info['siblings'] if x['rfilename'] == 'test.jsonl')
    if file.get('lfs'):
        actual, expected = hashlib.sha256(blob).hexdigest(), file['lfs']['sha256']
    else:
        actual = hashlib.sha1(f'blob {len(blob)}\0'.encode() + blob).hexdigest()
        expected = file['blobId']
    if actual != expected:
        raise ValueError('Dataset Git/LFS object hash mismatch')
    rows = [json.loads(line) for line in blob.splitlines() if line.strip()]
    if len(rows) != 500 or len({r['unique_id'] for r in rows}) != 500:
        raise ValueError('Expected the complete 500 unique test questions')
    packages = ('vllm','torch','transformers','numpy','safetensors','math-verify','latex2sympy2_extended','antlr4-python3-runtime','sympy')
    code = ('scripts/compare_gpu.py','scripts/gpu_admission.py','scripts/prepare_comparison.py','scripts/pipeline.py','reproduction.lock.json')
    manifest = {
        'schema_version': 1, 'run_id': a.run_id, 'project': str(project), 'run_dir': str(run),
        'source_base_commit': '17611902e37ff76e7bf0d5e7b3012e8bd55fffbe',
        'base': str(a.base.resolve()), 'repaired': str(a.repaired.resolve()),
        'verification_python': str(a.verification_python.absolute()),
        'evaluation_python': sys.executable,
        'dataset': {'repo': DATASET_REPO, 'revision': DATASET_REV, 'split': 'test', 'path': str(ds),
                    'source': url, 'sha256': hashlib.sha256(blob).hexdigest(), 'source_object_id': expected},
        'problem_count': 500, 'samples_per_problem': 4, 'seed': 0, 'batch_size': 16,
        'sampling': {'n': 1, 'temperature': 0.6, 'top_p': 0.95, 'top_k': 20, 'min_p': 0.0,
                     'presence_penalty': 1.5, 'frequency_penalty': 0.0, 'repetition_penalty': 1.0,
                     'max_tokens': 32768, 'skip_special_tokens': False},
        'engine': {'dtype': 'bfloat16', 'tensor_parallel_size': 1, 'max_model_len': 34816,
                   'gpu_memory_utilization': 0.80, 'max_num_seqs': 16, 'max_num_batched_tokens': 4096,
                   'enforce_eager': True, 'enable_prefix_caching': False, 'language_model_only': True, 'seed': 0},
        'arms': {arm: {'thinking': thinking, 'model_path': str((a.repaired if arm == 'repaired-thinking' else a.base).resolve())}
                 for arm, thinking in c.ARMS.items()},
        'runtime_versions': {name: metadata.version(name) for name in packages},
        'code_sha256': {rel: c.digest(project / rel) for rel in code},
        'protocol': 'matched-three-arm-math500-v1',
        'note': 'Local quality comparison. Full context/output budgets preserved. No agent has received held-out answers. Not exact published runtime/RNG replication.'
    }
    run.mkdir(parents=True, mode=0o700)
    c.atomic_json(run / 'manifest.json', manifest)
    print(json.dumps({'manifest': str(run / 'manifest.json'), 'sha256': c.canonical_hash(manifest), 'questions': 500, 'samples_per_arm': 2000}, indent=2))


if __name__ == '__main__':
    main()
