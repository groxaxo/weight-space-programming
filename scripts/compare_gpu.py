#!/usr/bin/env python3
"""Queue-owned, single-GPU, matched MATH-500 evaluation. No agent/API calls.

The manifest is generated once, pins the dataset and code, and is shared by all
three arms. Completed samples are atomically checkpointed. This is a local
benchmark harness, not an arbitrary-code agent or a security sandbox.
"""
from __future__ import annotations

import argparse
import fcntl
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any

ARMS = {'original-thinking': True, 'repaired-thinking': True, 'original-nonthinking': False}
INSTRUCTION = r'Please reason step by step, and put your final answer within \boxed{}.'
GIB = 1024 ** 3


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.' + path.name + '-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(value, f, indent=2, allow_nan=False)
            f.write('\n'); f.flush(); os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def save_record(path: Path, value: dict) -> None:
    """Atomic no-clobber publication; only the run's owner holds its arm lock."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.sample-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            with gzip.GzipFile(fileobj=f, mode='wb', mtime=0) as gz:
                gz.write(json.dumps(value, separators=(',', ':'), allow_nan=False).encode())
            f.flush(); os.fsync(f.fileno())
        os.link(name, path)
    finally:
        os.unlink(name)


def read_record(path: Path) -> dict:
    with gzip.open(path, 'rt') as f:
        return json.load(f)


def extract_answer(text: str) -> str | None:
    """Match the published evaluator: after last </think>, last boxed else fbox.

    An incomplete last box is an extraction failure, not a previous answer.
    """
    if '</think>' in text:
        text = text.rsplit('</think>', 1)[1].strip()
    idx = text.rfind('\\boxed')
    key = '\\boxed'
    if idx < 0:
        idx, key = text.rfind('\\fbox'), '\\fbox'
    if idx < 0:
        return None
    rest = text[idx + len(key):]
    r = rest.lstrip()
    if r.startswith('{'):
        depth = 0
        for k, char in enumerate(r):
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    return r[1:k].strip() or None
        return None
    if key == '\\boxed' and rest.startswith(' '):
        return rest.strip().split('$')[0].split('\n')[0].strip() or None
    return None


def prompt_ids(tokenizer: Any, problem: str, thinking: bool) -> list[int]:
    # Transformers versions differ in apply_chat_template's tokenized return type.
    # Render explicitly, then encode without duplicate BOS/EOS/template tokens.
    text = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': problem + '\n' + INSTRUCTION}],
        tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
    if not isinstance(text, str):
        raise TypeError('Expected a rendered chat-template string')
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not isinstance(ids, list) or not ids or any(type(x) is not int for x in ids):
        raise TypeError('Expected a nonempty list of integer token IDs')
    return ids


def sample_seed(question_id: str, sample: int, seed: int) -> int:
    return int.from_bytes(hashlib.sha256(f'{seed}:{question_id}:{sample}'.encode()).digest()[:4], 'big')


def mem_available() -> int:
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) * 1024
    raise RuntimeError('MemAvailable telemetry unavailable')


def resource_guard(root: Path) -> None:
    if shutil.disk_usage(root).free < 8 * GIB:
        raise RuntimeError('Stopping only this evaluation: less than 8 GiB free disk')
    if mem_available() < 8 * GIB:
        raise RuntimeError('Stopping only this evaluation: less than 8 GiB available host RAM')


def queue_identity() -> dict:
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    physical = os.environ.get('GPU_QUEUE_PHYSICAL_GPUS', '').split(',')
    job = os.environ.get('GPU_QUEUE_JOB_ID')
    if not job or len(visible) != 1 or not visible[0].startswith('GPU-') or len(physical) != 1 or not physical[0].isdigit():
        raise RuntimeError('Must run as an exclusive ONE-GPU GPUQ job; direct launches refused')
    return {'job_id': job, 'uuid': visible[0], 'physical_index': int(physical[0])}


def read_manifest(path: Path, check_runtime: bool = False) -> tuple[dict, str, list[dict]]:
    m = json.loads(path.read_text())
    if m.get('schema_version') != 1 or set(m.get('arms', {})) != set(ARMS):
        raise ValueError('Invalid experiment schema/arms')
    if m.get('samples_per_problem') != 4 or m.get('problem_count') != 500:
        raise ValueError('This frozen experiment requires 500 questions x 4 samples')
    if m['sampling']['max_tokens'] != 32768:
        raise ValueError('Frozen benchmark token budget must not silently change')
    for rel, sha in m['code_sha256'].items():
        p = Path(m['project']) / rel
        if digest(p) != sha:
            raise ValueError(f'Code changed since experiment was frozen: {rel}')
    ds = Path(m['dataset']['path'])
    if digest(ds) != m['dataset']['sha256']:
        raise ValueError('Dataset digest mismatch')
    rows = [json.loads(s) for s in ds.read_text().splitlines() if s.strip()]
    if len(rows) != 500 or len({r['unique_id'] for r in rows}) != 500:
        raise ValueError('MATH-500 row count/unique IDs mismatch')
    for r in rows:
        if not isinstance(r.get('problem'), str) or not isinstance(r.get('answer'), str):
            raise ValueError('Malformed problem/answer')
    for arm, thinking in ARMS.items():
        if m['arms'][arm].get('thinking') is not thinking:
            raise ValueError('Thinking-mode mapping changed')
    if check_runtime:
        for package, version in m['runtime_versions'].items():
            if importlib.metadata.version(package) != version:
                raise RuntimeError(f'Runtime version drift: {package}')
    return m, canonical_hash(m), rows


def load_records(m: dict, mh: str, arm: str, rows: list[dict]) -> dict[tuple[int, int], dict]:
    directory = Path(m['run_dir']) / arm / 'records'
    result = {}
    for p in sorted(directory.glob('*.json.gz')):
        r = read_record(p)
        if r.get('manifest_sha256') != mh or r.get('arm') != arm:
            raise ValueError('Refusing to mix records from different experiments')
        qi, si = r['question_index'], r['sample_index']
        if not (0 <= qi < len(rows) and 0 <= si < 4) or r['question_id'] != rows[qi]['unique_id']:
            raise ValueError('Invalid saved sample identity')
        if r['seed'] != sample_seed(r['question_id'], si, m['seed']):
            raise ValueError('Saved sample seed mismatch')
        if (qi, si) in result:
            raise ValueError('Duplicate saved sample')
        result[qi, si] = r
    return result


def bootstrap_interval(values: list[float], seed: int = 20261107) -> list[float]:
    """Resample QUESTIONS, not the four correlated samples independently."""
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(values, k=len(values))) / len(values) for _ in range(2000))
    return [means[49], means[1949]]


def describe(records: dict, nquestions: int) -> dict:
    rr = list(records.values())
    rates = [sum(bool(records[i, s]['correct']) for s in range(4)) / 4 for i in range(nquestions)]
    tokens = sorted(r['generated_tokens'] for r in rr)
    return {
        'completed_samples': len(rr), 'accuracy': statistics.mean(rates),
        'accuracy_95pct_question_bootstrap': bootstrap_interval(rates),
        'truncation_rate': sum(r['finish_reason'] == 'length' for r in rr) / len(rr),
        'extraction_failure_rate': sum(r['answer'] is None for r in rr) / len(rr),
        'grading_error_count': sum(r['grading_error'] is not None for r in rr),
        'prediction_parse_failure_count': sum(r['prediction_parse_failed'] for r in rr),
        'generated_tokens_total': sum(tokens), 'generated_tokens_mean': statistics.mean(tokens),
        'generated_tokens_median': statistics.median(tokens),
        'generated_tokens_p95': tokens[int(.95 * (len(tokens) - 1))],
        'per_question_accuracy': rates,
    }


def compare_if_complete(manifest: Path) -> dict:
    m, mh, rows = read_manifest(manifest)
    out = Path(m['run_dir'])
    out.mkdir(parents=True, exist_ok=True)
    with (out / '.comparison.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        arms = {}
        for arm in ARMS:
            records = load_records(m, mh, arm, rows)
            if len(records) != len(rows) * 4:
                return {'status': 'waiting_for_all_arms', 'incomplete_arm': arm, 'samples': len(records)}
            report = out / arm / 'summary.json'
            if not report.is_file():
                return {'status': 'waiting_for_summary', 'incomplete_arm': arm}
            arms[arm] = json.loads(report.read_text())
            if arms[arm].get('manifest_sha256') != mh or arms[arm].get('status') != 'complete':
                raise ValueError('Arm summary provenance mismatch')
        pairs = {}
        for a, b in [('repaired-thinking', 'original-thinking'), ('original-nonthinking', 'original-thinking'), ('repaired-thinking', 'original-nonthinking')]:
            delta = [x - y for x, y in zip(arms[a]['per_question_accuracy'], arms[b]['per_question_accuracy'])]
            pairs[f'{a}_minus_{b}'] = {'accuracy_difference': statistics.mean(delta), 'paired_question_bootstrap_95pct': bootstrap_interval(delta)}
        combined = {'status': 'complete', 'manifest_sha256': mh, 'arms': arms, 'paired_comparisons': pairs,
                    'limitations': ['Matched non-thinking sampling, not separately optimized non-thinking settings.',
                                    'Independent queued runs may start at different times. Timing is descriptive, not controlled hardware speed attribution.',
                                    'Not bit-for-bit reproduction of original random seeds, runtime, or reported scores.',
                                    'Public benchmark; no novel-edit discovery or contamination-free training claim.']}
        atomic_json(out / 'comparison.json', combined)
        lines = ['# Three-way MATH-500 comparison', '', '500 questions × 4 samples per arm; BF16; 32,768 output-token cap.', '',
                 '| Variant | Accuracy | 95% question-bootstrap CI | Truncated | Extraction failure | Output tokens |',
                 '|---|---:|---:|---:|---:|---:|']
        for arm, r in arms.items():
            lo, hi = r['accuracy_95pct_question_bootstrap']
            lines.append(f"| {arm} | {r['accuracy']:.2%} | {lo:.2%}–{hi:.2%} | {r['truncation_rate']:.2%} | {r['extraction_failure_rate']:.2%} | {r['generated_tokens_total']:,} |")
        lines += ['', '## Paired changes (percentage points)', '']
        for name, p in pairs.items():
            lo, hi = p['paired_question_bootstrap_95pct']
            lines.append(f"{name}: {p['accuracy_difference']*100:.2f} pp (95% CI {lo*100:.2f} to {hi*100:.2f}).")
        lines += ['', '## Limitations', ''] + combined['limitations']
        report = out / 'comparison.md'
        fd, tmp = tempfile.mkstemp(dir=out, prefix='.comparison-')
        try:
            with os.fdopen(fd, 'w') as f:
                f.write('\n'.join(lines) + '\n'); f.flush(); os.fsync(f.fileno())
            os.replace(tmp, report)
        finally:
            if os.path.exists(tmp): os.unlink(tmp)
        return {'status': 'complete', 'report': str(report)}


def run_arm(manifest: Path, arm: str) -> None:
    identity = queue_identity()
    m, mh, rows = read_manifest(manifest, check_runtime=True)
    root = Path(m['run_dir']); out = root / arm
    out.mkdir(parents=True, exist_ok=True)
    with (out / '.arm.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        records = load_records(m, mh, arm, rows)
        if len(records) == len(rows) * 4 and (out / 'summary.json').is_file():
            print(json.dumps(compare_if_complete(manifest))); return
        started = time.time()
        atomic_json(out / 'status.json', {'status': 'preflight', 'identity': identity, 'manifest_sha256': mh, 'started_utc_epoch': started})
        resource_guard(root)
        project = Path(m['project'])
        check = [m['verification_python'], str(project / 'scripts/pipeline.py')]
        subprocess.run(check + ['verify', '--base', m['base']], check=True, timeout=180)
        if arm == 'repaired-thinking':
            subprocess.run(check + ['verify-output', '--base', m['base'], '--output', m['repaired']], check=True, timeout=180)
        from math_verify import parse, verify
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(m['arms'][arm]['model_path'], local_files_only=True, trust_remote_code=False)
        prompts, gold = [], []
        for row in rows:
            ids = prompt_ids(tokenizer, row['problem'], ARMS[arm])
            if len(ids) + m['sampling']['max_tokens'] > m['engine']['max_model_len']:
                raise RuntimeError('Prompt plus full output budget exceeds context: no silent truncation')
            prompts.append(ids)
            parsed = parse('\\boxed{' + row['answer'] + '}', parsing_timeout=5)
            if not parsed:
                raise RuntimeError(f"Gold parse failed before model load: {row['unique_id']}")
            gold.append(parsed)
        os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')
        os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
        os.environ.setdefault('OMP_NUM_THREADS', '4')
        from vllm import LLM, SamplingParams
        import torch
        if torch.cuda.device_count() != 1 or '3090' not in torch.cuda.get_device_name(0):
            raise RuntimeError('Expected exactly one queue-admitted RTX 3090')
        identity['device_name'] = torch.cuda.get_device_name(0)
        atomic_json(out / 'status.json', {'status': 'loading_model', 'identity': identity, 'manifest_sha256': mh})
        llm = LLM(model=m['arms'][arm]['model_path'], tokenizer=m['arms'][arm]['model_path'], trust_remote_code=False, **m['engine'])
        tasks = [(i, s) for i in range(len(rows)) for s in range(4) if (i, s) not in records]
        generate_seconds = 0.0
        resumed = bool(records)
        for offset in range(0, len(tasks), m['batch_size']):
            resource_guard(root)
            batch = tasks[offset:offset + m['batch_size']]
            params = [SamplingParams(**m['sampling'], seed=sample_seed(rows[i]['unique_id'], s, m['seed'])) for i, s in batch]
            t = time.monotonic()
            outputs = llm.generate([{'prompt_token_ids': prompts[i]} for i, _ in batch], sampling_params=params, use_tqdm=False)
            duration = time.monotonic() - t; generate_seconds += duration
            if len(outputs) != len(batch):
                raise RuntimeError('vLLM output count mismatch')
            for (i, sample), request in zip(batch, outputs):
                if len(request.outputs) != 1 or request.prompt_token_ids != prompts[i]:
                    raise RuntimeError('vLLM output identity mismatch')
                output = request.outputs[0]
                if output.finish_reason not in ('stop', 'length'):
                    raise RuntimeError(f'Unexpected generation finish: {output.finish_reason!r}')
                answer = extract_answer(output.text)
                error, correct, parse_failed = None, False, False
                if answer is not None:
                    try:
                        pred = parse('\\boxed{' + answer + '}', parsing_timeout=5)
                        parse_failed = not bool(pred)
                        correct = bool(pred and verify(gold[i], pred, timeout_seconds=5))
                    except Exception as exc:
                        error = type(exc).__name__ + ': ' + str(exc)[:250]
                r = {'manifest_sha256': mh, 'arm': arm, 'question_index': i, 'sample_index': sample,
                     'question_id': rows[i]['unique_id'], 'seed': sample_seed(rows[i]['unique_id'], sample, m['seed']),
                     'correct': correct, 'answer': answer, 'grading_error': error, 'prediction_parse_failed': parse_failed,
                     'finish_reason': output.finish_reason, 'generated_tokens': len(output.token_ids),
                     'prompt_tokens': len(prompts[i]), 'text': output.text, 'gpu': identity,
                     'batch_generation_seconds': duration, 'batch_index': offset // m['batch_size']}
                save_record(out / 'records' / f'{i:04d}-{sample}.json.gz', r); records[i, sample] = r
            progress = {'status': 'running', 'manifest_sha256': mh, 'identity': identity, 'completed_samples': len(records),
                        'total_samples': len(rows) * 4, 'generation_seconds_this_attempt': generate_seconds}
            atomic_json(out / 'status.json', progress); print(json.dumps(progress), flush=True)
        report = describe(records, len(rows))
        report.update({'status': 'complete', 'arm': arm, 'manifest_sha256': mh, 'gpu': identity,
                       'generation_seconds_this_attempt': generate_seconds, 'resumed': resumed,
                       'wall_seconds_this_attempt': time.time() - started,
                       'runtime_versions': m['runtime_versions'], 'timing_note': 'This-attempt timing; do not infer total rate after a resume.'})
        atomic_json(out / 'summary.json', report)
        atomic_json(out / 'status.json', {'status': 'complete', 'manifest_sha256': mh, 'identity': identity, 'completed_samples': len(records)})
        print(json.dumps(compare_if_complete(manifest)), flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('validate', 'run', 'summarize'))
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--arm', choices=tuple(ARMS))
    a = p.parse_args()
    if a.action == 'validate':
        m, mh, rows = read_manifest(a.manifest, check_runtime=True)
        print(json.dumps({'status': 'manifest_valid', 'sha256': mh, 'questions': len(rows), 'samples_per_arm': len(rows) * 4}))
    elif a.action == 'summarize':
        print(json.dumps(compare_if_complete(a.manifest)))
    else:
        if not a.arm:
            p.error('--arm required for run')
        try:
            run_arm(a.manifest, a.arm)
        except Exception as exc:
            try:
                m = json.loads(a.manifest.read_text())
                atomic_json(Path(m['run_dir']) / a.arm / 'failure.json',
                            {'status': 'failed', 'error': type(exc).__name__ + ': ' + str(exc), 'time': time.time()})
            except Exception:
                pass
            raise


if __name__ == '__main__':
    main()
