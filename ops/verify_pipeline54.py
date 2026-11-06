#!/usr/bin/env python3
"""Bounded real-model pipeline control on .54; its small adapter is unpromoted.

Uses 16/4 authenticated public conversations for a complete two-step epoch, then
checks export, compilation, native manifest production and installed agent paths.
Run under systemd with this command's --restore-only as ExecStopPost.
"""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import signal
import sys
import time
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from coder_common import check, digest, encode_record, load_config, loads, verify_model_proof, write_json
from prepare_coder_data import read_dataset
from ops.run54_gpu_phase import gpu_window, restore, run_logged, serve_and_evaluate

IMAGE = 'python:3.12-slim@sha256:4840d514c583a4d3d8cad3940e68dee6f0f6f3635c12fcd0a3d74c21a5b93a54'
CPP = Path('/home/op/src/llama.cpp-prismml-mtp/src')


def wait_bonsai_health(seconds=180):
    deadline = time.monotonic() + seconds
    while True:
        try:
            with urlopen('http://127.0.0.1:18801/health', timeout=10) as response:
                health = loads(response.read())
            check(health.get('status') == 'ok', 'Bonsai is still warming up')
            return health
        except (OSError, ValueError):
            check(time.monotonic() < deadline, 'Bonsai health did not recover within the bounded wait')
            time.sleep(2)


def prepare_control(output, cfg, proof_path):
    from transformers import AutoTokenizer
    proof = loads(proof_path.read_text())
    verify_model_proof(cfg, Path(proof['snapshot']), proof_path)
    tokenizer = AutoTokenizer.from_pretrained(proof['snapshot'], local_files_only=True, trust_remote_code=False)
    parent = ROOT / 'assets/coder-corpus-v1/prepared'
    manifest, splits = read_dataset(parent)
    data = output / 'data'; data.mkdir()
    selected = {}
    for split, code_count, tool_count in [('train', 14, 2), ('validation', 3, 1)]:
        pools = {'code': [], 'tools': []}
        for row in splits[split]:
            size = len(encode_record(tokenizer, row, cfg['max_length'])['input_ids'])
            if size <= 1024:
                pools['tools' if row['tools'] else 'code'].append((size, row['task_id'], row))
        records = []
        for category, count in [('code', code_count), ('tools', tool_count)]:
            eligible = sorted(pools[category], key=lambda item: item[:2])
            check(len(eligible) >= count, 'Insufficient short public examples for control')
            records.extend(item[2] for item in eligible[:count])
        (data / (split + '.jsonl')).write_text(''.join(__import__('json').dumps(row) + '\n' for row in records))
        selected[split] = {'count': len(records), 'sha256': digest(data / (split + '.jsonl')),
                           'tasks': sorted({r['task_id'] for r in records})}
    write_json(data / 'dataset.json', {'schema_version': 1, 'sources': manifest['sources'], 'splits': selected,
        'parent_manifest_sha256': digest(parent / 'dataset.json'), 'purpose': 'bounded pipeline control; not model qualification'})
    read_dataset(data)
    control = {**cfg, 'max_length': 1024, 'epochs': 1, 'minimum_train_examples': 16, 'minimum_validation_examples': 4,
               'purpose': 'pipeline verification only'}
    write_json(output / 'control-config.json', control)
    write_json(output / 'inputs.json', {'model_proof_sha256': digest(proof_path), 'dataset': selected,
                                       'config': control, 'snapshot': proof['snapshot']})
    return proof['snapshot']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-pause-bonsai', action='store_true')
    parser.add_argument('--restore-only', action='store_true')
    parser.add_argument('--humaneval', action='store_true', help='Also regenerate all 164 tasks for both HF profiles')
    parser.add_argument('--base-conversion', type=Path, default=ROOT / 'runs/gguf-profile-q8-01/conversion.json')
    args = parser.parse_args(); output = args.output.absolute()
    if args.restore_only:
        restore(output / 'native'); restore(output)
        return
    check(os.uname().nodename == '4790', 'This installed-model control is for .54')
    check(not output.exists(), 'Refusing existing verification output')
    check(args.allow_pause_bonsai, 'Verification needs explicit Bonsai pause authorization')
    output.mkdir(parents=True)
    def terminate(signum, _frame): raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, terminate); signal.signal(signal.SIGINT, terminate)
    os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', TOKENIZERS_PARALLELISM='false',
                      PYTORCH_ALLOC_CONF='expandable_segments:True')
    python = str(ROOT / '.venv-train/bin/python')
    config = ROOT / 'configs/qwen35-4b-lr5e-6.json'; cfg = load_config(config)
    proof = ROOT / 'reports/model-inputs.json'
    snapshot = prepare_control(output, cfg, proof)
    def run(command, name, timeout=1800):
        run_logged([python, '-u', *command], output, name + '.log', timeout)
    try:
        with gpu_window(output, True):
            run(['scripts/train_coder.py', '--config', str(output / 'control-config.json'), '--data', str(output / 'data'),
                 '--output', str(output / 'training'), '--model-path', snapshot, '--model-proof', str(proof),
                 '--four-bit', '--direct-idle-gpu'], 'training')
            run(['scripts/select_coder_checkpoint.py', '--training', str(output / 'training'), '--step', '1',
                 '--output', str(output / 'selected')], 'selection')
            for label, adapter in [('hf-base', None), ('hf-control', output / 'selected/adapter')]:
                area = output / label; area.mkdir()
                options = argparse.Namespace(python=python, config=config, model_proof=proof, adapter=adapter,
                    port=18084, phase='evaluate', clients=True, humaneval=args.humaneval, image=IMAGE)
                serve_and_evaluate(options, area, cfg, snapshot)
                clients = loads((area / 'clients/acceptance.json').read_text())
                check(all(r.get('model_attestation', {}).get('verified') is True for r in clients['results']),
                      'Actual HF client attribution failed; inspect client reports')
            gate_command = [python, 'scripts/compare_coder.py', '--baseline', str(output / 'hf-base/evaluation.json'),
                            '--candidate', str(output / 'hf-control/evaluation.json'), '--output', str(output / 'development-gate.json')]
            if args.humaneval:
                gate_command.extend(['--baseline-humaneval', str(output / 'hf-base/humaneval.json'),
                    '--candidate-humaneval', str(output / 'hf-control/humaneval.json'),
                    '--clients', str(output / 'hf-control/clients/acceptance.json'),
                    '--training', str(output / 'selected/training-result.json')])
            run_logged(gate_command, output, 'gate.log', 60, accepted_codes=(0, 2))
        run(['scripts/compile_coder.py', '--config', str(config), '--model-proof', str(proof),
             '--training', str(output / 'selected'), '--output', str(output / 'compiled')], 'compilation')
        run(['scripts/export_coder_lora_gguf.py', '--llama-source', str(CPP), '--config', str(config),
             '--model-proof', str(proof), '--training', str(output / 'selected'), '--output', str(output / 'lora')], 'lora-export')
        run(['scripts/convert_coder_gguf.py', '--llama-source', str(CPP), '--config', str(config),
             '--model-proof', str(proof), '--reuse-base', str(args.base_conversion), '--lora-export', str(output / 'lora'),
             '--output', str(output / 'conversion'), '--profile', 'control'], 'conversion')
        run(['ops/run54_gguf_phase.py', '--phase', 'evaluate', '--conversion', str(output / 'conversion/conversion.json'),
             '--profile', 'control', '--output', str(output / 'native'), '--allow-pause-bonsai', '--clients'], 'native', 2400)
        health = wait_bonsai_health()
        reports = {}
        for label in ['hf-base', 'hf-control', 'native']:
            dev = loads((output / label / 'evaluation.json').read_text())
            clients = loads((output / label / 'clients/acceptance.json').read_text())
            check(all(r.get('model_attestation', {}).get('verified') is True for r in clients['results']),
                  'Actual client attribution failed: ' + label)
            reports[label] = {'scores': dev['scores'], 'clients_passed': clients['passed'],
                              'client_requests': {r['client']: r.get('model_attestation') for r in clients['results']},
                              'identity': dev['identity']}
        write_json(output / 'verification.json', {'status': 'pipeline_verified', 'scope': 'Real bounded pipeline control; no adapter promoted',
            'created_at_utc': datetime.now(timezone.utc).isoformat(), 'profiles': reports,
            'training_steps': 2, 'humaneval_regenerated': args.humaneval, 'bonsai_health': health,
            'gate': loads((output / 'development-gate.json').read_text()),
            'compiled_tensors': loads((output / 'compiled/compile.json').read_text())['verification'],
            'source_sha256': {str(p.relative_to(ROOT)): digest(p) for folder in ['scripts', 'ops'] for p in (ROOT / folder).glob('*.py')}})
    except BaseException as error:
        write_json(output / 'verification-failed.json', {'error': f'{type(error).__name__}: {error}'})
        raise


if __name__ == '__main__': main()
