#!/usr/bin/env python3
"""CPU-only Q8 conversion and immutable manifest producer for the native supervisor."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from coder_common import check, digest, load_config, loads, verify_model_proof, write_json
from pipeline import leaf, publish_directory, sync_directory


def verify_artifact(row):
    path = Path(row['output'])
    check(path.is_file() and not path.is_symlink() and path.stat().st_size == row['bytes']
          and digest(path) == row['sha256'], 'Converted artifact changed')
    with path.open('rb') as stream:
        check(stream.read(4) == b'GGUF', 'Not a GGUF artifact')
    return path


def convert(args):
    check(os.environ.get('CUDA_VISIBLE_DEVICES') in ('', '-1'), 'Conversion requires CUDA hidden')
    output = args.output.absolute()
    check(not os.path.lexists(output), 'Refusing existing conversion output')
    check(not (args.compiled and args.lora_export), 'Choose compiled or unmerged LoRA profile')
    check(args.profile != 'base' and Path(args.profile).name == args.profile and args.profile not in ('', '.', '..'), 'Invalid candidate profile')
    cfg = load_config(args.config)
    proof = loads(args.model_proof.read_text()); base = Path(proof['snapshot'])
    verify_model_proof(cfg, base, args.model_proof)
    source = args.llama_source.resolve()
    converter = source / 'convert_hf_to_gguf.py'
    protected = [converter, *sorted((source / 'conversion').rglob('*.py')), *sorted((source / 'gguf-py/gguf').rglob('*.py'))]
    source_hashes = {str(p.relative_to(source)): digest(p) for p in protected}
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.' + output.name + '.converting-', dir=output.parent))
    try:
        record = {'schema_version': 1, 'status': 'converted_unpromoted', 'quantization': 'Q8_0',
                  'model_id': cfg['model_id'], 'revision': cfg['revision'],
                  'model_proof_sha256': digest(args.model_proof), 'converter_source_sha256': source_hashes,
                  'producer_sha256': digest(Path(__file__)), 'created_at_utc': datetime.now(timezone.utc).isoformat(), 'conversions': []}
        env = dict(os.environ, PYTHONPATH=str(source / 'gguf-py'), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                   PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
        def quantize(label, directory):
            artifact = stage / (label + '-Q8_0.gguf')
            command = [args.python, str(converter), str(directory), '--outfile', str(artifact), '--outtype', 'q8_0',
                       '--use-temp-file', '--no-nextn', '--model-name', 'Qwen3.5-4B']
            with (stage / (label + '-conversion.log')).open('x') as log:
                subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1800)
            row = {'profile': label, 'source': str(directory), 'output': str(output / artifact.name),
                   'bytes': artifact.stat().st_size, 'sha256': digest(artifact), 'command_argv': command}
            verify_artifact({**row, 'output': str(artifact)})
            return row
        if args.reuse_base:
            previous = loads(args.reuse_base.read_text())
            check(previous['status'] == 'converted_unpromoted' and previous['quantization'] == 'Q8_0' and
                  previous['model_id'] == cfg['model_id'] and previous['revision'] == cfg['revision'], 'Different base conversion')
            rows = [r for r in previous['conversions'] if r['profile'] == 'base' and r.get('kind', 'model') == 'model']
            check(len(rows) == 1, 'Missing unique base conversion')
            verify_artifact(rows[0])
            record['conversions'].append(rows[0])
            record['reused_base_manifest_sha256'] = digest(args.reuse_base)
        else:
            record['conversions'].append(quantize('base', base))
        if args.compiled:
            compiled = loads((args.compiled / 'compile.json').read_text())
            check(compiled['status'] == 'compiled_unpromoted' and compiled['provenance']['model_id'] == cfg['model_id']
                  and compiled['provenance']['revision'] == cfg['revision'], 'Different compiled model')
            for name, expected in compiled['output_file_sha256'].items():
                check(digest(args.compiled / leaf(name)) == expected, 'Compiled input changed')
            shutil.copyfile(args.compiled / 'compile.json', stage / 'compile-record.json')
            record.update(compile_record_sha256=digest(stage / 'compile-record.json'), compiler_provenance=compiled['provenance'])
            record['conversions'].append(quantize(args.profile, args.compiled))
        if args.lora_export:
            exported = loads((args.lora_export / 'export.json').read_text())
            check(exported['status'] == 'exported_unpromoted' and exported['factor_dtype'] == 'F32' and
                  exported['model_id'] == cfg['model_id'] and exported['revision'] == cfg['revision'], 'Different adapter export')
            artifact = args.lora_export.resolve() / leaf(exported['artifact'])
            row = {'profile': args.profile, 'kind': 'lora', 'base_profile': 'base', 'output': str(artifact),
                   'bytes': exported['artifact_bytes'], 'sha256': exported['artifact_sha256'],
                   'export_record': str(output / 'lora-export.json'), 'export_record_sha256': digest(args.lora_export / 'export.json')}
            verify_artifact(row)
            shutil.copyfile(args.lora_export / 'export.json', stage / 'lora-export.json')
            record['conversions'].append(row)
        check(all(digest(source / n) == h for n, h in source_hashes.items()), 'Protected converter source changed')
        write_json(stage / 'conversion.json', record)
        for path in stage.iterdir():
            with path.open('rb') as stream: os.fsync(stream.fileno())
        sync_directory(stage); publish_directory(stage, output); sync_directory(output.parent)
        return record
    finally:
        if stage.exists(): shutil.rmtree(stage)


def main():
    from method_policy import blocked_legacy_operation
    blocked_legacy_operation('Legacy trained-checkpoint GGUF conversion')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[1] / 'configs/qwen35-4b-lr5e-6.json')
    parser.add_argument('--model-proof', type=Path, required=True)
    parser.add_argument('--llama-source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--compiled', type=Path)
    parser.add_argument('--lora-export', type=Path)
    parser.add_argument('--reuse-base', type=Path, help='Authenticate and reuse base Q8 from a prior conversion manifest')
    parser.add_argument('--profile', default='candidate')
    parser.add_argument('--python', default=sys.executable)
    args = parser.parse_args()
    result = convert(args)
    print(__import__('json').dumps({'status': result['status'], 'manifest': str(args.output / 'conversion.json')}))


if __name__ == '__main__': main()
