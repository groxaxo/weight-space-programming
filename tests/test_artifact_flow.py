"""Training-source → selected adapter → compiler → native-manifest contracts."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'tests')]
from coder_common import CoderError, digest, loads
from coder_artifacts import capture_recipe, verify_recipe
from compile_coder import compile_checkpoint
from convert_coder_gguf import convert
from test_compile_coder import fixture, TARGET
from ops.run54_gguf_phase import selected_identity


def write(path, data): path.write_text(json.dumps(data))


class ArtifactFlowTests(unittest.TestCase):
    def setup_flow(self, root):
        base, adapter, _ = fixture(root)
        cfg = loads((ROOT / 'configs/qwen35-4b.json').read_text())
        base.rename(root / cfg['revision']); base = root / cfg['revision']
        for name in ['tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja']:
            (base / name).write_text('{}')
        files = {p.name: {'size': p.stat().st_size, 'sha256': digest(p)} for p in base.iterdir()}
        proof_path = root / 'proof.json'
        write(proof_path, {'schema_version': 1, 'model_id': cfg['model_id'], 'revision': cfg['revision'],
                           'snapshot': str(base), 'files': files})
        training = root / 'training'; training.mkdir(); adapter.rename(training / 'adapter'); adapter = training / 'adapter'
        config = loads((adapter / 'adapter_config.json').read_text())
        config.update(base_model_name_or_path=str(base), target_modules=[TARGET]); write(adapter / 'adapter_config.json', config)
        hashes = capture_recipe(ROOT, training, ROOT / 'configs/qwen35-4b.json')
        write(training / 'run-manifest.json', {'recipe_sha256': hashes, 'model_id': cfg['model_id'], 'model_revision': cfg['revision']})
        write(training / 'completed-run-manifest.json', {'status': 'complete', 'exit_code': 0, 'job_id': 'synthetic'})
        write(training / 'training-result.json', {'status': 'trained_unpromoted', 'smoke_only': False,
              'before': {'eval_loss': .3}, 'after': {'eval_loss': .2},
              'adapter_files': {p.name: digest(p) for p in adapter.iterdir()}})
        write(training / 'adapter-targets.json', [TARGET])
        cp = training / 'checkpoints/checkpoint-1'; shutil.copytree(adapter, cp)
        write(cp / 'trainer_state.json', {'global_step': 1, 'log_history': [{'step': 1, 'epoch': 1, 'eval_loss': .2}]})
        cpp = root / 'llama'; (cpp / 'build/bin').mkdir(parents=True)
        (cpp / 'build/bin/llama-server').write_bytes(b'controlled-native-binary')
        converter = cpp / 'convert_hf_to_gguf.py'
        converter.write_text('import sys\nfrom pathlib import Path\nPath(sys.argv[sys.argv.index("--outfile")+1]).write_bytes(b"GGUFsynthetic-converter-control")\n')
        return base, training, proof_path, cpp, files, cfg

    def test_frozen_recipe_detects_missing_and_changed_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); _, training, *_ = self.setup_flow(root)
            verify_recipe(training)
            source = training / 'source/scripts/train_coder.py'
            source.write_text('changed')
            with self.assertRaises(CoderError): verify_recipe(training)
            source.unlink()
            with self.assertRaises(CoderError): verify_recipe(training)

    def test_conversion_manifest_is_consumable_and_tampering_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); base, training, proof, cpp, files, cfg = self.setup_flow(root)
            args = SimpleNamespace(output=root / 'native', compiled=None, lora_export=None, reuse_base=None,
                profile='candidate', config=ROOT / 'configs/qwen35-4b.json', model_proof=proof, llama_source=cpp, python=sys.executable)
            with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': ''}): convert(args)
            select = SimpleNamespace(conversion=args.output / 'conversion.json', profile='base', model_proof=proof)
            with patch('ops.run54_gguf_phase.CPP', cpp / 'build/bin'):
                artifact, identity = selected_identity(select, cfg)
                self.assertEqual(identity['artifact_sha256'], digest(artifact))
                artifact.write_bytes(b'GGUFchanged')
                with self.assertRaises(CoderError): selected_identity(select, cfg)

    def test_compiled_manifest_binds_provenance_and_failed_conversion_is_retryable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); base, training, proof, cpp, files, cfg = self.setup_flow(root)
            compiled = root / 'compiled'
            compile_checkpoint(base, training / 'adapter', [TARGET], files, compiled,
                               {'model_id': cfg['model_id'], 'revision': cfg['revision']})
            args = SimpleNamespace(output=root / 'native', compiled=compiled, lora_export=None, reuse_base=None,
                profile='candidate', config=ROOT / 'configs/qwen35-4b.json', model_proof=proof, llama_source=cpp, python=sys.executable)
            with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': ''}):
                with patch('convert_coder_gguf.subprocess.run', side_effect=OSError('injected converter failure')):
                    with self.assertRaises(OSError): convert(args)
                self.assertFalse(args.output.exists())
                self.assertFalse(list(root.glob('.native.converting-*')))
                result = convert(args)
            self.assertEqual(result['compile_record_sha256'], digest(args.output / 'compile-record.json'))
            self.assertEqual({r['profile'] for r in result['conversions']}, {'base', 'candidate'})
            self.assertTrue(all(Path(r['output']).is_file() for r in result['conversions']))

    @unittest.skipUnless(importlib.util.find_spec('torch') and sys.platform.startswith('linux'), 'full Linux training environment required')
    def test_selected_checkpoint_compiles_and_preserves_parent_recipe(self):
        import select_coder_checkpoint as selection
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); base, training, proof, cpp, files, cfg = self.setup_flow(root)
            output = root / 'selected'
            # Exercise retained export internals with the active CLI guard mocked.
            with patch('method_policy.blocked_legacy_operation'), \
                 patch.object(sys, 'argv', ['select', '--training', str(training), '--step', '1', '--output', str(output)]): selection.main()
            verify_recipe(output)
            result = compile_checkpoint(base, output / 'adapter', [TARGET], files, root / 'compiled')
            self.assertTrue(result['verification']['all_unedited_bytes_preserved'])

    @unittest.skipUnless(importlib.util.find_spec('torch') and sys.platform.startswith('linux'), 'full Linux training environment required')
    def test_failed_export_leaves_no_destination_and_can_be_retried(self):
        import select_coder_checkpoint as selection
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); _, training, *_ = self.setup_flow(root)
            output = root / 'selected'
            # Exercise retained export internals with the active CLI guard mocked.
            with patch('method_policy.blocked_legacy_operation'), \
                 patch.object(sys, 'argv', ['select', '--training', str(training), '--step', '1', '--output', str(output)]):
                with patch.object(selection.shutil, 'copytree', side_effect=OSError('injected copy failure')):
                    with self.assertRaises(OSError): selection.main()
                self.assertFalse(output.exists())
                self.assertFalse(list(root.glob('.selected.selecting-*')))
                selection.main()
                self.assertTrue((output / 'training-result.json').is_file())
