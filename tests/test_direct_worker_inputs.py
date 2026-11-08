"""CLI input snapshots survive path mutation during baseline authentication."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'tests')]
import direct_forward as forward
import direct_probe as probe
import direct_causal as causal
from direct_weights import load_lock, parent_identity
from pipeline import sha256
from test_direct_probe import first_assertion_fixture
from test_direct_causal_host import request as causal_request
from test_direct_forward import request as forward_request


class WorkerInputSnapshots(unittest.TestCase):
    def test_forward_verifier_binds_consumed_request_before_model_authentication(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); request = root / 'request.json'
            request.write_text(json.dumps(forward_request())); original = sha256(request)
            lock = load_lock()
            def authenticate(*args):
                request.write_text('{}')
                return {'status': 'baseline_verified', 'parent': parent_identity(lock)}
            with mock.patch.object(sys, 'argv', ['forward', '--mode', 'verify', '--request', str(request), '--output', str(root / 'worker.json')]), mock.patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': ''}), mock.patch.object(forward, 'verify_baseline', side_effect=authenticate), mock.patch.object(forward.signal, 'signal'), mock.patch.object(forward.signal, 'alarm'), mock.patch('builtins.print'):
                forward.main()
            report = json.loads((root / 'worker.json').read_text())
            self.assertEqual(report['request_sha256'], original)
            self.assertNotEqual(report['request_sha256'], sha256(request))

    def test_probe_and_causal_cli_bind_all_three_consumed_inputs_before_authentication(self):
        for module in (probe, causal):
            with self.subTest(worker=module.__name__), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                _, request, patient, source, lock = first_assertion_fixture()
                if module is causal:
                    request = causal_request(); request['lock_sha256'] = lock['_sha256']
                paths = {key: root / (key + '.json') for key in ('request', 'patient', 'source-request')}
                paths['patient'].write_text(json.dumps(patient)); paths['source-request'].write_text(json.dumps(source))
                request.update(patient_sha256=sha256(paths['patient']), source_request_sha256=sha256(paths['source-request']))
                paths['request'].write_text(json.dumps(request))
                original = {key: sha256(path) for key, path in paths.items()}
                def authenticate(*args):
                    for path in paths.values(): path.write_text('{}')
                    return {'status': 'baseline_verified', 'parent': parent_identity(lock)}
                cli = ['worker', '--request', str(paths['request']), '--patient', str(paths['patient']),
                       '--source-request', str(paths['source-request']), '--output', str(root / 'out')]
                if module is causal: cli += ['--donors', str(root / 'donors')]
                method = 'run_causal' if module is causal else 'run_probe'
                with mock.patch.object(sys, 'argv', cli), mock.patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': ''}), mock.patch.object(module, 'verify_baseline', side_effect=authenticate), mock.patch.object(module, method) as execute, mock.patch.object(module.signal, 'signal'), mock.patch.object(module.signal, 'alarm'):
                    module.main()
                actual = execute.call_args.kwargs
                if module is causal: actual = actual['bindings']
                self.assertEqual({name: actual[name] for name in ('request_sha256', 'patient_sha256', 'source_request_sha256')},
                                 {'request_sha256': original['request'], 'patient_sha256': original['patient'],
                                  'source_request_sha256': original['source-request']})
                proof = json.loads((root / 'out/request-proof.json').read_text())
                self.assertEqual(proof['request'], request)


if __name__ == '__main__': unittest.main()
