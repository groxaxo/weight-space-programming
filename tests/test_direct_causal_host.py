"""Admission regressions for the finite causal worker; no model is loaded."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from ops import direct_edit54 as host
from direct_causal import DONOR_NAMES, QUESTION
from direct_weights import MODEL_ID, REVISION
from pipeline import ValidationError


def request():
    return {'schema_version': 1, 'role': 'practice', 'question': QUESTION, 'stage': 1,
            'lock_sha256': 'a' * 64, 'protocol_sha256': 'b' * 64,
            'patient_sha256': 'c' * 64, 'source_request_sha256': 'd' * 64,
            'donor_files': {name: 'e' * 64 for name in DONOR_NAMES},
            'winner_group': None, 'stage1_result_sha256': None,
            'budget': {'max_forward_calls': 6, 'max_wall_seconds': 1700,
                       'postcheck_reserve_seconds': 120}}


class CausalHostTests(unittest.TestCase):
    def test_finite_practice_only_admission_rejects_changed_scope_and_budget(self):
        host.validate_host_request(request(), 'causal', 'a' * 64)
        mutations = [('role', 'independent-final'), ('stage', True), ('winner_group', 1),
                     ('donor_files', {'other-file.json': 'e' * 64}),
                     ('budget', {'max_forward_calls': 7, 'max_wall_seconds': 1700,
                                 'postcheck_reserve_seconds': 120})]
        for field, value in mutations:
            with self.subTest(field=field), self.assertRaises(ValidationError):
                host.validate_host_request({**request(), field: value}, 'causal', 'a' * 64)

    def test_donors_are_exclusive_hash_bound_copies(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / 'source'; source.mkdir()
            value = request()
            for name in DONOR_NAMES:
                content = name.encode()
                (source / name).write_bytes(content)
                value['donor_files'][name] = hashlib.sha256(content).hexdigest()
            target = root / 'frozen'
            host.freeze_donors(source, target, value)
            self.assertEqual({p.name for p in target.iterdir()}, DONOR_NAMES)
            for name in DONOR_NAMES:
                self.assertEqual((source / name).read_bytes(), (target / name).read_bytes())
            with self.assertRaises(FileExistsError):
                host.freeze_donors(source, target, value)
            (source / 'result.json').write_text('changed')
            with self.assertRaisesRegex(ValidationError, 'Donor hash differs'):
                host.freeze_donors(source, root / 'changed', value)

    def test_unbounded_or_symlink_donor_cannot_enter_the_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / 'source'; source.mkdir()
            for name in DONOR_NAMES:
                (source / name).write_bytes(b'{}')
            value = request()
            value['donor_files'] = {name: hashlib.sha256(b'{}').hexdigest() for name in DONOR_NAMES}
            first = sorted(DONOR_NAMES)[0]
            (source / first).unlink()
            (source / first).symlink_to(source / sorted(DONOR_NAMES)[1])
            with self.assertRaisesRegex(ValidationError, 'bounded regular file'):
                host.freeze_donors(source, root / 'frozen', value)

    def test_causal_mount_is_readonly_and_retains_existing_sandbox(self):
        command = host.container_command('wsp-direct-causal-test', Path('/code'), Path('/deps'),
                [], Path('/out'), Path('/request'), 10 * host.GIB, 'causal',
                patient=Path('/patient'), source_request=Path('/original'), donors=Path('/bound-donors'))
        self.assertIn('scripts/direct_causal.py', command)
        self.assertIn('type=bind,source=/bound-donors,target=/donors,readonly', command)
        self.assertEqual(command[command.index('--network') + 1], 'none')
        self.assertNotIn('--gpus', command)
        self.assertNotIn('--rm', command)
        mounts = [command[i + 1] for i, item in enumerate(command) if item == '--mount']
        self.assertEqual(sum(not item.endswith(',readonly') for item in mounts), 1)
        with self.assertRaisesRegex(ValidationError, 'frozen donor'):
            host.container_command('wsp-direct-causal-test', Path('/code'), Path('/deps'), [],
                Path('/out'), Path('/request'), host.GIB, 'causal',
                patient=Path('/patient'), source_request=Path('/original'))

    def test_causal_result_must_bind_the_exact_stage_donors_and_unchanged_parent(self):
        value = request()
        report = {'request_sha256': 'f' * 64, 'protocol_sha256': value['protocol_sha256'],
                'parent': {'model_id': MODEL_ID, 'revision': REVISION, 'lock_sha256': 'a' * 64},
                'training': False, 'role': 'practice', 'status': 'practice_only',
                'weights_edited': False, 'quality_evaluation': 'NOT_RUN',
                'patient_sha256': value['patient_sha256'],
                'source_request_sha256': value['source_request_sha256'],
                'stage': 1, 'question': QUESTION, 'donor_files': value['donor_files'],
                'parameters_unchanged': True, 'parameter_postcheck_complete': True, 'complete': True,
                'loaded_parameter_sha256_before': {'weight': '0' * 64},
                'loaded_parameter_sha256_after': {'weight': '0' * 64}}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory); (output / 'causal').mkdir()
            path = output / 'causal/result.json'; path.write_text(json.dumps(report))
            self.assertTrue(host.worker_evidence(output, 'causal', value, 'f' * 64, 'a' * 64)['worker_complete'])
            for field, changed in [('stage', 2), ('donor_files', {}), ('weights_edited', True),
                                   ('loaded_parameter_sha256_after', {'weight': '1' * 64})]:
                path.write_text(json.dumps({**report, field: changed}))
                with self.subTest(field=field), self.assertRaises(ValidationError):
                    host.worker_evidence(output, 'causal', value, 'f' * 64, 'a' * 64)


if __name__ == '__main__':
    unittest.main()
