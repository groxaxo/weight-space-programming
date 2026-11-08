"""CPU-only worker admission and sandbox contracts; no model is loaded here."""
import copy
import io
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import direct_forward
import direct_weights
from ops import direct_edit54
from pipeline import ValidationError


def request():
    return {'schema_version': 1, 'role': 'practice',
            'lock_sha256': direct_weights.load_lock()['_sha256'],
            'protocol_sha256': 'a' * 64,
            'cases': [{'id': 'practice-one', 'messages': [{'role': 'user', 'content': 'Implement f.'}]}],
            'generation': {'max_new_tokens': 192, 'max_input_tokens': 256, 'do_sample': False,
                           'thinking': False, 'seed': 20261108, 'threads': 1}}


class ForwardBoundaryTests(unittest.TestCase):
    def test_loading_metadata_handles_actual_set_fields_and_rejects_objects(self):
        info = {'missing_keys': set(), 'unexpected_keys': {'model.visual.b', 'model.visual.a'},
                'mismatched_keys': set(), 'error_msgs': []}
        result = direct_forward.loading_metadata(info)
        self.assertEqual(result['unexpected_keys'], ['model.visual.a', 'model.visual.b'])
        json.dumps(result, allow_nan=False)
        with self.assertRaises(ValidationError):
            direct_forward.loading_metadata({'unexpected_keys': {object()}})

    def test_invalid_report_does_not_leave_a_partial_final_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'result.json'
            with self.assertRaises(TypeError):
                direct_forward.write_new(path, {'bad': {object()}})
            self.assertFalse(path.exists())

    def test_token_journal_preserves_actual_steps_without_torch(self):
        import numpy as np
        handle = io.StringIO()
        journal = direct_forward.TokenJournal(handle)
        journal.put(np.asarray([[11, 22]], dtype=np.int64))
        journal.put(np.asarray([33], dtype=np.int64))
        journal.end()
        self.assertEqual([json.loads(line) for line in handle.getvalue().splitlines()],
                         [{'stage': 'prompt', 'ids': [[11, 22]]},
                          {'stage': 'generated', 'ids': [33]}])

    def test_only_frozen_practice_profile_is_admitted(self):
        lock = direct_weights.load_lock()
        direct_forward.validate_request(request(), lock)
        cases = []
        for key, value in [('role', 'independent-final'), ('schema_version', True),
                           ('lock_sha256', '0' * 64), ('protocol_sha256', 'x' * 64)]:
            changed = request(); changed[key] = value; cases.append(changed)
        changed = request(); changed['patch'] = {}; cases.append(changed)
        changed = request(); changed['generation']['do_sample'] = True; cases.append(changed)
        changed = request(); changed['cases'] *= 2; cases.append(changed)
        changed = request(); changed['cases'][0]['id'] = '../escape'; cases.append(changed)
        changed = request(); changed['cases'][0]['messages'][0]['role'] = 'tool'; cases.append(changed)
        for changed in cases:
            with self.subTest(changed=changed), self.assertRaises(ValidationError):
                direct_forward.validate_request(changed, lock)

    def test_resource_admission_preserves_host_reserves(self):
        gib = direct_edit54.GIB
        direct_edit54.admit({'mem_available_bytes': 14 * gib, 'disk_free_bytes': 2 * gib}, 10 * gib)
        for state in ({'mem_available_bytes': 14 * gib - 1, 'disk_free_bytes': 2 * gib},
                      {'mem_available_bytes': 20 * gib, 'disk_free_bytes': 2 * gib - 1}):
            with self.assertRaisesRegex(ValidationError, 'Not admitted'):
                direct_edit54.admit(state, 10 * gib)

    def test_host_schema_versions_follow_the_reviewed_worker_modes(self):
        original = {'schema_version': 1, 'role': 'practice', 'lock_sha256': 'a' * 64}
        direct_edit54.validate_host_request(original, 'observe', 'a' * 64)
        probe = {**original, 'schema_version': 2, 'budget': {'max_wall_seconds': 1700}}
        direct_edit54.validate_host_request(probe, 'probe', 'a' * 64)
        bad = [({**probe}, 'observe'), ({**probe, 'schema_version': True}, 'probe'),
               ({**probe, 'role': 'independent-final'}, 'probe'),
               ({**probe, 'patch': {}}, 'probe'),
               ({**probe, 'budget': {'max_wall_seconds': 1701}}, 'probe')]
        for value, mode in bad:
            with self.subTest(value=value, mode=mode), self.assertRaises(ValidationError):
                direct_edit54.validate_host_request(value, mode, 'a' * 64)

    def test_sandbox_has_no_gpu_network_home_or_socket_and_retains_failures(self):
        command = direct_edit54.container_command('wsp-direct-contract', Path('/frozen'),
                    Path('/dependencies'), [('/blob-one', '/model/file')], Path('/worker-out'),
                    Path('/frozen-request'), 10 * direct_edit54.GIB, 'observe')
        for flag, expected in [('--network', 'none'), ('--pull', 'never'), ('--cap-drop', 'ALL'),
                               ('--cpus', '1'), ('--memory-swap', str(10 * direct_edit54.GIB)),
                               ('--security-opt', 'no-new-privileges')]:
            self.assertEqual(command[command.index(flag) + 1], expected)
        self.assertIn('--read-only', command)
        self.assertNotIn('--gpus', command)
        self.assertNotIn('--privileged', command)
        self.assertNotIn('--rm', command)
        self.assertIn('CUDA_VISIBLE_DEVICES=', command)
        self.assertIn(direct_edit54.IMAGE, command)
        joined = ' '.join(command)
        self.assertNotIn('docker.sock', joined)
        self.assertNotIn('/home/op', joined)
        mounts = [command[i + 1] for i, item in enumerate(command) if item == '--mount']
        self.assertEqual(sum(not item.endswith(',readonly') for item in mounts), 1)
        self.assertTrue(any('target=/out' in item and not item.endswith(',readonly') for item in mounts))

    def test_unsafe_names_and_bind_separators_are_rejected(self):
        for name, deps in [('unrelated-job', '/deps'), ('wsp-direct-ok', '/data,readonly=false')]:
            with self.assertRaises(ValidationError):
                direct_edit54.container_command(name, Path('/code'), Path(deps), [], Path('/out'),
                                                Path('/request'), direct_edit54.GIB, 'verify')

    def test_decision_probe_adds_only_its_bound_readonly_patient_inputs(self):
        command = direct_edit54.container_command('wsp-direct-probe', Path('/frozen'), Path('/deps'),
                    [], Path('/out'), Path('/request'), 10 * direct_edit54.GIB, 'probe',
                    patient=Path('/patient-frozen'), source_request=Path('/request-original'))
        self.assertIn('type=bind,source=/patient-frozen,target=/patient.json,readonly', command)
        self.assertIn('type=bind,source=/request-original,target=/source-request.json,readonly', command)
        self.assertIn('scripts/direct_probe.py', command)
        self.assertNotIn('--gpus', command)
        with self.assertRaises(ValidationError):
            direct_edit54.container_command('wsp-direct-probe', Path('/frozen'), Path('/deps'),
                                            [], Path('/out'), Path('/request'), direct_edit54.GIB, 'probe')

    def test_bare_success_exit_or_wrong_identity_cannot_replace_a_worker_report(self):
        from direct_evidence_fixtures import make_worker
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with self.assertRaisesRegex(ValidationError, 'without its bound result'):
                direct_edit54.worker_evidence(output, 'verify', {}, 'f' * 64, direct_weights.load_lock()['_sha256'])
            value, report, path = make_worker(output, 'verify')
            check = lambda: direct_edit54.worker_evidence(output, 'verify', value, 'f' * 64, value['lock_sha256'])
            self.assertTrue(check()['worker_complete'])
            for field, changed in [('request_sha256', '0' * 64), ('training', True),
                                    ('quality_improvement_claimed', True), ('baseline_verification', None)]:
                path.write_text(json.dumps({**report, field: changed}))
                with self.subTest(field=field), self.assertRaises(ValidationError): check()

    def test_incomplete_probe_is_never_a_quality_pass(self):
        from direct_evidence_fixtures import make_worker
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            value, report, path = make_worker(output, 'probe', skipped=('length_control',))
            check = lambda: direct_edit54.worker_evidence(output, 'probe', value, 'f' * 64, value['lock_sha256'])
            result = check()
            self.assertFalse(result['worker_complete'])
            self.assertFalse(result['quality_improvement_claimed'])
            report['weights_edited'] = True; path.write_text(json.dumps(report))
            with self.assertRaises(ValidationError): check()


if __name__ == '__main__':
    unittest.main()
