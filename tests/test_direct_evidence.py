"""Adversarial host evidence and reservation tests. All worker data are synthetic."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'tests')]
import direct_evidence as evidence
import direct_weights as weights
from ops import direct_edit54 as host
from direct_evidence_fixtures import make_worker, write
from pipeline import ValidationError


class BoundEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(); self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name)
        self.request, self.report, self.path = make_worker(self.output, 'causal')

    def check(self):
        return host.worker_evidence(self.output, 'causal', self.request, 'f' * 64, self.request['lock_sha256'])

    def test_exact_official_parameter_set_matches_independently_archived_parent(self):
        original = json.loads((ROOT / 'docs/experiments/direct-weight-20261108/first-semantic-probe-02.result.json').read_text())
        self.assertEqual(evidence.expected_parameter_names(), set(original['loaded_parameter_sha256_before']))
        self.assertEqual(len(evidence.expected_parameter_names()), 426)

    def test_complete_execution_with_failed_scientific_gate_remains_no_quality_claim(self):
        self.assertEqual(self.report['diagnostic_gate']['status'], 'STOP_no_group')
        result = self.check()
        self.assertTrue(result['worker_complete'])
        self.assertFalse(result['quality_improvement_claimed'])
        self.assertEqual(result['independent_evaluation'], 'NOT_RUN')

    def test_complete_flags_cannot_hide_missing_duplicate_reordered_or_skipped_arms(self):
        arms = self.report['arms']
        for changed in ([], arms[:-1], arms + [arms[-1]], list(reversed(arms)),
                        [{'id': row['id'], 'status': 'NOT_RUN_budget'} for row in arms]):
            with self.subTest(ids=[row['id'] for row in changed]):
                write(self.path, {**self.report, 'arms': changed})
                with self.assertRaises(ValidationError): self.check()

    def test_count_only_maps_cannot_substitute_for_exact_426_names(self):
        for parameters in ({'bogus': '0' * 64}, {f'fake-{i}': '0' * 64 for i in range(426)},
                           {name: 'z' * 64 for name in evidence.expected_parameter_names()}):
            write(self.path, {**self.report, 'loaded_parameter_sha256_before': parameters,
                              'loaded_parameter_sha256_after': parameters})
            with self.assertRaises(ValidationError): self.check()

    def test_frozen_source_baseline_loaded_parent_and_tokenization_proofs_are_required(self):
        for key, value in (('producer_sha256', {}), ('evidence_schema_version', 1),
                           ('baseline_verification', None), ('loaded_parent_proof', None),
                           ('tokenization_proof', None)):
            write(self.path, {**self.report, key: value})
            with self.subTest(key=key), self.assertRaises(ValidationError): self.check()

    def test_late_raw_artifact_rewrite_or_hash_update_does_not_match_the_result(self):
        for field in ('score_artifact', 'causal_score_artifact'):
            row = self.report['arms'][0]
            path = self.path.parent / row[field]['path']; original = path.read_bytes()
            modified = json.loads(original); modified['next_token_id'] += 1
            write(path, modified)
            with self.subTest(field=field), self.assertRaises(ValidationError): self.check()
            changed = copy.deepcopy(self.report)
            changed['arms'][0][field] = evidence.artifact(path)
            write(self.path, changed)
            with self.subTest(field=field, rehashed=True), self.assertRaises(ValidationError): self.check()
            path.write_bytes(original); write(self.path, self.report)

    def test_cached_gate_is_recomputed_from_complete_raw_scores(self):
        changed = copy.deepcopy(self.report)
        changed['diagnostic_gate']['status'] = 'activation_group_qualified'
        changed['diagnostic_gate']['winner_group'] = 0
        write(self.path, changed)
        with self.assertRaisesRegex(ValidationError, 'verdict differs'): self.check()

    def test_missing_or_nonfinite_numeric_observation_is_refused_even_with_updated_hashes(self):
        import numpy as np
        row = self.report['arms'][0]
        path = self.path.parent / row['observation']
        with path.open('wb') as handle:
            np.savez(handle, layer_0_input_bf16=np.zeros(9216, dtype='<u2'))
        changed = copy.deepcopy(self.report); changed['arms'][0]['observation_sha256'] = evidence.artifact(path)['sha256']
        write(self.path, changed)
        with self.assertRaises(ValidationError): self.check()

    def test_observe_positive_and_token_completion_tampering(self):
        value, report, path = make_worker(self.output / 'observer', 'observe')
        check = lambda: host.worker_evidence(path.parent, 'observe', value, 'f' * 64, value['lock_sha256'])
        self.assertTrue(check()['worker_complete'])
        report['results'][0]['finish_reason'] = 'length'
        # Rebind the rewritten raw file as well; consistency with raw EOS still fails.
        row = report['results'][0]
        raw = {k: v for k, v in row.items() if k != 'generation_sha256'}
        row['generation_sha256'] = write(path.parent / (row['id'] + '-generation.json'), raw)['sha256']
        write(path, report)
        with self.assertRaisesRegex(ValidationError, 'completion fields'): check()


class LeaseRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(); self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name); self.lock = self.root / 'shared.lock'
        self.patch = mock.patch.object(host, 'DEFAULT_LEASE_PATH', self.lock); self.patch.start(); self.addCleanup(self.patch.stop)

    def run_dir(self, name):
        path = self.root / name; path.mkdir(); return path

    def inspect(self, *, running=False, returncode=0):
        state = {'Status': 'running' if running else 'exited', 'Running': running, 'Restarting': False, 'ExitCode': 0}
        return subprocess.CompletedProcess([], returncode, stdout=json.dumps([{'State': state}]), stderr='')

    def test_no_unlink_and_another_owner_is_refused_until_the_first_exits(self):
        with host.WorkerLease(self.run_dir('one')):
            inode = self.lock.stat().st_ino
            with self.assertRaisesRegex(ValidationError, 'another broker'):
                with host.WorkerLease(self.run_dir('two')): self.fail('second owner entered')
        with host.WorkerLease(self.run_dir('three')):
            self.assertEqual(self.lock.stat().st_ino, inode)
        self.assertEqual(json.loads(self.lock.read_text())['state'], 'released')

    def test_broker_exception_leaves_intent_and_live_unknown_worker_is_never_stopped(self):
        with self.assertRaises(RuntimeError):
            with host.WorkerLease(self.run_dir('one')) as lease:
                lease.launching('wsp-direct-lease-test', host.GIB)
                raise RuntimeError('simulated broker death')
        self.assertEqual(json.loads(self.lock.read_text())['state'], 'active')
        for response in (self.inspect(running=True), self.inspect(returncode=1)):
            with mock.patch.object(host.subprocess, 'run', return_value=response) as call:
                with self.assertRaises(ValidationError):
                    with host.WorkerLease(self.run_dir('recovery-' + str(response.returncode))): pass
                self.assertEqual(call.call_args.args[0], ['docker', 'inspect', 'wsp-direct-lease-test'])
        self.assertEqual(json.loads(self.lock.read_text())['state'], 'active')

    def test_confirmed_exit_recovers_crashed_intent_and_releases_after_own_worker(self):
        with host.WorkerLease(self.run_dir('one')) as lease:
            lease.launching('wsp-direct-lease-test', host.GIB)
        with mock.patch.object(host.subprocess, 'run', return_value=self.inspect()):
            with host.WorkerLease(self.run_dir('two')) as lease:
                lease.launching('wsp-direct-lease-next', host.GIB)
                lease.confirm_exit()
        self.assertEqual(json.loads(self.lock.read_text())['state'], 'released')


if __name__ == '__main__': unittest.main()
