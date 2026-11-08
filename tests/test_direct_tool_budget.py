"""The one technical replay cannot erase failed work or alter scientific gates."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from ops import direct_tool_cycle54 as cycle
import direct_tools as tools
from direct_weights import read_bound_json, read_json
from pipeline import ValidationError, sha256


class TechnicalReplayBudget(unittest.TestCase):
    def setUp(self):
        self.protocol = read_json(tools.PROTOCOL_FILE)
        self.path = ROOT / self.protocol['prior_budget_ledger']['path']
        self.prior, self.digest, _ = read_bound_json(self.path)

    def test_original_failed_costs_and_frozen_ledger_are_admitted_without_reset(self):
        cycle.validate_prior_budget(self.protocol, self.prior, self.digest)
        self.assertEqual(self.prior['generation_invocations_used'], 4)
        self.assertGreater(self.prior['total_worker_seconds'], 0)
        self.assertFalse(self.prior['reset'])
        self.assertEqual(self.digest, sha256(self.path))

    def test_reset_late_hash_fake_prior_count_or_changed_scientific_amendment_refused(self):
        for change in ({'reset': True}, {'generation_invocations_used': 0},
                       {'generation_invocations_used': 4.0}, {'all_prior_costs_reconciled': False},
                       {'total_worker_seconds': 0}, {'prior_verdict': 'passed'}):
            with self.subTest(change=change), self.assertRaises(ValidationError):
                cycle.validate_prior_budget(self.protocol, {**self.prior, **change}, self.digest)
        with self.assertRaises(ValidationError): cycle.validate_prior_budget(self.protocol, self.prior, '0' * 64)
        changed = copy.deepcopy(self.protocol)
        changed['owner_adopted_resource_amendment']['scientific_thresholds_cues_support_strengths_changed'] = True
        with self.assertRaises(ValidationError): cycle.validate_prior_budget(changed, self.prior, self.digest)

    def test_complete_tranche_must_fit_cumulative_seconds_even_when_proof_is_consistent(self):
        prior, protocol = copy.deepcopy(self.prior), copy.deepcopy(self.protocol)
        prior['total_worker_seconds'] = 20000
        protocol['prior_budget_ledger']['worker_seconds_charged'] = 20000
        with self.assertRaisesRegex(ValidationError, 'complete|Complete'):
            cycle.validate_prior_budget(protocol, prior, self.digest)

    def qualification(self):
        return {'kind': 'corrected-source-qualification', 'generation_invocations': 0,
                'protocol_sha256': sha256(tools.PROTOCOL_FILE), 'prior_ledger_sha256': self.digest,
                'source_sha256': tools.evidence.SOURCE_AT_IMPORT, 'preflight_sha256': 'a' * 64,
                'all_costs_reconciled': True, 'total_worker_seconds': 100.0}

    def test_new_preflight_runtime_and_unit_costs_are_added_before_any_model_call(self):
        self.assertEqual(cycle.validate_qualification(self.protocol, self.qualification(), 'a' * 64), 100.0)
        for change in ({'generation_invocations': 1}, {'generation_invocations': False},
                       {'all_costs_reconciled': False}, {'source_sha256': {}},
                       {'prior_ledger_sha256': '0' * 64}, {'total_worker_seconds': float('nan')}):
            with self.subTest(change=change), self.assertRaises(ValidationError):
                cycle.validate_qualification(self.protocol, {**self.qualification(), **change}, 'a' * 64)

    def test_qualification_cannot_bypass_actual_preflight_identity_or_complete_tranche_reserve(self):
        with self.assertRaises(ValidationError):
            cycle.validate_qualification(self.protocol, self.qualification(), 'b' * 64)
        with self.assertRaisesRegex(ValidationError, 'fit'):
            cycle.validate_qualification(self.protocol, {**self.qualification(), 'total_worker_seconds': 20000}, 'a' * 64)

    def test_one_claim_is_durable_never_unlinked_or_reused_after_worker_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); one, two = root / 'one', root / 'two'; one.mkdir(); two.mkdir()
            with mock.patch.object(cycle, 'REPLAY_CLAIM_ROOT', root):
                claim = cycle.claim_technical_replay(self.digest, one)
                inode = claim.stat().st_ino
                with self.assertRaisesRegex(ValidationError, 'already claimed'):
                    cycle.claim_technical_replay(self.digest, two)
                self.assertEqual(claim.stat().st_ino, inode)
                self.assertEqual(json.loads(claim.read_text())['replay_number'], 1)
                self.assertFalse(json.loads(claim.read_text())['cleanup'])


if __name__ == '__main__': unittest.main()
