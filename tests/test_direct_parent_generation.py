"""Authenticate the file generation actually consumed; synthetic checkpoints only."""
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'tests')]
import direct_weights as weights
import direct_forward as forward
import test_direct_weights as fixtures
from pipeline import ValidationError


class ParentGenerationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.DirectWeightContract(methodName='runTest')
        self.fixture.setUp(); self.addCleanup(self.fixture.doCleanups)

    def test_stable_authenticated_descriptor_consumes_exact_original_bytes(self):
        verified = weights.verify_baseline(self.fixture.lock, self.fixture.base)
        weights.assert_baseline_unchanged(verified)
        record = verified['files']['config.json']
        with weights.authenticated_file(record) as handle:
            self.assertEqual(weights.sha_bytes(handle.read()), record['sha256'])

    def test_same_size_inode_replacement_is_rejected_after_verification(self):
        verified = weights.verify_baseline(self.fixture.lock, self.fixture.base)
        record = verified['files']['config.json']; path = Path(record['path'])
        replacement = path.with_name('replacement.json'); replacement.write_bytes(path.read_bytes())
        os.replace(replacement, path)
        with self.assertRaisesRegex(ValidationError, 'changed before consumption'):
            weights.read_authenticated_json(record)

    def test_in_place_change_during_pinned_descriptor_use_is_rejected_on_exit(self):
        verified = weights.verify_baseline(self.fixture.lock, self.fixture.base)
        record = verified['files']['config.json']; path = Path(record['path'])
        with self.assertRaisesRegex(ValidationError, 'changed during consumption'):
            with weights.authenticated_file(record) as handle:
                original = handle.read()
                path.write_bytes(original + b' ')

    def test_changed_sidecar_is_rejected_before_any_model_library_load(self):
        verified = weights.verify_baseline(self.fixture.lock, self.fixture.base)
        path = Path(verified['files']['config.json']['path']); path.write_bytes(path.read_bytes() + b' ')
        with self.assertRaisesRegex(ValidationError, 'changed before consumption'):
            forward.load_text_model(self.fixture.lock, verified)

    def test_both_budgets_and_all_declared_eos_ids_have_correct_censor_status(self):
        model = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=[2, 4]))
        tokenizer = SimpleNamespace(eos_token_id=2)
        for maximum in (192, 384):
            for eos in (2, 4):
                with self.subTest(maximum=maximum, eos=eos):
                    result = forward.completion_status([3] * (maximum - 1) + [eos], maximum, model, tokenizer)
                    self.assertEqual(result['finish_reason'], 'eos')
                    self.assertTrue(result['reached_length_cap'])
                    self.assertFalse(result['censored']); self.assertFalse(result['hit_token_limit'])
            result = forward.completion_status([3] * maximum, maximum, model, tokenizer)
            self.assertEqual(result['finish_reason'], 'length'); self.assertTrue(result['censored'])
        self.assertTrue(forward.completion_status([3], 192, model, tokenizer)['censored'])


if __name__ == '__main__': unittest.main()
