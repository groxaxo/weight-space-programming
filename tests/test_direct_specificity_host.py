"""Admission and data-boundary regression tests; never launch a model."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from ops import direct_tool_specificity54 as h
from pipeline import ValidationError, sha256


class SpecificityHost(unittest.TestCase):
    def simulated_cycle(self, *, censored=False):
        sys.path.insert(0, str(ROOT / 'tests'))
        import test_direct_tool_specificity as fixtures
        records = fixtures.SpecificityContracts('runTest').records()
        row = {'censored': True} if censored else fixtures.SpecificityContracts('runTest').row(fixtures.CALL)
        records[1]['worker']['results'][0] = row
        records[1]['analysis'] = h.s.analyze(row, 'S')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('conditions', 'protocol', 'ledger'): (root / name).write_text('{}')
            args = SimpleNamespace(conditions=root / 'conditions', ledger=root / 'ledger', protocol=root / 'protocol',
                protocol_sha256=sha256(root / 'protocol'), output=root / 'run')
            clock, calls = [0.0], []
            def worker(_args, area, *_remaining):
                control = _remaining[-1]['control']; calls.append(control)
                area.mkdir()
                # The early worker receipt misses a deliberately slow resource/
                # release tail. The real controller must charge caller wall time.
                (area / 'execution.json').write_text(json.dumps({'complete': True, 'charged_worker_seconds': 1}))
                clock[0] += 100.25
                return copy.deepcopy(records[h.s.CONTROLS.index(control)])
            with mock.patch.object(h, 'validate_budget', return_value=4000), \
                    mock.patch.object(h.s, 'validate_request'), mock.patch.object(h.s, 'validate_conditions'), \
                    mock.patch.object(h.host, 'DEFAULT_LEASE_PATH', root / 'lease'), \
                    mock.patch.object(h.time, 'monotonic', side_effect=lambda: clock[0]), \
                    mock.patch.object(h, 'run_one', side_effect=worker), mock.patch('builtins.print'):
                h.run_cycle(args, {'manifest_sha256': 'a' * 64}, {})
            return json.loads((args.output / 'specificity-gate.json').read_text()), calls

    def test_censor_stop_retains_cause_and_only_unstarted_controls_are_not_run(self):
        result, calls = self.simulated_cycle(censored=True)
        self.assertEqual(calls, ['B', 'S'])
        self.assertEqual(result['status'], 'STOP_inconclusive')
        self.assertEqual(result['not_run_controls'], ['C', 'N', 'P'])
        self.assertEqual(result['attempted_unverified_controls'], [])
        self.assertEqual(result['cumulative_worker_seconds'], 4204)

    def test_completed_semantic_failure_finishes_five_and_charges_release_tail(self):
        result, calls = self.simulated_cycle()
        self.assertEqual(calls, list(h.s.CONTROLS))
        self.assertEqual(result['status'], 'STOP_semantic_specificity')
        self.assertEqual(result['not_run_controls'], [])
        self.assertEqual(result['cumulative_worker_seconds'], 4510)
        self.assertTrue(all(c['charged_worker_seconds'] == 102 for c in result['costs']))

    def test_attempted_unverified_controls_are_distinct_from_unstarted_controls(self):
        self.assertEqual(h.unfinished_controls(1, 2),
                         {'attempted_unverified_controls': ['S'], 'not_run_controls': ['C', 'N', 'P']})
        self.assertEqual(h.unfinished_controls(5, 5),
                         {'attempted_unverified_controls': [], 'not_run_controls': []})
        with self.assertRaises(ValidationError): h.unfinished_controls(3, 2)

    def ledger(self):
        return {'kind': 'specificity-source-qualification', 'all_costs_reconciled': True,
                'generation_invocations_used': 8, 'protocol_sha256': 'a' * 64,
                'conditions_sha256': 'b' * 64, 'source_sha256': h.s.evidence.SOURCE_AT_IMPORT,
                'specificity_sha256': h.s.SOURCE_SHA256, 'driver_sha256': sha256(ROOT / h.DRIVER),
                'total_worker_seconds': 4000.0}

    def test_entire_tranche_and_all_prior_costs_must_fit(self):
        ledger = self.ledger()
        self.assertEqual(h.validate_budget(ledger, 'a' * 64, 'b' * 64), 4000)
        for field, value in (('generation_invocations_used', 7), ('generation_invocations_used', True),
                             ('total_worker_seconds', 3000), ('total_worker_seconds', 12601),
                             ('total_worker_seconds', float('nan')), ('all_costs_reconciled', False),
                             ('driver_sha256', 'f' * 64), ('specificity_sha256', 'f' * 64)):
            bad = {**ledger, field: value}
            with self.subTest(field=field, value=value), self.assertRaises(ValidationError):
                h.validate_budget(bad, 'a' * 64, 'b' * 64)

    def test_new_entrypoint_preserves_container_isolation_and_readonly_inputs(self):
        argv = h.command('wsp-direct-specificity-test', Path('/code'), Path('/deps'),
                         [('/blob', '/model/shard')], Path('/out'), Path('/request'),
                         10 * h.host.GIB, Path('/stage1-data'), Path('/protocol'), 'a' * 64,
                         Path('/conditions'))
        for option in ('--read-only', '--cap-drop', '--security-opt', '--network', '--cpus'):
            self.assertIn(option, argv)
        self.assertEqual(argv[argv.index('--network') + 1], 'none')
        self.assertNotIn('scripts/direct_forward.py', argv)
        self.assertIn('scripts/direct_tool_specificity.py', argv)
        mounts = [argv[i + 1] for i, item in enumerate(argv) if item == '--mount']
        self.assertEqual(len([m for m in mounts if not m.endswith(',readonly')]), 1)
        self.assertTrue(any('target=/stage1,readonly' in m for m in mounts))
        self.assertFalse(any('docker.sock' in m or 'target=/home' in m for m in mounts))
        compile_argv = h.command('wsp-direct-specificity-compile', Path('/code'), Path('/deps'),
            [], Path('/out'), Path('/request'), 2 * h.host.GIB, Path('/stage1'), Path('/protocol'),
            'a' * 64, compile_only=True)
        self.assertIn('inside-compile', compile_argv)
        self.assertNotIn('scripts/direct_tool_specificity.py', compile_argv)

    def test_capture_refuses_symlinks_oversize_changed_and_existing_destinations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / 'source'; source.write_bytes(b'original')
            link = root / 'link'; link.symlink_to(source)
            for incoming, expected, size in ((link, sha256(source), 100),
                                             (source, 'f' * 64, 100), (source, sha256(source), 1)):
                with self.assertRaises(ValidationError): h.capture_file(incoming, root / 'out', expected, size)
            h.capture_file(source, root / 'out', sha256(source))
            with self.assertRaises(FileExistsError): h.capture_file(source, root / 'out', sha256(source))
            self.assertEqual(source.read_bytes(), b'original')

    def test_compiler_revalidates_stage1_before_importing_tokenizer(self):
        args = SimpleNamespace(stage1=Path('/missing'), protocol=Path('/protocol'), protocol_sha256='a' * 64)
        with mock.patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES': ''}), \
                mock.patch.object(h.s, 'load_stage1', side_effect=ValidationError('stage1 rejected')), \
                mock.patch.object(h, 'verify_baseline') as baseline:
            with self.assertRaisesRegex(ValidationError, 'stage1 rejected'): h.inside_compile(args)
            baseline.assert_not_called()


if __name__ == '__main__': unittest.main()
