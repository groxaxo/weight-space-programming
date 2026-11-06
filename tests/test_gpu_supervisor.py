"""Restoration must survive a failed phase or a missing transient marker."""
import importlib.util
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

PATH = Path(__file__).resolve().parents[1] / "ops/run54_gpu_phase.py"
SPEC = importlib.util.spec_from_file_location("gpu_phase_test", PATH)
PHASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PHASE)


class RestorationTests(unittest.TestCase):
    def test_recovery_needs_only_the_standard_library(self):
        with tempfile.TemporaryDirectory() as directory:
            code = f'''import sys, importlib.util, json
from pathlib import Path
spec=importlib.util.spec_from_file_location('recovery', {str(PATH)!r})
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
root=Path({directory!r});module.GPU_STATE=root/'host'
output=root/'out';output.mkdir()
owner={{'service':module.SERVICE,'was_active':False,'lease_id':'stdlib-owner','output':str(output)}}
module.GPU_STATE.mkdir()
(module.GPU_STATE/'owner.json').write_text(json.dumps(owner))
(output/'service-before-phase.json').write_text(json.dumps(owner))
assert module.restore(output)
assert (output/'bonsai-restored.json').exists()
assert not (module.GPU_STATE/'owner.json').exists()
'''
            subprocess.run([sys.executable, '-S', '-c', code], check=True, capture_output=True, timeout=15)

    @contextmanager
    def isolated(self, active=True):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = {'active': active, 'events': []}
            def systemctl(command, **kwargs):
                if 'is-active' in command:
                    return SimpleNamespace(returncode=0 if state['active'] else 3)
                if 'start' in command: state['active'] = True; state['events'].append('start')
                if 'stop' in command: state['active'] = False; state['events'].append('stop')
                return SimpleNamespace(returncode=0)
            with patch.object(PHASE, 'GPU_STATE', root / 'host'), \
                 patch.object(PHASE.os, 'uname', return_value=SimpleNamespace(nodename='4790')), \
                 patch.object(PHASE, 'wait_idle_gpu'), \
                 patch.object(PHASE.subprocess, 'run', side_effect=systemctl):
                yield root, state

    def test_readiness_wait_preserves_cpu_environment_and_waits_for_sampled_utilization(self):
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': ''}), \
             patch.object(PHASE.subprocess, 'check_output', side_effect=['0, 11890, 35\n', '0, 11890, 0\n']) as query, \
             patch.object(PHASE.time, 'sleep'):
            PHASE.wait_idle_gpu()
            self.assertEqual(query.call_count, 2)
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], '')

    def test_readiness_timeout_does_not_relax_memory_or_utilization_requirements(self):
        with patch.object(PHASE.subprocess, 'check_output', return_value='0, 100, 0\n'), \
             patch.object(PHASE.time, 'monotonic', side_effect=[0, 61]), patch.object(PHASE.time, 'sleep'):
            with self.assertRaisesRegex(ValueError, 'did not become idle'):
                PHASE.wait_idle_gpu()

    def test_failed_readiness_restores_the_service_and_releases_its_lease(self):
        with self.isolated() as (root, state):
            output = root / 'out'; output.mkdir()
            with patch.object(PHASE, 'wait_idle_gpu', side_effect=RuntimeError('readiness failed')):
                with self.assertRaises(RuntimeError):
                    with PHASE.gpu_window(output, True): self.fail('admitted occupied GPU')
            self.assertTrue(state['active'])
            self.assertFalse((PHASE.GPU_STATE / 'owner.json').exists())

    def test_initially_inactive_service_is_not_started_by_cleanup(self):
        with self.isolated(False) as (root, state):
            output = root / 'out'; output.mkdir()
            with PHASE.gpu_window(output, False): self.assertFalse(state['active'])
            PHASE.restore(output)
            self.assertEqual(state['events'], [])

    def test_checkouts_share_one_host_gpu_lock(self):
        with self.isolated() as (root, state):
            a, b = root / 'a', root / 'b'; a.mkdir(); b.mkdir()
            with patch.object(PHASE, 'ROOT', root / 'checkout-a'), PHASE.gpu_window(a, True):
                with patch.object(PHASE, 'ROOT', root / 'checkout-b'), self.assertRaises(BlockingIOError):
                    with PHASE.gpu_window(b, True): self.fail('second checkout acquired the GPU')
                self.assertFalse(state['active'])

    def test_abandoned_owner_is_recovered_before_the_next_admission(self):
        with self.isolated(False) as (root, state):
            a, b = root / 'a', root / 'b'; a.mkdir(); b.mkdir()
            PHASE.GPU_STATE.mkdir()
            owner = {'lease_id': 'dead-owner', 'output': str(a), 'was_active': True, 'service': PHASE.SERVICE}
            PHASE.write_json(a / 'service-before-phase.json', owner)
            PHASE.write_json(PHASE.GPU_STATE / 'owner.json', owner)
            with PHASE.gpu_window(b, True): self.assertFalse(state['active'])
            self.assertTrue(state['active'])
            self.assertTrue((a / 'bonsai-restored.json').exists())
            self.assertFalse((PHASE.GPU_STATE / 'owner.json').exists())

    def test_uncompleted_old_callback_defers_to_the_live_owner(self):
        with self.isolated() as (root, state):
            a, b = root / 'a', root / 'b'; a.mkdir(); b.mkdir()
            PHASE.write_json(a / 'service-before-phase.json', {'service': PHASE.SERVICE, 'was_active': True})
            with PHASE.gpu_window(b, True):
                self.assertFalse(PHASE.restore(a))
                self.assertFalse(state['active'])

    def exercise(self, remove_marker=False, fail=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "runs").mkdir()
            output = root / "output"
            output.mkdir()
            state = {"active": True}
            def systemctl(command, **_kwargs):
                if "is-active" in command:
                    return SimpleNamespace(returncode=0 if state["active"] else 3)
                if "stop" in command: state["active"] = False
                if "start" in command: state["active"] = True
                return SimpleNamespace(returncode=0)
            with patch.object(PHASE, "ROOT", root), patch.object(PHASE, "GPU_STATE", root / "host-state"), patch.object(PHASE, 'wait_idle_gpu'), patch.object(PHASE.os, "uname", return_value=SimpleNamespace(nodename="4790")), \
                 patch.object(PHASE.subprocess, "run", side_effect=systemctl):
                try:
                    with PHASE.gpu_window(output, True):
                        self.assertFalse(state["active"])
                        if remove_marker: (output / "restore-bonsai-required").unlink()
                        if fail: raise RuntimeError("failed model phase")
                except RuntimeError:
                    self.assertTrue(fail)
                self.assertTrue(state["active"], "Bonsai must return after the owned GPU window")
                self.assertTrue((output / "bonsai-restored.json").is_file())

    def test_initial_state_journal_survives_a_lost_transient_marker(self):
        self.exercise(remove_marker=True)

    def test_failed_phase_restores_the_preexisting_service(self):
        self.exercise(fail=True)


if __name__ == "__main__": unittest.main()
