"""Restoration must survive a failed phase or a missing transient marker."""
import importlib.util
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
            with patch.object(PHASE, "ROOT", root), patch.object(PHASE.os, "uname", return_value=SimpleNamespace(nodename="4790")), \
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
