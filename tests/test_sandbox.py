"""Integration checks for the actual pinned Docker grader; no model/GPU required."""
import os
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_coder import run_python

IMAGE = os.environ.get("WSP_TEST_DOCKER_IMAGE")


@unittest.skipUnless(IMAGE, "Set WSP_TEST_DOCKER_IMAGE to the installed pinned Python image")
class SandboxTests(unittest.TestCase):
    def test_correct_code_completes_the_tests(self):
        result = run_python("answer=42", "assert answer==42", IMAGE)
        self.assertTrue(result["passed"])
        self.assertTrue(result["tests_completed"])

    def test_incorrect_code_fails(self):
        self.assertFalse(run_python("answer=41", "assert answer==42", IMAGE)["passed"])

    def test_early_successful_exits_cannot_bypass_assertions(self):
        for code in ("raise SystemExit(0)", "import os; os._exit(0)"):
            with self.subTest(code=code):
                self.assertFalse(run_python(code, "assert False", IMAGE)["passed"])


if __name__ == "__main__":
    unittest.main()
