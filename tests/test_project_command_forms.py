"""Recognize the real unittest module-before-options form seen in OpenCode."""
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ops.project_clients import is_test_command


class ProjectCommandForms(unittest.TestCase):
    def test_module_before_verbose_executes_the_same_protected_suite(self):
        for module in ('test_scheduler','test_scheduler.py'):
            for command in (f'python3 -m unittest {module} -v',
                            f'cd /workspace && python3 -m unittest {module} -v 2>&1'):
                with self.subTest(command=command):self.assertTrue(is_test_command(command))
        for command in ('echo python3 -m unittest test_scheduler -v',
                        'python3 -m unittest another_module -v',
                        'python3 -m unittest test_scheduler -v && echo done'):
            with self.subTest(command=command):self.assertFalse(is_test_command(command))


if __name__=='__main__':unittest.main()
