"""Practice grading is separate from model execution and final acceptance."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from ops import grade_direct_practice as grade
from evaluate_coder import CODE_CASES
from pipeline import ValidationError


def fixture(text='def wsp_intersection(a,b): return None'):
    prompt = next(c[1] for c in CODE_CASES if c[0] == 'half_open_overlap')
    generation = {'id': 'half_open_overlap-r1', 'text': text, 'hit_token_limit': False}
    request = {'role': 'practice', 'cases': [{'id': generation['id'],
                                           'messages': [{'role': 'user', 'content': prompt}]}]}
    return generation, request


class PracticeGradingTests(unittest.TestCase):
    def test_completed_exposed_prompt_uses_original_assertions(self):
        generation, request = fixture()
        code, tests = grade.prepare(generation, request)
        self.assertEqual(code, generation['text'])
        self.assertEqual(tests, next(c[2] for c in CODE_CASES if c[0] == 'half_open_overlap'))

    def test_final_material_censored_outputs_and_prompt_changes_are_refused(self):
        pairs = []
        generation, request = fixture(); request['role'] = 'independent-final'; pairs.append((generation, request))
        generation, request = fixture(); generation['hit_token_limit'] = True; pairs.append((generation, request))
        generation, request = fixture(); request['cases'][0]['messages'][0]['content'] += ' changed'; pairs.append((generation, request))
        generation, request = fixture(); generation['id'] = 'other'; pairs.append((generation, request))
        for generation, request in pairs:
            with self.subTest(generation=generation, request=request), self.assertRaises(ValidationError):
                grade.prepare(generation, request)

    @unittest.skipUnless(os.environ.get('WSP_TEST_DOCKER_IMAGE'), 'Requires installed pinned Docker image')
    def test_retained_sandbox_distinguishes_correct_code_and_early_success_exit(self):
        correct = ('def wsp_intersection(a,b):\n'
                   ' if a[0]>a[1] or b[0]>b[1]: raise ValueError("reversed")\n'
                   ' start=max(a[0],b[0]); end=min(a[1],b[1])\n'
                   ' return (start,end) if start<end else None\n')
        for text, expected in ((correct, True), ('raise SystemExit(0)', False)):
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                generation, request = fixture(text)
                (root / 'generation.json').write_text(json.dumps(generation))
                (root / 'request.json').write_text(json.dumps(request))
                output = root / 'control'
                result = subprocess.run([sys.executable, str(ROOT / 'ops/grade_direct_practice.py'),
                    '--generation', str(root / 'generation.json'), '--request', str(root / 'request.json'),
                    '--output', str(output)], capture_output=True, text=True, timeout=45)
                self.assertEqual(result.returncode, 0, result.stderr)
                report = json.loads((output / 'result.json').read_text())
                self.assertEqual(report['passed'], expected)
                self.assertEqual(report['tests_completed'], expected)
                self.assertTrue((output / 'container-inspect.json').is_file())
                self.assertNotIn('--rm', report['command'])


if __name__ == '__main__':
    unittest.main()
