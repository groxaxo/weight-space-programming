"""Semantic edge cases, grounded edits and family isolation for tuning data."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
from build_semantic_repair_corpus import samples, conversations
from build_agent_flow_corpus import controlled_results, IMAGE
from coder_common import normalize_record
from prepare_coder_data import prepare, read_dataset


class SemanticRepairDataTests(unittest.TestCase):
    def test_edits_are_grounded_and_success_is_only_reported_after_observation(self):
        rows = samples(2)
        controls = {r['key']:{'after':{'exit_code':0,'stdout':'','stderr':'Ran 1 test\nOK'}} for r in rows}
        records = [normalize_record(r) for r in conversations(rows,controls)]
        self.assertEqual(len(records),60)
        for record in records:
            if not record['tools']: continue
            calls = [c for m in record['messages'] for c in m.get('tool_calls',[])]
            self.assertEqual(len(calls),4)
            read, tests, edit, run = [c['function'] for c in calls]
            pk, old, new, sep = ('path','old_string','new_string','|') if edit['name']=='patch' else ('filePath','oldString','newString',': ')
            observed = record['messages'][3]['content']
            self.assertEqual('\n'.join(line.split(sep,1)[1] for line in observed.splitlines())+'\n',edit['arguments'][old])
            self.assertEqual(read['arguments'][pk],edit['arguments'][pk])
            self.assertTrue(tests['arguments'][pk].endswith('/test_unit.py'))
            self.assertEqual(run['arguments']['command'],'python3 -m unittest -v')
            self.assertEqual(json.loads(record['messages'][-2]['content'])['exit_code'],0)
            compile(edit['arguments'][new],'authored-repair.py','exec')
        controls[rows[0]['key']]['after']['exit_code']=1
        with self.assertRaisesRegex(ValueError,'Unsuccessful repair'):
            conversations(rows,controls)

    @unittest.skipUnless(sys.platform.startswith('linux'),'Atomic publication requires Linux')
    def test_code_variants_and_both_interfaces_keep_one_family_split(self):
        rows = samples(2)
        controls = {r['key']:{'after':{'exit_code':0,'stdout':'','stderr':'OK'}} for r in rows}
        records = conversations(rows,controls)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); path=root/'repairs.jsonl'
            path.write_text(''.join(json.dumps(r)+'\n' for r in records))
            prepare([path],[{'name':'test-repairs','revision':'fixture','license':'Apache-2.0'}],root/'prepared')
            _,splits=read_dataset(root/'prepared')
            self.assertFalse({r['task_id'] for r in splits['train']} & {r['task_id'] for r in splits['validation']})
            self.assertEqual(sum(map(len,splits.values())),60)
            for split in splits.values():
                for family in {r['task_id'] for r in split}:
                    self.assertEqual(sum(r['task_id']==family for r in split),6)

    @unittest.skipUnless(os.environ.get('WSP_TEST_DOCKER_IMAGE'),'Requires pinned Docker execution')
    def test_generator_duplicate_boolean_and_priority_controls_fail_before_and_pass_after(self):
        self.assertEqual(os.environ['WSP_TEST_DOCKER_IMAGE'],IMAGE)
        rows=samples(1); controls=controlled_results(rows,IMAGE)
        self.assertEqual(set(controls),{r['key'] for r in rows})
        for observed in controls.values():
            self.assertNotEqual(observed['before']['exit_code'],0)
            self.assertEqual(observed['after']['exit_code'],0)


if __name__=='__main__': unittest.main()
