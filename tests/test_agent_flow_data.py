"""Control authored trajectory correctness and provenance without model inference."""
import copy
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from build_agent_flow_corpus import samples,conversations,controlled_results
from coder_common import normalize_record
from prepare_coder_data import prepare,read_dataset


def fixture_controls(rows):
    # Synthetic unit-test feedback, distinct from the producer's Docker evidence.
    return {r['key']:{'control_cwd':'/isolated/'+r['key'],
        'before':{'exit_code':1,'stdout':'','stderr':'failed '+r['key']},
        'after':{'exit_code':0,'stdout':'','stderr':'passed '+r['key']}} for r in rows}


class AgentFlowDataTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('WSP_TEST_DOCKER_IMAGE') and shutil.which('docker'),
                         'Pinned Docker controls require the Linux test environment')
    def test_actual_unittest_controls_fail_before_and_pass_after_each_repair(self):
        rows=samples(1)
        observed=controlled_results(rows,os.environ['WSP_TEST_DOCKER_IMAGE'])
        self.assertEqual(set(observed),{r['key'] for r in rows})
        for row in rows:
            result=observed[row['key']]
            self.assertNotEqual(result['before']['exit_code'],0)
            self.assertEqual(result['after']['exit_code'],0)
            self.assertIn('Ran '+str(row['checks'])+' tests',result['after']['stderr'])

    def test_tool_results_remain_bound_to_the_corresponding_repair_and_actual_exit(self):
        rows=samples(1);controls=fixture_controls(rows)
        for raw in conversations(rows,controls):
            record=normalize_record(raw)
            calls=[];results={}
            for message in record['messages']:
                calls.extend(message.get('tool_calls',[]))
                if message['role']=='tool':results[message['tool_call_id']]=message['content']
            writes=[i for i,c in enumerate(calls) if c['function']['name'] in ('write','write_file')]
            self.assertEqual(len(writes),1)
            terminals=[(i,c) for i,c in enumerate(calls) if c['function']['name'] in ('bash','terminal')]
            for i,call in terminals:
                result=json.loads(results[call['id']])
                self.assertEqual(result['exit_code'],1 if i<writes[0] else 0)
            self.assertEqual(terminals[-1][0],len(calls)-1)
            self.assertEqual(record['messages'][-1]['role'],'assistant')
            self.assertFalse(record['messages'][-1].get('tool_calls'))

    def test_complete_file_writes_repair_docstrings_and_do_not_modify_test_files(self):
        rows=samples(2)
        for raw in conversations(rows,fixture_controls(rows)):
            record=normalize_record(raw)
            calls=[c['function'] for m in record['messages'] for c in m.get('tool_calls',[])]
            reads=[c['arguments'] for c in calls if c['name'] in ('read','read_file')]
            writes=[c['arguments'] for c in calls if c['name'] in ('write','write_file')]
            key='filePath' if 'filePath' in reads[0] else 'path'
            self.assertEqual(len(reads),2)
            self.assertEqual(writes[0][key],reads[0][key])
            self.assertNotEqual(writes[0][key],reads[1][key])
            self.assertEqual(Path(reads[1][key]).name,'test_unit.py')
            compile(writes[0]['content'],'authored-repaired.py','exec')

    @unittest.skipUnless(sys.platform.startswith('linux'),'Atomic dataset publication requires Linux')
    def test_semantic_families_keep_both_clients_variants_and_repairs_in_one_split(self):
        rows=samples(2);records=conversations(rows,fixture_controls(rows))
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'flows.jsonl'
            source.write_text(''.join(json.dumps(r)+'\n' for r in records))
            prepare([source],[{'name':'synthetic-unit-fixture','revision':'test','license':'owner-authored'}],
                    root/'prepared',seed=20261106)
            _,splits=read_dataset(root/'prepared')
            families=[{r['task_id'] for r in splits[name]} for name in ('train','validation')]
            self.assertFalse(families[0]&families[1])
            self.assertEqual(len(families[0]|families[1]),8)
            self.assertEqual(sum(len(s) for s in splits.values()),len(records))

    def test_feedback_path_rebinding_preserves_the_original_control_record(self):
        rows=samples(1);controls=fixture_controls(rows)
        for r in controls.values():
            for state in ('before','after'):
                r[state]['stderr']=r['control_cwd']+'\n'+r[state]['stderr']
        original=copy.deepcopy(controls)
        conversations(rows,controls)
        self.assertEqual(controls,original)


if __name__=='__main__':unittest.main()
