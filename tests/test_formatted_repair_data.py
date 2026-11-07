"""Readable program targets and faithful failed-check reporting controls."""
import ast
import json
import os
import re
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from build_formatted_repair_corpus import format_python, samples, repair_records, observation_records
from build_agent_flow_corpus import controlled_results, IMAGE
from coder_common import normalize_record


def controls(rows):
    return {r['key']:{'before':{'exit_code':1,'stdout':'','stderr':'Ran 3 tests\nFAILED (failures=2, errors=1)'},
                      'after':{'exit_code':0,'stdout':'','stderr':'Ran 3 tests\nOK'}} for r in rows}


class FormattedRepairDataTests(unittest.TestCase):
    def test_split_checks_keep_mutation_postcondition_with_the_function_call(self):
        row=next(r for r in samples(1) if r['family']=='named_ports')
        name=ast.parse(row['fixed']).body[0].name
        checks=next(n for n in ast.parse(row['tests']).body if isinstance(n,ast.ClassDef))
        method=next(m for m in checks.body if any(isinstance(n,ast.Call)
            and isinstance(n.func,ast.Attribute) and n.func.attr=='assertEqual' and n.args
            and isinstance(n.args[0],ast.Name) and n.args[0].id=='data' for n in ast.walk(m)))
        self.assertTrue(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name)
            and n.func.id==name for n in ast.walk(method)),
            'Mutation assertion became vacuous when split away from the implementation call')

    def test_format_preserves_ast_and_rejects_invalid_compound_after_semicolon(self):
        source='def count(xs):\n    result = 0\n    for x in xs: result += x\n    if result: return result\n    return 0\n'
        formatted=format_python(source)
        self.assertEqual(ast.dump(ast.parse(formatted)),ast.dump(ast.parse(source)))
        self.assertIn('for x in xs:\n        result += x',formatted)
        with self.assertRaises(SyntaxError):
            format_python('def count(xs):\n    for x in xs: print(x); if x: break\n')
        for row in samples(1):
            for key in ('initial','fixed'):
                self.assertEqual(row[key],ast.unparse(ast.parse(row[key]))+'\n')

    def test_status_reports_execute_once_and_copy_actual_failure_and_error_counts(self):
        rows=samples(2); records=observation_records(rows,controls(rows))
        self.assertEqual(len(records),40)
        families={r['task_id'] for r in records}
        self.assertEqual(families,{'semantic-repair:'+r['family'] for r in rows})
        for raw in records:
            record=normalize_record(raw)
            calls=[c for m in record['messages'] for c in m.get('tool_calls',[])]
            self.assertEqual(len(calls),1)
            self.assertIn(calls[0]['function']['name'],('run_tests','check_suite'))
            observed=json.loads(record['messages'][-2]['content'])
            answer=json.loads(record['messages'][-1]['content'])
            self.assertEqual(answer,{k:observed[k] for k in answer})
            status=answer.get('exit_code',answer.get('returncode'))
            self.assertEqual(answer.get('failed',answer.get('fail_count')),3 if status else 0)

    @unittest.skipUnless(os.environ.get('WSP_TEST_DOCKER_IMAGE'),'Requires actual pinned Docker')
    def test_failure_edit_retest_trajectories_use_observed_broken_and_fixed_outcomes(self):
        self.assertEqual(os.environ['WSP_TEST_DOCKER_IMAGE'],IMAGE)
        rows=samples(1); actual=controlled_results(rows,IMAGE)
        counts={sum(map(int,re.findall(r'(?:failures|errors)=(\d+)',v['before']['stderr']))) for v in actual.values()}
        self.assertGreater(len(counts),1)
        for raw in repair_records(rows,actual):
            record=normalize_record(raw)
            if not record['tools']:continue
            calls=[c for m in record['messages'] for c in m.get('tool_calls',[])]
            self.assertEqual(len(calls),5)
            self.assertIn(calls[2]['function']['name'],('terminal','bash'))
            self.assertIn(calls[3]['function']['name'],('patch','edit'))
            self.assertIn(calls[4]['function']['name'],('terminal','bash'))
            self.assertNotEqual(json.loads(record['messages'][7]['content'])['exit_code'],0)
            self.assertEqual(json.loads(record['messages'][-2]['content'])['exit_code'],0)


if __name__=='__main__':unittest.main()
