import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import usage_audit as u
import compare_gpu as c


def raw(i=100, cached=60, o=20, reasoning=10):
    return dict(input_tokens=i, cached_input_tokens=cached, output_tokens=o,
                reasoning_output_tokens=reasoning, cache_write_input_tokens=0, total_tokens=i+o)


class UsageAuditTests(unittest.TestCase):
    def test_subsets_are_not_added_to_total(self):
        t=u.normalize(raw())
        self.assertEqual(t['total_tokens'],120)
        self.assertEqual(t['noncached_input_plus_output_tokens'],60)
        self.assertEqual(t['non_reasoning_output_tokens'],10)

    def test_known_project_totals(self):
        t=u.add_usage([u.normalize(raw(1057241,991872,7168,4415)),u.normalize(raw(786523,708096,9054,5188)),u.normalize(raw(33691,12288,10358,7250))])
        self.assertEqual(t['total_tokens'],1904035)
        self.assertEqual(t['cached_input_tokens'],1712256)
        self.assertEqual(t['noncached_input_plus_output_tokens'],191779)
        self.assertEqual(t['reasoning_output_tokens'],16853)

    def test_unknown_breakdown_stays_unknown(self):
        t=u.normalize({'input_tokens':20,'output_tokens':5})
        self.assertIsNone(t['cached_input_tokens'])
        self.assertIsNone(t['noncached_input_plus_output_tokens'])
        self.assertIsNone(u.add_usage([t,u.normalize(raw())])['reasoning_output_tokens'])

    def test_negative_or_boolean_counts_rejected(self):
        for bad in (-1,True,1.5,'1'):
            with self.assertRaises(ValueError):u.normalize({'input_tokens':bad,'output_tokens':0})

    def test_subsets_cannot_exceed_parent(self):
        with self.assertRaises(ValueError):u.normalize(raw(cached=101))
        with self.assertRaises(ValueError):u.normalize(raw(reasoning=21))

    def test_inconsistent_total_rejected(self):
        v=raw();v['total_tokens']=999
        with self.assertRaises(ValueError):u.normalize(v)

    def test_no_usage_is_not_zero(self):
        with self.assertRaises(ValueError):u.add_usage([])

    def test_same_session_not_counted_twice(self):
        s={'session_id':'one','usage':raw()}
        with self.assertRaises(ValueError):u.agent_totals([s,s])

    def make_rollout(self,path,conflict=False,wrong_final=False):
        a,b=raw(),raw(50,20,10,4)
        cumulative=raw(150,80,30,14)
        if wrong_final:cumulative=raw(151,80,30,14)
        rows=[{'type':'session_meta','payload':{'id':'session','timestamp':'2026-11-06T00:00:00Z'}},
              {'type':'turn_context','payload':{'model':'gpt-6.1-sol','effort':'max'}},
              {'type':'response_item','payload':{'sensitive_text':'MUST_NOT_EXPORT'}},
              {'type':'token_usage_record','payload':{'response_id':'a','usage':a}},
              {'type':'event_msg','payload':{'type':'token_count','info':{'total_token_usage':a}}},
              {'type':'token_usage_record','payload':{'response_id':'a','usage':b if conflict else a}},
              {'type':'token_usage_record','payload':{'response_id':'b','usage':b}},
              {'type':'event_msg','timestamp':'2026-11-06T00:01:00Z','payload':{'type':'token_count','info':{'total_token_usage':cumulative}}}]
        path.write_text('\n'.join(json.dumps(r) for r in rows))

    def test_duplicate_receipts_and_cumulative_snapshots_not_double_counted(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'rollout.jsonl';self.make_rollout(p)
            r=u.rollout_usage(p)
            self.assertEqual(r['usage']['total_tokens'],180)
            self.assertEqual(r['unique_response_receipts'],2)
            self.assertNotIn('MUST_NOT_EXPORT',json.dumps(r))
            self.assertNotIn('response_id',json.dumps(r))

    def test_conflicting_duplicate_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'r';self.make_rollout(p,conflict=True)
            with self.assertRaises(ValueError):u.rollout_usage(p)

    def test_final_counter_must_match_receipts(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'r';self.make_rollout(p,wrong_final=True)
            with self.assertRaises(ValueError):u.rollout_usage(p)

    def test_planner_event_reconciles_with_provenance(self):
        with tempfile.TemporaryDirectory() as d:
            e,p=Path(d)/'events',Path(d)/'provenance'
            e.write_text(json.dumps({'type':'thread.started','thread_id':'plan'})+'\n'+json.dumps({'type':'turn.completed','usage':raw()}))
            p.write_text(json.dumps({'model':'gpt-6.1-sol','reasoning_effort':'max','usage':[raw()]}))
            self.assertEqual(u.plan_usage(e,p)['usage']['total_tokens'],120)
            p.write_text(json.dumps({'model':'gpt-6.1-sol','reasoning_effort':'max','usage':[raw(o=21)]}))
            with self.assertRaises(ValueError):u.plan_usage(e,p)

    def test_saved_samples_are_lower_bound_and_no_partial_accuracy(self):
        with tempfile.TemporaryDirectory() as d:
            m={'run_dir':d,'seed':0,'samples_per_problem':4,'sampling':{'max_tokens':32768}}
            row={'unique_id':'q0'}; arm='original-thinking'
            r={'manifest_sha256':'hash','arm':arm,'question_index':0,'sample_index':0,'question_id':'q0',
               'seed':c.sample_seed('q0',0,0),'prompt_tokens':5,'generated_tokens':10,'correct':True,'finish_reason':'stop','grading_error':None}
            c.save_record(Path(d)/arm/'records'/'0000-0.json.gz',r)
            result=u.saved_samples(m,'hash',arm,[row])
            self.assertEqual(result['generated_tokens_saved_samples'],10)
            self.assertEqual(result['total_tokens_saved_samples'],15)
            self.assertIsNone(result['full_arm_accuracy'])
            self.assertIn('in-flight',result['coverage'])

    def test_wrong_sample_manifest_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            m={'run_dir':d,'seed':0,'samples_per_problem':4,'sampling':{'max_tokens':32768}}
            c.save_record(Path(d)/'original-thinking'/'records'/'0.json.gz',{'manifest_sha256':'bad','arm':'original-thinking','question_index':0,'sample_index':0,'question_id':'q0'})
            with self.assertRaises(ValueError):u.saved_samples(m,'good','original-thinking',[{'unique_id':'q0'}])

    def test_unavailable_queue_not_misreported_as_queued(self):
        result=u.read_queue('abc','/nonexistent/gpuq')
        self.assertEqual(result['status'],'unavailable')


if __name__ == '__main__':unittest.main()
