"""Unified framework contracts, without official weights, tools or training."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'tests')]
import wsp_lab as lab
import wsp_interventions as iv
import wsp_pairs as pairs
import wsp_worker as worker
from ops import wsp_lab54 as host
from direct_weights import load_lock,bf16_float
from pipeline import ValidationError,sha256


def intervention(location='output'):
    return {'layer':0,'location':location,'position':-1,'off_target_position':0,
            'indices':[1,3],'operation':'zero','words':[]}


def protocol():
    return {'schema_version':1,'id':'engineering-fixture','role':'practice','method':'forward-only-fixed-interventions',
            'parent':lab.parent_identity(load_lock()),'generation':dict(lab.PROFILE),
            'case':{'id':'fixture','messages':[{'role':'user','content':'Reply with a greeting.'}],'tools':[],
                    'assessment':{'kind':'exact_text','expected':'synthetic fixture','family':'engineering','locality':True}},
            'intervention':intervention(),'budget':{'worker_seconds':1800,'total_worker_seconds':21600,'prior_worker_seconds':0},
            'design_sha256':'a'*64,'source_sha256':lab.source_manifest()}


class FixedWords(unittest.TestCase):
    def words(self,location='output'):
        a=np.zeros(iv.WIDTHS[location],dtype='<u2');a[[1,3]]=[0x3f80,0x4000];return a

    def test_zero_half_and_noop_are_private_exact_bf16_operations(self):
        before=self.words(); saved=before.copy(); spec=intervention()
        for arm in ('base','noop'):
            after,stats=iv.apply_words(before,spec,arm)
            self.assertTrue(np.array_equal(before,after));self.assertEqual(stats['delta_l2'],0)
        after,stats=iv.apply_words(before,spec,'treatment')
        self.assertEqual(after[[1,3]].tolist(),[0,0]);self.assertEqual(stats['changed_entries'],2)
        after,_=iv.apply_words(before,spec,'half')
        self.assertEqual(bf16_float(after[[1,3]]).tolist(),[.5,1])
        self.assertTrue(np.array_equal(before,saved))

    def test_replace_and_add_use_explicit_finite_words(self):
        spec={**intervention(),'operation':'replace','words':[0x4040,0x4080]}
        after,_=iv.apply_words(self.words(),spec,'treatment')
        self.assertEqual(bf16_float(after[[1,3]]).tolist(),[3,4])
        spec['operation']='add'
        after,_=iv.apply_words(self.words(),spec,'treatment')
        self.assertEqual(bf16_float(after[[1,3]]).tolist(),[4,6])
        spec={**intervention(),'operation':'replace','words':[0x8000,0]}
        after,_=iv.apply_words(np.zeros(2560,dtype='<u2'),spec,'treatment')
        self.assertEqual(after[[1,3]].tolist(),[0x8000,0])

    def test_hash_seeded_random_controls_are_deterministic_and_actual_norm_matched(self):
        baseline=self.words();spec=intervention();_,reference=iv.apply_words(baseline,spec,'treatment')
        for arm in ('random_a','random_b'):
            a,stats=iv.apply_words(baseline,spec,arm);b,_=iv.apply_words(baseline,spec,arm)
            self.assertTrue(np.array_equal(a,b));self.assertTrue(iv.norm_matched(stats,reference))
        self.assertFalse(iv.norm_matched({**reference,'delta_l2':reference['delta_l2']*1.051},reference))
        self.assertFalse(iv.norm_matched({'changed_entries':0,'delta_l2':0},{'changed_entries':0,'delta_l2':0}))

    def test_round_to_even_and_overflow_have_no_clipping_fallback(self):
        result=iv.bf16_words(np.array([1+1/256,1+3/256,-1-1/256],dtype='<f4'))
        self.assertEqual(bf16_float(result).tolist(),[1,1+1/64,-1])
        with self.assertRaises(ValidationError):iv.bf16_words(np.array([np.finfo(np.float32).max],dtype='<f4'))

    def test_invalid_indices_words_and_fitting_commands_fail_closed(self):
        for change in ({'indices':[True]},{'indices':[3,1]},{'indices':[1,1]},
                       {'operation':'fit'},{'words':[0x7f80]},{'location':'embedding'},{'layer':32}):
            with self.subTest(change=change),self.assertRaises(ValidationError):iv.validate({**intervention(),**change})


class ProtocolBoundary(unittest.TestCase):
    def test_valid_plan_is_eight_fixed_practice_arms_and_never_candidate_admission(self):
        value=protocol();lab.validate_protocol(value);plan=lab.plan(value)
        self.assertEqual(plan['arms'],list(iv.ARMS));self.assertEqual(plan['worst_case_worker_seconds'],14400)
        self.assertFalse(plan['weights_edited']);self.assertFalse(plan['weight_candidate_eligible'])

    def test_final_role_optimizers_changed_source_profile_or_parent_are_rejected(self):
        for field,value in (('role','final'),('method','lora'),('design_sha256','bad'),
                            ('schema_version',True),('generation',{**lab.PROFILE,'do_sample':True}),
                            ('parent',{**protocol()['parent'],'revision':'0'*40}),('optimizer','adam')):
            p=protocol();p[field]=value
            with self.subTest(field=field),self.assertRaises(ValidationError):lab.validate_protocol(p)
        p=protocol();p['source_sha256']['scripts/wsp_lab.py']='b'*64
        with self.assertRaises(ValidationError):lab.validate_protocol(p)

    def test_all_arm_costs_and_prior_work_must_fit(self):
        p=protocol();p['budget']['prior_worker_seconds']=7201
        with self.assertRaises(ValidationError):lab.validate_protocol(p)
        p=protocol();p['budget']['worker_seconds']=1801
        with self.assertRaises(ValidationError):lab.validate_protocol(p)

    def test_historical_thirteen_source_bank_remains_unchanged(self):
        import direct_tool_specificity as old
        self.assertEqual({n:sha256(ROOT/n) for n in old.STAGE1_SOURCES},old.STAGE1_SOURCES)
        old.validate_pins();self.assertEqual(len(lab.evidence.PRODUCER_FILES),13)

    def test_supplied_tool_association_and_arguments_are_data_only(self):
        from evaluate_coder import RUN
        c={'id':'fixture','tools':[RUN],'messages':[{'role':'user','content':'Run parser tests.'},
            {'role':'assistant','content':None,'tool_calls':[{'id':'actual_call','type':'function',
                'function':{'name':'run_tests','arguments':{'target':'parser'}}}]},
            {'role':'tool','tool_call_id':'actual_call','content':'{"exit_code":1}'}]}
        lab.validate_case(c)
        bad=copy.deepcopy(c);bad['messages'][-1]['tool_call_id']='invented'
        with self.assertRaises(ValidationError):lab.validate_case(bad)
        bad=copy.deepcopy(c);bad['messages'][1]['tool_calls'][0]['function']['arguments']='{"target":"parser"}'
        with self.assertRaises(ValidationError):lab.validate_case(bad)
        bad=copy.deepcopy(c);bad['tools'][0]['function']['parameters']['$ref']='https://example.com/schema'
        with self.assertRaises(ValidationError):lab.validate_case(bad)

    def test_cli_catalog_has_no_torch_peft_or_remote_execution_dependency(self):
        command=[sys.executable,'-c',"import sys;sys.path.insert(0,'scripts');import wsp_lab;wsp_lab.active_policy();assert 'torch' not in sys.modules;assert 'peft' not in sys.modules"]
        subprocess.run(command,cwd=ROOT,check=True,capture_output=True)
        p=subprocess.run([sys.executable,'scripts/wsp_lab.py','catalog'],cwd=ROOT,check=True,capture_output=True,text=True)
        self.assertEqual(len(json.loads(p.stdout)['integration']),6)

    def test_frozen_answers_are_not_rendered_into_model_inputs(self):
        p=protocol();p['case']['assessment']['expected']='ANSWER_NOT_IN_MODEL_INPUT'
        class Tokenizer:
            chat_template='synthetic engineering template'
            def apply_chat_template(self,messages,tokenize,**kwargs):
                return [1,2] if tokenize else messages[0]['content']
            def encode(self,text,**kwargs):return [1,2]
        ids,proof=lab.render_prompt(Tokenizer(),p)
        self.assertEqual(ids,[1,2]);self.assertNotIn('ANSWER_NOT_IN_MODEL_INPUT',proof['rendered'])

    def test_new_worker_reuses_readonly_cpu_sandbox_and_narrow_design_mount(self):
        argv=host.container_command('wsp-direct-lab-test',Path('/code'),Path('/deps'),[],Path('/out'),Path('/protocol'),Path('/design'),'a'*64,'base')
        self.assertIn('--read-only',argv);self.assertIn('--network',argv);self.assertNotIn('--gpus',argv)
        self.assertIn('type=bind,source=/design,target=/design.md,readonly',argv)
        self.assertIn('scripts/wsp_worker.py',argv);self.assertNotIn('scripts/direct_forward.py',argv)
        self.assertNotIn('--rm',argv)


class PairedMetrics(unittest.TestCase):
    def test_frozen_literal_grader_preserves_json_types_duplicates_and_bare_format(self):
        a={'kind':'json_exact','expected':{'exit_code':1,'failed':2},'family':'tool','locality':False}
        self.assertTrue(pairs.assess_text('{"failed":2,"exit_code":1}',a))
        for text in ('```json\n{"failed":2,"exit_code":1}\n```','{"failed":2,"exit_code":true}',
                     '{"failed":2,"exit_code":1,"failed":2}','{"failed":2,"exit_code":1,"extra":0}'):
            self.assertFalse(pairs.assess_text(text,a))
        with self.assertRaises(ValidationError):pairs.validate_assessment({**a,'kind':'python_exec'})
    def fixture(self):
        cases=[{'id':str(i),'family':'f'+str(i),'input_sha256':str(i+1)*64,'locality':i==2} for i in range(3)]
        m={'schema_version':1,'role':'practice','runtime_sha256':'a'*64,'grader_sha256':'b'*64,'cases':cases}
        rows=[]
        for c in cases:
            for arm in ('base','candidate'):
                passed=(c['id'] in ('0','2')) if arm=='base' else (c['id'] in ('1','2'))
                rows.append({'id':c['id'],'arm':arm,'input_sha256':c['input_sha256'],'runtime_sha256':'a'*64,
                             'grader_sha256':'b'*64,'passed':passed,'censored':False,'complete':True,
                             'request_sha256':'c'*64,'output_sha256':'d'*64})
        return m,rows

    def test_rescues_regressions_families_and_byte_locality_use_same_denominator(self):
        m,rows=self.fixture();r=pairs.paired_summary(m,rows)
        self.assertEqual((r['base_passes'],r['candidate_passes'],r['rescues'],r['regressions']),(2,2,1,1))
        self.assertEqual(r['family_net'],{'f0':-1,'f1':1,'f2':0});self.assertEqual(r['byte_identical_rate'],1)
        self.assertEqual(r['quality_evaluation'],'NOT_RUN');self.assertFalse(r['weight_candidate_eligible'])

    def test_missing_censored_duplicate_and_identity_mismatch_do_not_disappear(self):
        m,rows=self.fixture();r=pairs.paired_summary(m,rows[:-1])
        self.assertEqual(r['denominator'],3);self.assertFalse(r['complete']);self.assertEqual(r['missing_pairs'],['2'])
        for mutate in (lambda a:a.append(copy.deepcopy(a[0])),lambda a:a[0].update(runtime_sha256='f'*64),
                       lambda a:a[0].update(censored=True)):
            data=copy.deepcopy(rows);mutate(data)
            with self.assertRaises(ValidationError):pairs.paired_summary(m,data)
        m['role']='final'
        with self.assertRaises(ValidationError):pairs.paired_summary(m,rows)

    def test_kl_requires_full_normalized_vocabulary_not_top_k_or_nan(self):
        a=np.log([[.8,.2],[.5,.5]]);b=np.log([[.5,.5],[.5,.5]])
        self.assertEqual(pairs.token_kl(a,a),0);self.assertGreater(pairs.token_kl(a,b),0)
        for bad in (np.log([[.4,.1],[.4,.1]]),np.array([[float('nan'),0],[0,0]])):
            with self.assertRaises(ValidationError):pairs.token_kl(a,bad)


class ControllerFlow(unittest.TestCase):
    def simulate(self,fail=False,noop_mismatch=False):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);design=root/'design.md';design.write_text('Synthetic controller fixture; no model calls.\n')
            p=protocol();p['design_sha256']=sha256(design);path=root/'protocol.json';path.write_text(json.dumps(p))
            args=SimpleNamespace(protocol=path,design=design,output=root/'run',baseline=root/'absent',deps=root/'absent')
            clock=[0.];attempts=[];params={n:'a'*64 for n in lab.evidence.expected_parameter_names()}
            words=np.zeros(2560,dtype='<u2');words[[1,3]]=[0x3f80,0x4000]
            def fake(_args,proto,_hash,arm,area):
                area.mkdir();attempts.append(arm);clock[0]+=100.25
                if fail and arm=='treatment':raise RuntimeError('synthetic broker failure without receipt')
                _,stats=iv.apply_words(words,proto['intervention'],arm)
                (area/'execution.json').write_text(json.dumps({'worker_seconds':1}))
                return {'arm':arm,'complete':True,'parameters_unchanged':True,'parameter_postcheck_complete':True,
                        'loaded_parameter_sha256_before':params,'loaded_parameter_sha256_after':params,
                        'runtime':{'kind':'synthetic'},'intervention':stats,
                        'results':[{'input_ids':[1,2,3],'output_ids':[8,248046] if noop_mismatch and arm=='noop' else [9,248046],'censored':False,
                                    'text':'synthetic fixture','text_with_special_tokens':'synthetic fixture<|im_end|>',
                                    'observation':'unused.npz','observation_sha256':'a'*64}], 'protocol_sha256':_hash}
            with mock.patch.object(host.os,'uname',return_value=SimpleNamespace(nodename='4790')), \
                 mock.patch.object(host.host,'DEFAULT_LEASE_PATH',root/'lease'), \
                 mock.patch.object(host.time,'monotonic',side_effect=lambda:clock[0]), \
                 mock.patch.object(host,'run_arm',side_effect=fake), \
                 mock.patch.object(lab.evidence,'numeric_arrays',return_value={'fixture':words}),mock.patch('builtins.print'):
                if fail:
                    with self.assertRaisesRegex(RuntimeError,'synthetic broker'):host.run(args)
                else:host.run(args)
            return json.loads((args.output/'result.json').read_text()),attempts

    def test_complete_controller_has_fixed_order_and_charges_late_receipt_tail(self):
        result,attempts=self.simulate()
        self.assertEqual(attempts,list(iv.ARMS));self.assertEqual(result['status'],'fixed_activation_records_complete')
        self.assertEqual(result['worker_seconds'],816);self.assertFalse(result['weight_candidate_eligible'])
        self.assertEqual(result['quality_evaluation'],'NOT_RUN')
        self.assertEqual(result['development_metrics']['status'],'development_pair_graded')
        self.assertEqual(result['development_metrics']['candidate_kind'],'activation_treatment_only')

    def test_failed_attempt_without_receipt_is_charged_and_distinct_from_not_run(self):
        result,attempts=self.simulate(True)
        self.assertEqual(attempts,['base','noop','treatment']);self.assertEqual(result['worker_seconds'],306)
        self.assertEqual(result['attempted_unverified_arms'],['treatment'])
        self.assertEqual(result['not_run_arms'],list(iv.ARMS[3:]));self.assertEqual(result['status'],'STOP_infrastructure')

    def test_bad_noop_stops_before_launching_uninterpretable_treatment(self):
        result,attempts=self.simulate(noop_mismatch=True)
        self.assertEqual(attempts,['base','noop']);self.assertEqual(result['status'],'STOP_noop_replay')
        self.assertEqual(result['not_run_arms'],list(iv.ARMS[2:]));self.assertEqual(result['worker_seconds'],204)


if __name__=='__main__':unittest.main()
