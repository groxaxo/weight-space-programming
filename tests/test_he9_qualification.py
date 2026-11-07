"""Synthetic acceptance evidence: no model scores or live clients in these tests."""
import copy
import hashlib
from http.server import HTTPServer
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.request import urlopen

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from coder_common import CoderError,adapter_identity,digest,fingerprint,load_config
from evaluate_coder import CODE_CASES,TOOL_CASES
import qualify_he9 as gate
from qualify_he9 import qualify
from ops.replay_he9 import LIMITS
from serve_coder import handler_for
import test_he9 as he9_fixtures


class QualificationTests(unittest.TestCase):
    def fixture(self):
        base=copy.deepcopy(he9_fixtures.He9Tests().fixture(3))
        candidate=copy.deepcopy(he9_fixtures.He9Tests().fixture(4,True))
        training={'status':'trained_unpromoted','smoke_only':False,'before':{'eval_loss':.3},
            'after':{'eval_loss':.2},'adapter_files':{'adapter_model.safetensors':'a'*64,'adapter_config.json':'a'*64}}
        cfg=load_config(ROOT/'configs/qwen35-4b.json')
        for report in (base,candidate):
            identity=report['results'][0]['identity']
            identity.update(identity_schema=2,base_model=cfg['model_id'],revision=cfg['revision'])
            identity['runtime']=copy.deepcopy(identity['runtime'])
            identity['runtime']['launch_options']=dict(LIMITS)
            identity['runtime']['backend']='llama.cpp'
            if identity['adapter']:identity.update(adapter_identity(training))
            report['protocol_sha256']=gate.source_hashes(None,gate.HE9_SOURCES)
            for row in report['results']:
                row['identity']=identity
                row['harness_sha256']={'ops/client_proxy.py':digest(ROOT/'ops/client_proxy.py')}
                row['model_attestation'].update(identity_sha256=fingerprint(identity),proxy_sha256=digest(ROOT/'ops/client_proxy.py'))
        phases=[]
        image='python:3.12-slim@sha256:'+'b'*64
        for report,instance in ((base,'synthetic-base-instance'),(candidate,'synthetic-candidate-instance')):
            identity=report['results'][0]['identity']
            rows=[]
            for name,_,_ in CODE_CASES:
                rows.append({'id':name,'category':'code','passed':True,'tests_completed':True,
                    'container_exit_code':73,'exit_code':0,'timed_out':False})
            rows.extend({'id':case['id'],'category':'tools','passed':True,
                'trace':[{'message':{'role':'assistant','content':'Synthetic trace only.'}}]} for case in TOOL_CASES)
            development={'schema_version':1,'identity':identity,'suite_sha256':fingerprint([CODE_CASES,TOOL_CASES]),
                'evaluator_sha256':digest(ROOT/'scripts/evaluate_coder.py'),'sandbox_image':image,'results':rows,
                'scores':{'code':{'passed':8,'total':8},'tools':{'passed':6,'total':6}}}
            phase={'completed':{'status':'complete'},'ready':{'object':'list',
                'data':[{'id':cfg['served_model_name'],'object':'model','owned_by':'local'}],
                'wsp':identity,'wsp_instance':instance},
                'development':development}
            for kind,harness in (('file','smoke_clients.py'),('project','project_clients.py')):
                results=[]
                for name in ('opencode','hermes'):
                    row={'client':name,'version':'synthetic-'+name,'passed':True,'exit_code':0,'timed_out':False,
                        'tests_sha256':'c'*64,'source_after_sha256':'d'*64,
                        'source_sha256':{'scheduler/validation.py':'d'*64,'scheduler/planning.py':'e'*64},
                        'model_attestation':{'verified':True,'requests':2,'instance_id':instance,
                            'identity_sha256':fingerprint(identity),'proxy_sha256':digest(ROOT/'ops/client_proxy.py')},
                        'independent_tests':{'passed':True,'tests_completed':True,'container_exit_code':73},
                        'test_command_observed':True,'remaining_test_containers':[],
                        'container_isolation':[{'verified':True,'network':'none','privileged':False,
                            'cap_drop':['ALL'],'pids_limit':64,'memory_bytes':768*1024**2,'user':'1000:1000'}]}
                    results.append(row)
                phase[kind]={'schema_version':1,'identity':identity,'sandbox_image':image,'results':results,
                    'passed':True,'harness_sha256':digest(ROOT/'ops'/harness)}
            phases.append(phase)
        return [base,candidate,*phases,training]

    def test_complete_full_context_pair_qualifies_without_promoting_an_endpoint(self):
        result=qualify(*self.fixture())
        self.assertEqual(result['status'],'qualified_for_native_he9_use')
        self.assertEqual(result['identity']['limits']['context'],262144)
        self.assertFalse(result['adapter_promoted'])

    def test_real_frontend_health_discovery_is_accepted_with_matching_client_instances(self):
        data=self.fixture();cfg=load_config(ROOT/'configs/qwen35-4b.json')
        for report,phase in ((data[0],data[2]),(data[1],data[3])):
            backend=SimpleNamespace(cfg=cfg,identity=report['results'][0]['identity'])
            server=HTTPServer(('127.0.0.1',0),handler_for(backend))
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            try:
                with urlopen(f'http://127.0.0.1:{server.server_port}/health',timeout=5) as response:
                    phase['ready']=json.load(response)
            finally:
                server.shutdown();server.server_close();thread.join(timeout=5)
            self.assertNotIn('status',phase['ready'])
            for kind in ('file','project'):
                for row in phase[kind]['results']:
                    row['model_attestation']['instance_id']=phase['ready']['wsp_instance']
        self.assertEqual(qualify(*data)['status'],'qualified_for_native_he9_use')

    def test_discovery_cannot_omit_or_substitute_the_model_or_instance(self):
        for mode in ('status-only','wrong-model','duplicate-model','empty-instance'):
            data=self.fixture();ready=data[3]['ready']
            if mode=='status-only':
                ready.pop('object');ready.pop('data');ready['status']='ok'
            elif mode=='wrong-model':ready['data'][0]['id']='unrelated-model'
            elif mode=='duplicate-model':ready['data'].append(copy.deepcopy(ready['data'][0]))
            else:ready['wsp_instance']=''
            with self.subTest(mode=mode),self.assertRaises(CoderError):qualify(*data)

    def test_he9_gain_cannot_replace_a_failed_development_or_real_client_case(self):
        for mode in ('development','file','project'):
            data=self.fixture();phase=data[3]
            if mode=='development':
                phase['development']['results'][0]['passed']=False
                phase['development']['scores']['code']['passed']=7
            else:
                phase[mode]['results'][0]['passed']=False
                phase[mode]['passed']=False
            with self.subTest(mode=mode):self.assertEqual(qualify(*data)['status'],'not_promoted')

    def test_a_he9_tie_still_fails_even_with_perfect_client_and_development_reports(self):
        data=self.fixture()
        data[1]['results'][3].update(base_pass=False,plus_pass=False,plus_passed=0,error='synthetic unresolved case')
        self.assertEqual(qualify(*data)['status'],'not_promoted')

    def test_partial_duplicate_and_misclassified_development_evidence_is_rejected(self):
        for mode in ('partial','duplicate','category'):
            data=self.fixture();rows=data[3]['development']['results']
            if mode=='partial':rows.pop()
            elif mode=='duplicate':rows[1]['id']=rows[0]['id']
            else:rows[0]['category']='tools'
            with self.subTest(mode=mode),self.assertRaises(CoderError):qualify(*data)

    def test_model_or_instance_drift_cannot_satisfy_the_client_gate(self):
        for mode in ('instance','identity','harness'):
            data=self.fixture();report=data[3]['file']
            if mode=='instance':report['results'][0]['model_attestation']['instance_id']='another-server'
            elif mode=='identity':report['identity']=copy.deepcopy(report['identity']);report['identity']['adapter_sha256']='f'*64
            else:report['harness_sha256']='f'*64
            with self.subTest(mode=mode),self.assertRaises(CoderError):qualify(*data)

    def test_success_cannot_skip_tests_or_project_isolation(self):
        for mode in ('assertions','command','isolation','source'):
            data=self.fixture();row=data[3]['project']['results'][0]
            if mode=='assertions':row['independent_tests']['tests_completed']=False
            elif mode=='command':row['test_command_observed']=False
            elif mode=='isolation':row['container_isolation']=[]
            else:row.pop('source_sha256')
            with self.subTest(mode=mode),self.assertRaises(CoderError):qualify(*data)

    def test_both_identities_must_still_declare_the_native_q8_runtime(self):
        for mode in ('backend','kv'):
            data=self.fixture()
            for report,phase in ((data[0],data[2]),(data[1],data[3])):
                identity=report['results'][0]['identity']
                if mode=='backend':identity['runtime']['backend']='synthetic-other-engine'
                else:identity['runtime']['launch_options']['cache_type_v']='f16'
                for row in report['results']:row['model_attestation']['identity_sha256']=fingerprint(identity)
                for kind in ('file','project'):
                    for row in phase[kind]['results']:row['model_attestation']['identity_sha256']=fingerprint(identity)
            with self.subTest(mode=mode),self.assertRaises(CoderError):qualify(*data)

    def test_pinned_measurement_sources_use_ancestor_blobs_and_reject_unknown_revisions(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'ops').mkdir()
            path=root/'ops/project_clients.py';path.write_text('old frozen detector\n')
            def git(*args):
                return subprocess.check_output(['git','-C',str(root),'-c','core.hooksPath=/dev/null',
                    '-c','commit.gpgsign=false',*args],stderr=subprocess.DEVNULL,text=True).strip()
            git('init','-q');git('config','user.name','Synthetic test');git('config','user.email','test@example.invalid')
            git('add','.');git('commit','-qm','frozen');old=git('rev-parse','HEAD')
            path.write_text('later corrected detector\n');git('add','.');git('commit','-qm','correction')
            with patch.object(gate,'ROOT',root):
                frozen=gate.source_hashes(old,('ops/project_clients.py',))
                self.assertEqual(frozen['ops/project_clients.py'],hashlib.sha256(b'old frozen detector\n').hexdigest())
                self.assertNotEqual(frozen,gate.source_hashes(None,('ops/project_clients.py',)))
                for revision in ('HEAD','0'*40):
                    with self.subTest(revision=revision),self.assertRaises(CoderError):
                        gate.source_hashes(revision,('ops/project_clients.py',))


if __name__=='__main__':unittest.main()
