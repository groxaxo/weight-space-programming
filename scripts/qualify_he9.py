#!/usr/bin/env python3
"""Combine exact HE9 improvement with complete paired development/client proof.

This selected-subset qualification never asserts that the separate all-164 gate
passed, changes a client profile, starts a server or promotes an adapter.
"""
import argparse
import json
import math
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from coder_common import adapter_identity,check,digest,fingerprint,load_config,loads,write_json
from compare_he9 import compare_he9
from evaluate_coder import CODE_CASES,TOOL_CASES
from ops.replay_he9 import LIMITS


def valid_hash(value):
    return isinstance(value,str) and re.fullmatch('[a-f0-9]{64}',value) is not None


def development(report,identity):
    check(report['schema_version']==1 and report['identity']==identity,'Development model identity differs')
    check(report['suite_sha256']==fingerprint([CODE_CASES,TOOL_CASES]) and
          report['evaluator_sha256']==digest(ROOT/'scripts/evaluate_coder.py'),'Different development suite or source')
    expected={row[0]:'code' for row in CODE_CASES}|{row['id']:'tools' for row in TOOL_CASES}
    rows=report['results']
    check(len(rows)==14 and len({r['id'] for r in rows})==14 and {r['id'] for r in rows}==set(expected),
          'All fourteen distinct development results are required')
    for row in rows:
        check(row['category']==expected[row['id']] and type(row['passed']) is bool,'Invalid development category or verdict')
        if row['passed'] and row['category']=='code':
            check(row.get('tests_completed') is True and row.get('container_exit_code')==73
                and row.get('exit_code')==0 and row.get('timed_out') is False,'Passing code skipped executable assertions')
        if row['passed'] and row['category']=='tools':
            check(isinstance(row.get('trace'),list) and bool(row['trace']),'Passing tool case has no model trace')
    scores={category:{'passed':sum(r['passed'] for r in rows if r['category']==category),
        'total':sum(r['category']==category for r in rows)} for category in ('code','tools')}
    check(report['scores']==scores,'Inconsistent development totals')
    return {r['id']:r for r in rows}


def clients(report,identity,instance,kind):
    check(report['schema_version']==1 and report['identity']==identity,'Client model identity differs')
    harness='smoke_clients.py' if kind=='file' else 'project_clients.py'
    check(report['harness_sha256']==digest(ROOT/'ops'/harness),'Different actual-client harness source')
    rows=report['results']
    check(len(rows)==2 and {r['client'] for r in rows}=={'opencode','hermes'} and
          all(type(r['passed']) is bool for r in rows),'Both actual clients are required')
    check(type(report['passed']) is bool and report['passed']==all(r['passed'] for r in rows),'Inconsistent client totals')
    passed=True
    for row in rows:
        check(isinstance(row.get('version'),str) and row['version'],'Missing installed client version')
        proof=row.get('model_attestation',{})
        check(proof.get('verified') is True and type(proof.get('requests')) is int and proof['requests']>0
            and proof.get('instance_id')==instance and proof.get('identity_sha256')==fingerprint(identity)
            and proof.get('proxy_sha256')==digest(ROOT/'ops/client_proxy.py'),'Missing matching actual-client attribution')
        tests=row.get('independent_tests',{})
        complete=(row['passed'] and row.get('exit_code')==0 and row.get('timed_out') is False
            and tests.get('passed') is True and tests.get('tests_completed') is True
            and tests.get('container_exit_code')==73 and not row.get('error'))
        if kind=='file':
            complete=complete and valid_hash(row.get('source_after_sha256'))
        if kind=='project':
            source=row.get('source_sha256',{})
            complete=complete and (isinstance(source,dict) and
                set(source)=={'scheduler/validation.py','scheduler/planning.py'} and
                all(valid_hash(value) for value in source.values()))
            observations=row.get('container_isolation',[])
            complete=complete and row.get('test_command_observed') is True and bool(observations)
            for observed in observations:
                complete=complete and (observed.get('verified') is True and observed.get('network')=='none'
                    and observed.get('privileged') is False and 'ALL' in (observed.get('cap_drop') or [])
                    and type(observed.get('pids_limit')) is int and 0<observed['pids_limit']<=64
                    and type(observed.get('memory_bytes')) is int and 0<observed['memory_bytes']<=768*1024**2
                    and isinstance(observed.get('user'),str) and observed['user'].split(':')[0]=='1000')
            complete=complete and row.get('remaining_test_containers')==[] and not row.get('cleanup_error')
        if row['passed']:
            check(complete,'Client claims success without complete executable/isolation evidence')
            check(valid_hash(row.get('tests_sha256')),'Missing immutable client tests identity')
        passed=passed and complete
    return passed,{r['client']:r for r in rows}


def qualify(base_he9,candidate_he9,base_phase,candidate_phase,training):
    he9=compare_he9(base_he9,candidate_he9)
    identities=[r['results'][0]['identity'] for r in (base_he9,candidate_he9)]
    cfg=load_config(ROOT/'configs/qwen35-4b.json')
    for identity in identities:
        check(identity.get('identity_schema')==2 and identity['base_model']==cfg['model_id'] and
            identity['revision']==cfg['revision'],'Missing pinned schema-2 Qwen identity')
        check(identity['limits']=={'context':LIMITS['context'],'output':LIMITS['output']},'Different HE9 serving limits')
        check(identity['quantization']=='Q8_0' and identity['runtime'].get('backend')=='llama.cpp',
              'The native Q8 runtime is required')
        launch=identity['runtime']['launch_options']
        check(all(launch.get(key)==LIMITS[key] for key in ('context','output','thinking','cache_type_k','cache_type_v')),
              'Native context, mode or Q8 KV settings differ from HE9')
    evaluated=[];client_rows=[];client_pass=[]
    images=[]
    for phase,identity in zip((base_phase,candidate_phase),identities):
        check(phase['completed'].get('status')=='complete','Compatibility phase is incomplete')
        ready=phase['ready']
        check(ready.get('status')=='ok' and ready.get('wsp')==identity and
              isinstance(ready.get('wsp_instance'),str) and ready['wsp_instance'],'Different compatibility serving instance')
        evaluated.append(development(phase['development'],identity))
        pairs={}
        for kind in ('file','project'):
            report=phase[kind]
            check(report['sandbox_image']==phase['development']['sandbox_image'],'Client and development sandboxes differ')
            passed,rows=clients(report,identity,ready['wsp_instance'],kind)
            pairs[kind]=rows
            if identity.get('adapter'):client_pass.append(passed)
        client_rows.append(pairs);images.append(phase['development']['sandbox_image'])
    check(images[0]==images[1] and '@sha256:' in images[0],'Paired development sandboxes differ')
    for kind in ('file','project'):
        for name in ('opencode','hermes'):
            before,after=(r[kind][name] for r in client_rows)
            check(before['version']==after['version'],'Installed client version changed between arms')
            if 'tests_sha256' in before and 'tests_sha256' in after:
                check(before['tests_sha256']==after['tests_sha256'],'Client fixture tests changed between arms')
    check(training['status']=='trained_unpromoted' and training['smoke_only'] is False,'A complete trained adapter is required')
    for key,value in adapter_identity(training).items():
        check(identities[1].get(key)==value,'Training belongs to a different measured adapter: '+key)
    before,after=training['before']['eval_loss'],training['after']['eval_loss']
    check(all(type(v) in (int,float) and math.isfinite(v) and v>=0 for v in (before,after)),'Invalid held-out loss')
    first,second=evaluated
    losses=[key for key in first if first[key]['passed'] and not second[key]['passed']]
    gains=[key for key in first if not first[key]['passed'] and second[key]['passed']]
    all_fourteen=all(r['passed'] for r in second.values())
    accepted=(he9['status']=='he9_qualified_pending_compatibility' and all_fourteen and
              all(client_pass) and after<before)
    return {'schema_version':1,'status':'qualified_for_native_he9_use' if accepted else 'not_promoted',
        'adapter_promoted':False,'he9':he9,'development':{'all_fourteen_passed':all_fourteen,
            'baseline_passed':sum(r['passed'] for r in first.values()),
            'candidate_passed':sum(r['passed'] for r in second.values()),'regressions':losses,'improvements':gains},
        'candidate_file_clients_passed':client_pass[0],'candidate_project_clients_passed':client_pass[1],
        'validation_loss_improved':after<before,'identity':identities[1],
        'scope':'Exact failure-selected HE9 plus fourteen authored development cases and paired actual file/project clients. '
            'No all-164 HumanEval qualification, general coding guarantee, client-profile change or automatic endpoint promotion.'}


def phase_inputs(directory,paths):
    names={'completed':'phase-completed.json','ready':'server-ready.json','development':'evaluation.json',
        'file':'clients/acceptance.json','project':'project-clients/acceptance.json'}
    result={}
    for key,name in names.items():
        path=directory/name;paths.append(path);result[key]=loads(path.read_text())
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('baseline-he9','candidate-he9','baseline-phase','candidate-phase','training','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();paths=[args.baseline_he9,args.candidate_he9,args.training]
    result=qualify(loads(args.baseline_he9.read_text()),loads(args.candidate_he9.read_text()),
        phase_inputs(args.baseline_phase,paths),phase_inputs(args.candidate_phase,paths),loads(args.training.read_text()))
    result['input_sha256']={str(p):digest(p) for p in paths}
    result['qualifier_sha256']=digest(Path(__file__))
    write_json(args.output,result)
    print(json.dumps(result,indent=2))
    raise SystemExit(0 if result['status']=='qualified_for_native_he9_use' else 2)


if __name__=='__main__':main()
