#!/usr/bin/env python3
"""Authored agent trajectories, with actual isolated unittest controls.

This is synthetic supervised data, not a measured OpenCode/Hermes score.
Only the supplied public replay and previous authored corpus are read. No
benchmark/task/acceptance fixture files or canonical answers are imported.
The main project and currently executing benchmark source are not modified.
"""
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

IMAGE='python:3.12-slim@sha256:4840d514c583a4d3d8cad3940e68dee6f0f6f3635c12fcd0a3d74c21a5b93a54'
OLD_AUTHORED='55f8f4224ee640108138ab195aae6b90adbd3dc03f8ad40ab6fce81876b77499'
SYSTEM='You are a coding assistant. Read the supplied files, make a grounded edit, run the actual checks, inspect their result, then finish. Do not repeat an unchanged successful check or invent a result.'


def fixtures():
    # These functions are independently authored; none implements an HE9 task
    # or either of the acceptance clients' discount/scheduler fixtures.
    return [
      ('strip_prefix','text, prefix','Remove one leading prefix, leaving other occurrences unchanged.',
       '    if not isinstance(text,str) or not isinstance(prefix,str): raise ValueError("strings required")\n    return text[len(prefix):] if prefix and text.startswith(prefix) else text\n',
       [('("ababa", "ab")','"aba"'),('("xabab", "ab")','"xabab"'),('("same", "")','"same"')]),
      ('overlay_pairs','pairs','Apply key/value pairs in order; later values replace earlier values.',
       '    result = {}\n    for key,value in pairs: result[key] = value\n    return result\n',
       [('([(1,"a"),(2,"b"),(1,"c")],)','{1:"c",2:"b"}'),('(iter(()),)','{}')]),
      ('clamp_integer','value, low, high','Clamp an integer to inclusive bounds; reject bool and reversed bounds.',
       '    if any(type(v) is not int for v in (value,low,high)) or low>high: raise ValueError("invalid bounds or type")\n    return max(low,min(high,value))\n',
       [('(-3,0,5)','0'),('(10,0,5)','5'),('(3,0,5)','3')]),
      ('transpose_rows','rows','Transpose rectangular iterable rows into tuples; empty input gives an empty list.',
       '    data = [tuple(row) for row in rows]\n    if not data: return []\n    if any(len(row)!=len(data[0]) for row in data): raise ValueError("ragged rows")\n    return list(zip(*data))\n',
       [('([[1,2],[3,4]],)','[(1,3),(2,4)]'),('(iter(()),)','[]'),('([[],[]],)','[]')]),
      ('decode_switch','value','Parse on/off strings case-insensitively after stripping; reject every other value.',
       '    if not isinstance(value,str) or value.strip().lower() not in ("on","off"): raise ValueError("invalid switch")\n    return value.strip().lower()=="on"\n',
       [('(" ON ",)','True'),('("off",)','False')]),
      ('range_count','start, stop, step','Count a positive-step integer range exactly without converting integers to floats.',
       '    if any(type(v) is not int for v in (start,stop,step)) or step<=0: raise ValueError("positive integer step required")\n    return max(0,(stop-start+step-1)//step)\n',
       [('(0,10,3)','4'),('(5,2,3)','0'),('(0,10**30+1,2)','(10**30+2)//2')]),
      ('normalize_index','index, length','Normalize a negative integer index; reject out-of-range indices and invalid lengths.',
       '    if type(index) is not int or type(length) is not int or length<0: raise ValueError("invalid index")\n    result = index+length if index<0 else index\n    if not 0<=result<length: raise IndexError("out of range")\n    return result\n',
       [('(-1,4)','3'),('(0,4)','0'),('(-4,4)','0')]),
      ('reduce_ratio','numerator, denominator','Return a reduced integer ratio with positive denominator; reject a zero denominator.',
       '    from math import gcd\n    if type(numerator) is not int or type(denominator) is not int or denominator==0: raise ValueError("invalid ratio")\n    factor = gcd(numerator,denominator)\n    sign = -1 if denominator<0 else 1\n    return sign*numerator//factor,sign*denominator//factor\n',
       [('(6,-8)','(-3,4)'),('(0,7)','(0,1)'),('(-9,-6)','(3,2)')]),
    ]


def samples(variants=8):
    rows=[]
    for family,signature,description,body,checks in fixtures():
        for variant in range(variants):
            fn='helper_'+family+'_'+str(variant)
            fixed=f'def {fn}({signature}):\n    """{description}"""\n'+body
            for repair in (False,True):
                # Repair examples intentionally include an incomplete docstring;
                # both before/after states are exercised by the control runner.
                initial=(f'def {fn}({signature}):\n    """{description}\n    return None\n' if repair else
                         f'def {fn}({signature}):\n    """{description}"""\n    return None\n')
                lines=['import unittest',f'from unit import {fn}','class Checks(unittest.TestCase):']
                for i,(arguments,expected) in enumerate(checks):
                    lines.append(f'    def test_{i}(self): self.assertEqual({fn}(*{arguments}), {expected})')
                lines.append('if __name__=="__main__": unittest.main()')
                tests='\n'.join(lines)+'\n'
                key=family+'-'+str(variant)+'-'+str(int(repair))
                rows.append({'key':key,'family':family,'variant':variant,'repair':repair,'initial':initial,
                    'fixed':fixed,'tests':tests,'description':description,'checks':len(checks)})
    return rows


CONTROL_RUNNER=r'''
from pathlib import Path
import json,os,subprocess,sys,tempfile
rows=json.loads(Path('/controls/fixtures.json').read_text()); results={}
for row in rows:
 with tempfile.TemporaryDirectory(dir='/tmp') as directory:
  root=Path(directory);(root/'test_unit.py').write_text(row['tests'])
  states={}
  for state,source in [('before',row['initial']),('after',row['fixed'])]:
   (root/'unit.py').write_text(source)
   p=subprocess.run([sys.executable,'-m','unittest','-v'],cwd=root,capture_output=True,text=True,
       env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},timeout=5)
   states[state]={'exit_code':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
  assert states['before']['exit_code']!=0 and states['after']['exit_code']==0,row['key']
  expected = 'Ran '+str(row['checks'])+' test'+('s' if row['checks'] != 1 else '')+' in '
  assert expected in states['after']['stderr'],row['key']
  results[row['key']]={'control_cwd':str(root),**states}
print(json.dumps(results))
'''


def controlled_results(rows,image):
    with tempfile.TemporaryDirectory(prefix='wsp-flow-controls-') as directory:
        root=Path(directory);root.chmod(0o755)
        (root/'fixtures.json').write_text(json.dumps(rows));(root/'run.py').write_text(CONTROL_RUNNER)
        (root/'fixtures.json').chmod(0o644);(root/'run.py').chmod(0o644)
        name='wsp-flow-'+root.name.rsplit('-',1)[-1]
        command=['docker','run','--rm','--name',name,'--network','none','--read-only','--cap-drop','ALL',
            '--security-opt','no-new-privileges','--pids-limit','48','--memory','512m','--cpus','1',
            '--user','65534:65534','--tmpfs','/tmp:rw,nosuid,size=64m,mode=1777',
            '--mount',f'type=bind,src={root},dst=/controls,readonly','--entrypoint','python3',image,'-B','/controls/run.py']
        try:
            result=subprocess.run(command,capture_output=True,text=True,timeout=180,check=True)
            return json.loads(result.stdout)
        finally:subprocess.run(['docker','rm','-f',name],capture_output=True,timeout=15)


def tool(name,description,properties,required):
    return {'type':'function','function':{'name':name,'description':description,'parameters':{
        'type':'object','properties':{k:{'type':v} for k,v in properties.items()},
        'required':required,'additionalProperties':False}}}


def conversations(rows,controls):
    records=[]
    for row in rows:
        for client in ('opencode','hermes'):
            path='/workspace/flow-'+row['key']+'/unit.py'
            tests_path=str(Path(path).with_name('test_unit.py'))
            cwd=str(Path(path).parent)
            if client=='opencode':
                read,write,terminal='read','write','bash';path_key='filePath'
            else:read,write,terminal='read_file','write_file','terminal';path_key='path'
            tools=[tool(read,'Read the complete file with line numbers.',{path_key:'string'},[path_key]),
                tool(write,'Replace the whole file after reading it; preserve valid Python.',{path_key:'string','content':'string'},[path_key,'content']),
                tool(terminal,'Run a foreground command once and inspect its actual result.',{'command':'string','workdir':'string'},['command'])]
            prompt=f'Use the {client} tool interface. Fix {path}. {row["description"]} Read {path} and {tests_path}, edit only unit.py, run python3 -m unittest -v in {cwd}, inspect the result, and finish after success. Do not edit the tests.'
            messages=[{'role':'system','content':SYSTEM},{'role':'user','content':prompt}]
            serial=0
            def call(name,args,response):
                nonlocal serial
                serial+=1;call_id='flow_'+str(serial)
                messages.extend([{'role':'assistant','content':'','tool_calls':[{'id':call_id,'type':'function','function':{'name':name,'arguments':args}}]},
                    {'role':'tool','tool_call_id':call_id,'content':response}])
            def numbered(text):
                separator=': ' if client=='opencode' else '|'
                return '\n'.join(str(i)+separator+line for i,line in enumerate(text.splitlines(),1))
            def observed(state):
                raw=controls[row['key']][state]
                origin=controls[row['key']]['control_cwd']
                result={'exit_code':raw['exit_code'],'stdout':raw['stdout'].replace(origin,cwd),
                    'stderr':'\n'.join(raw['stderr'].replace(origin,cwd).splitlines()[-8:])}
                return json.dumps(result)
            call(read,{path_key:path},numbered(row['initial']))
            call(read,{path_key:tests_path},numbered(row['tests']))
            if row['repair']:
                call(terminal,{'command':'python3 -m unittest -v','workdir':cwd},observed('before'))
            call(write,{path_key:path,'content':row['fixed']},'Complete file written; content verified.')
            # Preserve actual observed outcomes, while rebinding the isolated
            # control path to this authored workspace and bounding stderr.
            call(terminal,{'command':'python3 -m unittest -v','workdir':cwd},observed('after'))
            messages.append({'role':'assistant','content':f'Implemented the required helper and preserved the tests. python3 -m unittest -v completed with exit code 0; {row["checks"]} tests passed.'})
            records.append({'task_id':'agent-flow:'+row['family'],'tools':tools,'messages':messages})
    return records


def main():
    from method_policy import blocked_legacy_operation
    blocked_legacy_operation("Legacy supervised-training corpus preparation")
    parser=argparse.ArgumentParser(description=__doc__)
    for flag in ('project','public-corpus','existing-corpus','output','config','model-proof'):
        parser.add_argument('--'+flag,type=Path,required=True)
    parser.add_argument('--image',default=IMAGE)
    args=parser.parse_args();project=args.project.resolve()
    sys.path[:0]=[str(project),str(project/'scripts')]
    from coder_common import check,digest,encode_record,load_config,loads,normalize_record,verify_model_proof,write_json
    from prepare_coder_data import prepare,read_dataset
    check(args.image==IMAGE,'The pinned authored-control Python image is required')
    check(not args.output.exists(),'Refusing an existing trajectory corpus')
    cfg=load_config(args.config);proof=loads(args.model_proof.read_text())
    verify_model_proof(cfg,Path(proof['snapshot']),args.model_proof)
    previous,_=read_dataset(args.existing_corpus/'prepared')
    inputs=[];sources=[]
    for name in ('code','tools'):
        path=args.public_corpus/(name+'.jsonl');identity=loads((args.public_corpus/(name+'.source.json')).read_text())
        expected,=[r for r in previous['sources'] if r['name']==identity['name']]
        check(digest(path)==expected['input_sha256'],'Public replay input changed')
        inputs.append(path);sources.append(identity)
    authored=args.existing_corpus/'authored.jsonl'
    check(digest(authored)==OLD_AUTHORED,'Previous authored data changed')
    prior,=[r for r in previous['sources'] if r['name']=='project-authored-corrective-v1']
    inputs.append(authored);sources.append(prior)
    rows=samples();controls=controlled_results(rows,args.image)
    records=[normalize_record(r) for r in conversations(rows,controls)]
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(proof['snapshot'],local_files_only=True,trust_remote_code=False)
    lengths=[len(encode_record(tokenizer,r,cfg['max_length'])['input_ids']) for r in records]
    args.output.mkdir(parents=True)
    shutil.copyfile(__file__,args.output/'generator.py')
    write_json(args.output/'execution-controls.json',{'image':args.image,'fixtures':len(rows),'results':controls,
        'scope':'Actual offline Docker unittest subprocesses for authored fixtures. This is not an OpenCode/Hermes model evaluation.'})
    path=args.output/'agent-flows.jsonl'
    path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
    source={'name':'project-authored-agent-flows-v1','revision':digest(Path(__file__)),'license':'owner-authored; retained with this repository',
        'split':'train','selection':'Synthetic full-file repair/terminal completion trajectories with independently executed toy tests. No benchmark or acceptance fixture files read. All variants/clients of a semantic helper share one task identity.'}
    prepared=prepare(inputs+[path],sources+[source],args.output/'prepared',seed=20261106)
    write_json(args.output/'build.json',{'created_at_utc':datetime.now(timezone.utc).isoformat(),
        'generator_sha256':digest(Path(__file__)),'config_sha256':digest(args.config),'agent_flows_sha256':digest(path),
        'previous_authored_sha256':OLD_AUTHORED,'execution_controls_sha256':digest(args.output/'execution-controls.json'),
        'new_records':len(records),'families':len({r['task_id'] for r in records}),'maximum_sequence_length':max(lengths),
        'tool_output_transform':'Rebind the isolated temporary root to each authored workspace; retain stderr last eight lines and actual exit/stdout. Raw controls are retained separately.',
        'dataset_manifest_sha256':digest(args.output/'prepared/dataset.json'),
        'split_counts':{k:v['count'] for k,v in prepared['splits'].items()},
        'scope':'Development-informed synthetic trajectories, not untouched benchmark evidence or real CLI results. Canonical HE9 answers and acceptance fixture code are excluded.'})
    print(json.dumps({'new_records':len(records),'max_length':max(lengths),'splits':{k:v['count'] for k,v in prepared['splits'].items()}}))


if __name__=='__main__':main()
