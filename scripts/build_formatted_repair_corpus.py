#!/usr/bin/env python3
"""AST-equivalent readable repair targets and measured one-check status reports.

The prior ten exercise families are reused, with the same task IDs and split.
No acceptance or benchmark input is read. File edits are simulations; broken
and fixed unittest outcomes are separately measured in pinned offline Docker.
"""
import argparse
import ast
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil

from build_semantic_repair_corpus import samples as original_samples, conversations as original_conversations
from build_agent_flow_corpus import controlled_results, tool, IMAGE
from coder_common import check, digest, encode_record, load_config, loads, normalize_record, verify_model_proof, write_json
from prepare_coder_data import prepare, read_dataset


def samples(variants=12):
    rows = original_samples(variants)
    for row in rows:
        for key in ('initial', 'fixed'):
            row[key] = format_python(row[key])
        tree = ast.parse(row['tests'])
        checks = next(node for node in tree.body if isinstance(node,ast.ClassDef))
        method, = checks.body
        setup = []
        statements = list(method.body)
        while statements and isinstance(statements[0],(ast.Assign,ast.AnnAssign)):
            setup.append(statements.pop(0))
        # Adjacent assertions can form a call/postcondition pair. Keep them
        # together; negative contexts are separate independently set-up cases.
        groups = []
        for statement in statements:
            if isinstance(statement,ast.Expr) and groups and all(isinstance(n,ast.Expr) for n in groups[-1]):
                groups[-1].append(statement)
            else:
                groups.append([statement])
        methods = []
        for index,group in enumerate(groups):
            case = copy.deepcopy(method)
            case.name = 'test_contract_'+str(index)
            case.body = copy.deepcopy(setup)+copy.deepcopy(group)
            methods.append(case)
        checks.body = methods
        row['tests'] = ast.unparse(ast.fix_missing_locations(tree))+'\n'
        row['checks'] = len(methods)
    return rows


def format_python(source):
    """Expand compact suites while preserving the complete executable AST."""
    before = ast.parse(source)
    formatted = ast.unparse(before) + '\n'
    check(ast.dump(before) == ast.dump(ast.parse(formatted)), 'Formatting changed program semantics')
    return formatted

def repair_records(rows, controls):
    records = []
    for row in rows:
        for record in original_conversations([row],controls):
            if record['tools']:
                runner = record['tools'][-1]['function']['name']
                args = record['messages'][-3]['tool_calls'][0]['function']['arguments']
                observed = controls[row['key']]['before']
                diagnostics = re.findall(r'^\w*(?:Error|Exception):[^\n]*',observed['stderr'],re.M)[-2:]
                output = {'exit_code':observed['exit_code'],'stdout':observed['stdout'],
                          'stderr':'\n'.join(diagnostics+observed['stderr'].splitlines()[-4:])}
                record['messages'][6:6] = [
                    {'role':'assistant','content':'','tool_calls':[{'id':'before_check','type':'function','function':{'name':runner,'arguments':dict(args)}}]},
                    {'role':'tool','tool_call_id':'before_check','content':json.dumps(output)}]
                record['messages'][0]['content'] += ' Use readable Python blocks with normal indentation. After a failed check, inspect and edit the implementation before rerunning it.'
            else:
                record['messages'][0]['content'] += ' Use readable Python blocks with normal indentation.'
            records.append(record)
    return records


def observation_records(rows, controls):
    """Reporting requests stop after one actual outcome, including failures."""
    records = []
    for row in rows:
        for state in ('before', 'after'):
            result = controls[row['key']][state]
            count = sum(map(int,re.findall(r'(?:failures|errors)=(\d+)',result['stderr'])))
            check(result['exit_code'] == 0 or count > 0, 'Missing observed failure counts')
            name = 'run_tests' if row['variant'] % 2 == 0 else 'check_suite'
            keys = ('exit_code','failed') if row['variant'] % 3 else ('returncode','fail_count')
            answer = {keys[0]:result['exit_code'],keys[1]:count}
            observed = {**answer,'stdout':result['stdout'],'stderr':'\n'.join(result['stderr'].splitlines()[-4:])}
            target = 'toy-'+row['key']+'-'+state
            records.append({'task_id':'semantic-repair:'+row['family'],
                'tools':[tool(name,'Execute the named unit check and return its measured status.',{'target':'string'},['target'])],
                'messages':[{'role':'user','content':f'Check {target}. Respond only with a JSON object containing {keys[0]} and {keys[1]} from the tool result. This is a status report, so leave the files unchanged.'},
                    {'role':'assistant','content':'','tool_calls':[{'id':'outcome','type':'function','function':{'name':name,'arguments':{'target':target}}}]},
                    {'role':'tool','tool_call_id':'outcome','content':json.dumps(observed)},
                    {'role':'assistant','content':json.dumps(answer)}]})
    return records

def main():
    from method_policy import blocked_legacy_operation
    blocked_legacy_operation("Legacy supervised-training corpus preparation")
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('public-corpus','existing-corpus','output','config','model-proof'):
        parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args()
    check(not args.output.exists(), 'Refusing existing semantic repair corpus')
    cfg = load_config(args.config); proof = loads(args.model_proof.read_text())
    verify_model_proof(cfg,Path(proof['snapshot']),args.model_proof)
    parent,_ = read_dataset(args.existing_corpus/'prepared')
    inputs = [args.public_corpus/(name+'.jsonl') for name in ('code','tools')]+[args.existing_corpus/'authored.jsonl']
    sources = []
    for path in inputs:
        source, = [s for s in parent['sources'] if s['input_sha256']==digest(path)]
        sources.append(source)
    rows = samples(); controls = controlled_results(rows,IMAGE)
    records = [normalize_record(r) for r in repair_records(rows,controls)+observation_records(rows,controls)]
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(proof['snapshot'],local_files_only=True,trust_remote_code=False)
    lengths = [len(encode_record(tokenizer,r,cfg['max_length'])['input_ids']) for r in records]
    args.output.mkdir(parents=True)
    shutil.copyfile(__file__,args.output/'generator.py')
    parent_generator = Path(__file__).with_name('build_semantic_repair_corpus.py')
    shutil.copyfile(parent_generator,args.output/'parent-generator.py')
    # The original authored replay is retained and authenticated in provenance;
    # only its code formatting and formatting instruction change in this input.
    original = inputs[-1]; replay = []
    for line in original.read_text().splitlines():
        record = loads(line)
        if record['task_id'].startswith('authored-code:'):
            record['messages'][-1]['content'] = format_python(record['messages'][-1]['content'])
            record['messages'][-2]['content'] = record['messages'][-2]['content'].replace(
                'Return only concise complete Python code, without examples, docstrings or commentary.',
                'Return only complete, readable Python code with normal indented blocks.')
        replay.append(record)
    formatted = args.output/'authored-formatted.jsonl'
    formatted.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in replay))
    inputs[-1] = formatted
    sources[-1] = {**sources[-1], 'name':sources[-1]['name']+'-formatted',
        'original_input_sha256':digest(original), 'revision':digest(Path(__file__)),
        'transformation':'AST-equivalent code formatting and readable-code instruction; original task IDs/splits retained.'}
    path = args.output/'semantic-repairs.jsonl'
    path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
    write_json(args.output/'execution-controls.json',{'image':IMAGE,'results':controls})
    source = {'name':'project-authored-formatted-semantic-repair-v2','revision':digest(Path(__file__)),
        'license':'Apache-2.0; owner-authored in this repository','split':'train',
        'selection':'Development-informed separate exercises; no benchmark/acceptance files or answers imported. All family variants and interfaces share a split.'}
    prepared = prepare(inputs+[path],sources+[source],args.output/'prepared',seed=20261106)
    write_json(args.output/'build.json',{'created_at_utc':datetime.now(timezone.utc).isoformat(),
        'generator_sha256':digest(Path(__file__)), 'parent_generator_sha256':digest(parent_generator),'config_sha256':digest(args.config),'new_records':len(records),
        'maximum_sequence_length':max(lengths),'new_records_sha256':digest(path),
        'original_authored_sha256':digest(original),'formatted_authored_sha256':digest(formatted),
        'formatting':'AST-equivalent standard blocks, no compound statement compression',
        'execution_controls_sha256':digest(args.output/'execution-controls.json'),
        'dataset_manifest_sha256':digest(args.output/'prepared/dataset.json'),
        'split_counts':{k:v['count'] for k,v in prepared['splits'].items()},
        'scope':'Synthetic repair trajectories with measured offline controls, not actual client performance. Compact development/acceptance tasks are now tuning-informed.'})
    print(json.dumps({'new_records':len(records),'maximum_sequence_length':max(lengths),'splits':{k:v['count'] for k,v in prepared['splits'].items()}}))

if __name__ == '__main__': main()
