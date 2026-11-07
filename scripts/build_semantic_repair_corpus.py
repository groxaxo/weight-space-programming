#!/usr/bin/env python3
"""Development-informed semantic repair exercises with isolated execution controls.

The generator reads public replay, not benchmark prompts, acceptance fixtures or
their answers. Tool feedback for file edits is simulated; unittest feedback is
measured in the pinned offline container. Variants and both clients share one
family identity, including across code and project trajectories.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from build_agent_flow_corpus import controlled_results, tool, IMAGE
from coder_common import check, digest, encode_record, load_config, loads, normalize_record, verify_model_proof, write_json
from prepare_coder_data import prepare, read_dataset


def fixtures():
    # No chunking function or scheduler normalization/planning solution is used.
    return [
        ('positive_power', 'base, exponent',
         'Return base raised to exponent. Exponent must be a strictly positive built-in integer, excluding bool; otherwise raise ValueError.',
         '    if type(exponent) is not int or exponent <= 0: raise ValueError("exponent")\n    return base ** exponent\n',
         '    return base ** exponent\n',
         'self.assertEqual(FN(3,2),9)\nself.assertEqual(FN(2,5),32)\nfor bad in (True,False,0,-1,2.0,"2",None):\n    with self.assertRaises(ValueError): FN(2,bad)'),
        ('repeat_token', 'token, count',
         'Return token repeated count times. Count is a nonnegative built-in int, never bool. Invalid counts raise ValueError, including for an empty token.',
         '    if type(count) is not int or count < 0: raise ValueError("count")\n    return token * count\n',
         '    if not isinstance(count,int) or count < 0: raise ValueError("count")\n    return token * count\n',
         'self.assertEqual(FN("x",3),"xxx")\nself.assertEqual(FN("x",0),"")\nfor bad in (True,False,-1,1.5,None):\n    with self.assertRaises(ValueError): FN("",bad)'),
        ('unique_labels', 'labels',
         'Materialize an iterable of distinct nonempty strings as a tuple in original order. Reject bare strings, None, duplicate or invalid labels with ValueError. Accept one-shot generators.',
         '    if labels is None or isinstance(labels,str): raise ValueError("labels")\n    result = tuple(labels)\n    if any(not isinstance(x,str) or not x for x in result) or len(set(result)) != len(result): raise ValueError("labels")\n    return result\n',
         '    if any(not isinstance(x,str) or not x for x in labels): raise ValueError("labels")\n    return tuple(labels)\n',
         'self.assertEqual(FN(x for x in ["red","blue"]),("red","blue"))\nself.assertEqual(FN(iter(())),())\nfor bad in (None,"ab",["a","a"],[""],[2]):\n    with self.assertRaises(ValueError): FN(bad)'),
        ('weighted_total', 'pairs',
         'Return sum(value*weight) divided by sum(weight) for an iterable of numeric pairs. Materialize once; generators work. Raise ValueError if the total weight is zero.',
         '    pairs = list(pairs)\n    weight = sum(w for v,w in pairs)\n    if weight == 0: raise ValueError("zero weight")\n    return sum(v*w for v,w in pairs) / weight\n',
         '    weight = sum(w for v,w in pairs)\n    return sum(v*w for v,w in pairs) / weight\n',
         'self.assertEqual(FN(iter([(2,1),(8,2)])),6)\nself.assertEqual(FN([(5,4)]),5)\nwith self.assertRaises(ValueError): FN(iter(()))'),
        ('named_ports', 'rows',
         'Index an iterable of (name, port) pairs. Names must be unique nonempty strings; ports must be built-in ints from 1 to 65535, excluding bool. Raise ValueError on invalid rows; preserve input.',
         '    result = {}\n    for name,port in rows:\n        if not isinstance(name,str) or not name or name in result or type(port) is not int or not 1 <= port <= 65535: raise ValueError("port record")\n        result[name] = port\n    return result\n',
         '    return dict(rows)\n',
         'data=[("web",80),("tls",443)]\nself.assertEqual(FN(iter(data)),dict(data))\nself.assertEqual(data,[("web",80),("tls",443)])\nfor bad in ([("a",2),("a",3)],[("",2)],[("a",True)],[("a",0)]):\n    with self.assertRaises(ValueError): FN(bad)'),
        ('ready_priority', 'items, ready',
         'Return the first item in original items order that belongs to ready, or None. Both are arbitrary iterables, including generators. Selection follows items order, not ready order.',
         '    ready = set(ready)\n    return next((item for item in items if item in ready),None)\n',
         '    return next(iter(ready),None)\n',
         'self.assertEqual(FN(iter(["c","a","b"]),iter(["b","a"])),"a")\nself.assertEqual(FN(["early","late"],["late","early"]),"early")\nself.assertIsNone(FN(["x"],[]))'),
        ('inventory_names', 'records',
         'Read dict records with key name and optional aliases (default empty). Return (name, tuple(unique aliases)) preserving all orders. Reject duplicate/empty/nonstring names, bare-string or None aliases, or invalid alias strings with ValueError. Generators must not be consumed twice.',
         '    result, seen = [], set()\n    for record in records:\n        name = record["name"]\n        aliases = record.get("aliases",())\n        if not isinstance(name,str) or not name or name in seen or aliases is None or isinstance(aliases,str): raise ValueError("record")\n        aliases = tuple(aliases)\n        if any(not isinstance(a,str) or not a for a in aliases): raise ValueError("alias")\n        seen.add(name)\n        result.append((name,tuple(dict.fromkeys(aliases))))\n    return result\n',
         '    return [(r["name"],tuple(r.get("aliases",()))) for r in records]\n',
         'rows=({"name":"red","aliases":iter(["r","r","rouge"])},{"name":"blue"})\nself.assertEqual(FN(iter(rows)),[("red",("r","rouge")),("blue",())])\nfor bad in ([{"name":"a"},{"name":"a"}],[{"name":"a","aliases":None}],[{"name":"a","aliases":"abc"}],[{"name":""}]):\n    with self.assertRaises(ValueError): FN(bad)'),
        ('interval_measure', 'spans',
         'Sum individual widths of half-open (start, stop) spans without merging. Equal endpoints are valid and contribute zero. A start strictly greater than stop raises ValueError. Accept generators.',
         '    total = 0\n    for start,stop in spans:\n        if start > stop: raise ValueError("reversed")\n        total += stop-start\n    return total\n',
         '    total = 0\n    for start,stop in spans:\n        if start >= stop: raise ValueError("reversed")\n        total += stop-start\n    return total\n',
         'self.assertEqual(FN(iter([(2,2),(1,5),(-3,-1)])),6)\nself.assertEqual(FN([]),0)\nwith self.assertRaises(ValueError): FN([(4,2)])'),
        ('distinct_columns', 'rows',
         'Return column count for rectangular iterable rows. Materialize every row once, including generator rows. Empty outer iterable gives zero. Ragged rows raise ValueError.',
         '    rows = [tuple(row) for row in rows]\n    if not rows: return 0\n    width = len(rows[0])\n    if any(len(row) != width for row in rows): raise ValueError("ragged")\n    return width\n',
         '    return len(list(next(iter(rows))))\n',
         'self.assertEqual(FN(iter([iter([1,2]),iter([3,4])])),2)\nself.assertEqual(FN(iter(())),0)\nwith self.assertRaises(ValueError): FN([[1],[2,3]])'),
        ('integer_offsets', 'lengths',
         'Return exclusive prefix offsets starting with zero for iterable lengths. Lengths are nonnegative built-in ints, excluding bool. Reject invalid elements with ValueError and preserve zero lengths.',
         '    offsets = [0]\n    for length in lengths:\n        if type(length) is not int or length < 0: raise ValueError("length")\n        offsets.append(offsets[-1]+length)\n    return offsets\n',
         '    return [0] + list(lengths)\n',
         'self.assertEqual(FN(iter([2,0,3])),[0,2,2,5])\nself.assertEqual(FN([]),[0])\nfor bad in ([True],[-1],[1.5]):\n    with self.assertRaises(ValueError): FN(bad)'),
    ]


def samples(variants=12):
    rows = []
    for family, signature, description, good, bad, assertions in fixtures():
        for variant in range(variants):
            name = f'repair_{family}_{variant}'
            prefix = f'def {name}({signature}):\n'
            lines = ['import unittest', f'from unit import {name}', 'class Checks(unittest.TestCase):', '    def test_contract(self):']
            lines.extend('        '+line for line in assertions.replace('FN',name).splitlines())
            rows.append({'key':f'{family}-{variant}', 'family':family, 'variant':variant,
                         'description':description, 'checks':1, 'initial':prefix+bad, 'fixed':prefix+good,
                         'tests':'\n'.join(lines)+'\n'})
    return rows


def conversations(rows, controls):
    records = []
    for row in rows:
        family = 'semantic-repair:' + row['family']
        name = row['fixed'].split('(')[0].removeprefix('def ')
        records.append({'task_id':family, 'tools':[], 'messages':[
            {'role':'user','content':f'Implement {name}. {row["description"]} Return only complete Python code.'},
            {'role':'assistant','content':row['fixed']}]})
        for client in ('hermes','opencode'):
            path = '/workspace/repair-'+row['key']+'/unit.py'; cwd = str(Path(path).parent)
            read, edit, runner, pk, oldk, newk = (('read_file','patch','terminal','path','old_string','new_string') if client == 'hermes' else
                ('read','edit','bash','filePath','oldString','newString'))
            schemas = [tool(read,'Read the file with line numbers.',{pk:'string'},[pk]),
                tool(edit,'Replace unique observed text.',{pk:'string',oldk:'string',newk:'string'},[pk,oldk,newk]),
                tool(runner,'Run a foreground check and inspect its result.',{'command':'string','workdir':'string'},['command'])]
            messages = [{'role':'system','content':'Read, repair the implementation, run tests, inspect the result, then finish. Keep tests unchanged. Never rerun a successful unchanged check.'},
                {'role':'user','content':f'Fix {path}. {row["description"]} Read unit.py and test_unit.py. Use {client} tools. Run python3 -m unittest -v in {cwd} and report its observed outcome.'}]
            def call(name, args, response):
                identifier = 'repair_'+str(len(messages))
                messages.extend([{'role':'assistant','content':'','tool_calls':[{'id':identifier,'type':'function','function':{'name':name,'arguments':args}}]},
                                 {'role':'tool','tool_call_id':identifier,'content':response}])
            sep = '|' if client == 'hermes' else ': '
            numbered = lambda text:'\n'.join(str(i)+sep+line for i,line in enumerate(text.splitlines(),1))
            call(read,{pk:path},numbered(row['initial']))
            call(read,{pk:str(Path(path).with_name('test_unit.py'))},numbered(row['tests']))
            call(edit,{pk:path,oldk:row['initial'],newk:row['fixed']},'Unique replacement applied; syntax valid.')
            observed = controls[row['key']]['after']
            check(observed['exit_code'] == 0, 'Unsuccessful repair control')
            call(runner,{'command':'python3 -m unittest -v','workdir':cwd},json.dumps({
                'exit_code':observed['exit_code'], 'stdout':observed['stdout'],
                'stderr':'\n'.join(observed['stderr'].splitlines()[-4:])}))
            messages.append({'role':'assistant','content':'Repaired the implementation. python3 -m unittest -v exited 0; the contract test passed.'})
            records.append({'task_id':family,'tools':schemas,'messages':messages})
    return records


def main():
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
    records = [normalize_record(r) for r in conversations(rows,controls)]
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(proof['snapshot'],local_files_only=True,trust_remote_code=False)
    lengths = [len(encode_record(tokenizer,r,cfg['max_length'])['input_ids']) for r in records]
    args.output.mkdir(parents=True)
    shutil.copyfile(__file__,args.output/'generator.py')
    path = args.output/'semantic-repairs.jsonl'
    path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
    write_json(args.output/'execution-controls.json',{'image':IMAGE,'results':controls})
    source = {'name':'project-authored-semantic-repair-v1','revision':digest(Path(__file__)),
        'license':'Apache-2.0; owner-authored in this repository','split':'train',
        'selection':'Development-informed separate exercises; no benchmark/acceptance files or answers imported. All family variants and interfaces share a split.'}
    prepared = prepare(inputs+[path],sources+[source],args.output/'prepared',seed=20261106)
    write_json(args.output/'build.json',{'created_at_utc':datetime.now(timezone.utc).isoformat(),
        'generator_sha256':digest(Path(__file__)),'config_sha256':digest(args.config),'new_records':len(records),
        'maximum_sequence_length':max(lengths),'new_records_sha256':digest(path),
        'execution_controls_sha256':digest(args.output/'execution-controls.json'),
        'dataset_manifest_sha256':digest(args.output/'prepared/dataset.json'),
        'split_counts':{k:v['count'] for k,v in prepared['splits'].items()},
        'scope':'Synthetic repair trajectories with measured offline controls, not actual client performance. Compact development/acceptance tasks are now tuning-informed.'})
    print(json.dumps({'new_records':len(records),'maximum_sequence_length':max(lengths),'splits':{k:v['count'] for k,v in prepared['splits'].items()}}))


if __name__ == '__main__': main()
