#!/usr/bin/env python3
"""Extend authenticated toy trajectories with exact edits and truthful stopping.

Observed test feedback comes from the previous corpus's isolated Docker controls.
File/edit API responses remain authored simulations, not measured client scores.
No benchmark prompts, expected answers or acceptance fixture files are read.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil

from build_agent_flow_corpus import IMAGE, samples, tool
from coder_common import check, digest, encode_record, load_config, loads, normalize_record, verify_model_proof, write_json
from prepare_coder_data import prepare, read_dataset

FLOW_MANIFEST = '5b71c46bac4a5d951e27b2a86cb09d7dd5bbad8ef986e10b3c16b278a73c41fc'


def conversations(rows, controls):
    records = []
    for row in rows:
        for client in ('opencode', 'hermes'):
            path = '/workspace/exact-edit-' + row['key'] + '/unit.py'
            cwd = str(Path(path).parent)
            tests_path = str(Path(path).with_name('test_unit.py'))
            read, edit, terminal, path_key, old_key, new_key = (
                ('read', 'edit', 'bash', 'filePath', 'oldString', 'newString') if client == 'opencode' else
                ('read_file', 'patch', 'terminal', 'path', 'old_string', 'new_string'))
            schemas = [tool(read, 'Read the complete file with line numbers.', {path_key: 'string'}, [path_key]),
                tool(edit, 'Replace exact, unique text in an observed file.',
                     {path_key: 'string', old_key: 'string', new_key: 'string'}, [path_key, old_key, new_key]),
                tool(terminal, 'Run a foreground command and inspect its actual result.',
                     {'command': 'string', 'workdir': 'string'}, ['command'])]
            messages = [{'role': 'system', 'content': 'Read first. Use an exact grounded edit, preserve valid Python, run the actual tests, and stop after inspecting the result.'},
                {'role': 'user', 'content': f'Use the {client} interface to fix {path}. {row["description"]} Read it and {tests_path}, edit only unit.py, run python3 -m unittest -v in {cwd}, then finish. Keep the tests unchanged.'}]
            serial = 0

            def call(name, arguments, response):
                nonlocal serial
                serial += 1; identifier = 'exact_' + str(serial)
                messages.extend([{'role': 'assistant', 'content': '', 'tool_calls': [{'id': identifier,
                    'type': 'function', 'function': {'name': name, 'arguments': arguments}}]},
                    {'role': 'tool', 'tool_call_id': identifier, 'content': response}])

            def numbered(text):
                separator = ': ' if client == 'opencode' else '|'
                return '\n'.join(str(i) + separator + line for i, line in enumerate(text.splitlines(), 1))

            def observed(state):
                raw = controls[row['key']][state]
                origin = controls[row['key']]['control_cwd']
                return {'returncode': raw['exit_code'], 'stdout': raw['stdout'].replace(origin, cwd),
                        'stderr': '\n'.join(raw['stderr'].replace(origin, cwd).splitlines()[-8:])}

            call(read, {path_key: path}, numbered(row['initial']))
            call(read, {path_key: tests_path}, numbered(row['tests']))
            if row['repair']:
                call(terminal, {'command': 'python3 -m unittest -v', 'workdir': cwd}, json.dumps(observed('before')))
            call(edit, {path_key: path, old_key: row['initial'], new_key: row['fixed']},
                 'Exact unique match replaced. Python syntax is valid.')
            call(terminal, {'command': 'python3 -m unittest -v', 'workdir': cwd}, json.dumps(observed('after')))
            messages.append({'role': 'assistant', 'content': f'Applied the repair and ran python3 -m unittest -v. Exit code 0; all {row["checks"]} tests passed.'})
            records.append({'task_id': 'agent-flow:' + row['family'], 'tools': schemas, 'messages': messages})

            # The non-repair variant already has both actual before/after
            # executions. Reuse them as observation-only tasks; never teach a
            # failure to become success without a further observed execution.
            if not row['repair']:
                for state in ('before', 'after'):
                    response = observed(state)
                    counts = re.findall(r'(?:failures|errors)=(\d+)', response['stderr'])
                    failures = sum(map(int, counts)) if response['returncode'] else 0
                    check(response['returncode'] == 0 or failures > 0,
                          'Failed unittest control lacks its observed failure/error count')
                    response['fail_count'] = failures
                    runner = 'run_tests' if row['variant'] % 2 == 0 else terminal
                    tools = [tool(runner, 'Run the requested check once and return its actual outcome.',
                                  {'command': 'string', 'workdir': 'string'}, ['command', 'workdir'])]
                    prompt = f'Use the {client} interface. Run python3 -m unittest -v exactly once in {cwd}. Report only JSON with returncode and fail_count from the observed result. Do not edit files or rerun the command.'
                    identifier = 'observe_' + state
                    transcript = [{'role': 'system', 'content': 'Execute the requested check once. Report its actual outcome, including failures; then stop.'},
                        {'role': 'user', 'content': prompt + ' The checked state is ' + state + '.'},
                        {'role': 'assistant', 'content': '', 'tool_calls': [{'id': identifier, 'type': 'function',
                            'function': {'name': runner, 'arguments': {'command': 'python3 -m unittest -v', 'workdir': cwd}}}]},
                        {'role': 'tool', 'tool_call_id': identifier, 'content': json.dumps(response)},
                        {'role': 'assistant', 'content': json.dumps({'returncode': response['returncode'], 'fail_count': failures})}]
                    records.append({'task_id': 'agent-flow:' + row['family'], 'tools': tools, 'messages': transcript})
    return records


def main():
    from method_policy import blocked_legacy_operation
    blocked_legacy_operation("Legacy supervised-training corpus preparation")
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('public-corpus', 'existing-corpus', 'flow-corpus', 'output', 'config', 'model-proof'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    check(not args.output.exists(), 'Refusing existing continuation corpus')
    cfg = load_config(args.config); proof = loads(args.model_proof.read_text())
    verify_model_proof(cfg, Path(proof['snapshot']), args.model_proof)
    check(digest(args.flow_corpus / 'prepared/dataset.json') == FLOW_MANIFEST,
          'Previous flow corpus identity changed')
    previous, _ = read_dataset(args.flow_corpus / 'prepared')
    controls_path = args.flow_corpus / 'execution-controls.json'
    build = loads((args.flow_corpus / 'build.json').read_text())
    check(digest(controls_path) == build['execution_controls_sha256'], 'Actual execution controls changed')
    controls = loads(controls_path.read_text())
    check(controls['image'] == IMAGE, 'Controls use a different Python image')
    inputs = [args.public_corpus / (name + '.jsonl') for name in ('code', 'tools')]
    inputs += [args.existing_corpus / 'authored.jsonl', args.flow_corpus / 'agent-flows.jsonl']
    sources = []
    for path in inputs:
        source, = [row for row in previous['sources'] if row['input_sha256'] == digest(path)]
        sources.append(source)
    rows = samples()
    check(set(controls['results']) == {r['key'] for r in rows}, 'Execution controls do not match the toy fixture identities')
    records = [normalize_record(row) for row in conversations(rows, controls['results'])]
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(proof['snapshot'], local_files_only=True, trust_remote_code=False)
    lengths = [len(encode_record(tokenizer, row, cfg['max_length'])['input_ids']) for row in records]
    args.output.mkdir(parents=True)
    shutil.copyfile(__file__, args.output / 'generator.py')
    path = args.output / 'exact-edits-and-observations.jsonl'
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in records))
    source = {'name': 'project-authored-agent-continuation-v1', 'revision': digest(Path(__file__)),
        'license': 'owner-authored; retained with this repository', 'split': 'train',
        'selection': 'Exact string edits and once-only truthful test observations on the previous eight authored toy families. All clients, variants and states retain their previous semantic task identity. No benchmark or acceptance fixture answers imported.'}
    prepared = prepare(inputs + [path], sources + [source], args.output / 'prepared', seed=20261106)
    write_json(args.output / 'build.json', {'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'generator_sha256': digest(Path(__file__)), 'config_sha256': digest(args.config),
        'new_records': len(records), 'maximum_sequence_length': max(lengths),
        'new_records_sha256': digest(path), 'parent_flow_manifest_sha256': FLOW_MANIFEST,
        'execution_controls_sha256': digest(controls_path), 'dataset_manifest_sha256': digest(args.output / 'prepared/dataset.json'),
        'split_counts': {key: value['count'] for key, value in prepared['splits'].items()},
        'scope': 'Development-informed supervised data. File/edit feedback is simulated; unittest outcomes are separately observed offline Docker controls, not a real CLI score.'})
    print(json.dumps({'new_records': len(records), 'max_length': max(lengths),
        'splits': {key: value['count'] for key, value in prepared['splits'].items()}}))


if __name__ == '__main__': main()
