#!/usr/bin/env python3
"""One front door for the compatible, no-training weight-space research logic."""
from __future__ import annotations
import argparse
import copy
import json
import math
from pathlib import Path
import re
import subprocess
import sys

import direct_evidence as evidence
from coder_common import canonical
from direct_probe import _ids
from direct_weights import ROOT, load_lock, parent_identity, publish_json, read_json
from pipeline import require, sha256
from wsp_interventions import ARMS, validate as validate_intervention

PROFILE = {'max_new_tokens': 96, 'max_input_tokens': 512, 'do_sample': False,
           'thinking': False, 'seed': 20261109, 'threads': 1}
EXTRA_SOURCES = ('scripts/wsp_lab.py', 'scripts/wsp_interventions.py', 'scripts/wsp_pairs.py',
                 'scripts/wsp_worker.py', 'ops/wsp_lab54.py', 'scripts/method_policy.py',
                 'research-policy.json', 'configs/wsp-lab-sources.json')


def source_manifest():
    return {name: sha256(ROOT / name) for name in (*evidence.PRODUCER_FILES, *EXTRA_SOURCES)}


def exact(value, names, label):
    require(type(value) is dict and set(value) == set(names), 'Unexpected ' + label + ' fields')


def digest(value):
    require(type(value) is str and re.fullmatch('[0-9a-f]{64}', value), 'Invalid SHA-256')


def active_policy():
    value = read_json(ROOT / 'research-policy.json')
    require(value['active_method'] == 'forward-only-direct-bf16-weight-editing'
            and all(value[k] is False for k in ('training_allowed', 'lora_allowed', 'backward_allowed',
                    'optimizers_allowed', 'iterative_loss_fitting_allowed', 'reuse_08b_coordinates_allowed',
                    'service_preemption_allowed', 'cleanup_allowed'))
            and value['preserve_all_artifacts'] is True, 'Active no-training policy differs')


def validate_case(case):
    from jsonschema import Draft202012Validator
    require(type(case) is dict and {'id','messages','tools'} <= set(case) <= {'id','messages','tools','assessment'},
            'Unexpected case fields')
    if 'assessment' in case:
        from wsp_pairs import validate_assessment
        validate_assessment(case['assessment'])
    require(type(case['id']) is str and re.fullmatch('[a-z0-9_-]{1,80}', case['id']), 'Invalid case identity')
    require(type(case['tools']) is list and len(case['tools']) <= 4, 'Invalid bounded tool schemas')
    schemas = {}
    for tool in case['tools']:
        exact(tool, ('type', 'function'), 'tool'); function = tool['function']
        require(tool['type'] == 'function' and type(function) is dict
                and {'name', 'parameters'} <= set(function) <= {'name', 'parameters', 'description'}, 'Invalid function tool')
        name = function['name']
        require(type(name) is str and re.fullmatch('[a-zA-Z_][a-zA-Z0-9_]{0,63}', name) and name not in schemas,
                'Invalid/duplicate tool name')
        schema = function['parameters']
        require(type(schema) is dict and schema.get('type') == 'object' and len(canonical(tool)) <= 16384,
                'Expected a bounded object-argument schema')
        stack = [schema]
        while stack:
            item = stack.pop()
            if isinstance(item, dict):
                require(not {'$ref', '$dynamicRef'} & set(item), 'Schema references are not admitted')
                stack.extend(item.values())
            elif isinstance(item, list): stack.extend(item)
        Draft202012Validator.check_schema(schema); schemas[name] = Draft202012Validator(schema)
    messages = case['messages']; pending = {}; seen_calls = set()
    require(type(messages) is list and 1 <= len(messages) <= 8, 'Invalid bounded conversation')
    for message in messages:
        require(type(message) is dict and message.get('role') in ('system', 'user', 'assistant', 'tool'), 'Invalid message role')
        role = message['role']
        if role == 'tool':
            exact(message, ('role', 'content', 'tool_call_id'), 'tool result')
            require(message['tool_call_id'] in pending, 'Result does not match an actual prior supplied call')
            pending.pop(message['tool_call_id'])
        elif role == 'assistant' and 'tool_calls' in message:
            exact(message, ('role', 'content', 'tool_calls'), 'assistant call')
            require(not pending and type(message['tool_calls']) is list and 1 <= len(message['tool_calls']) <= 2,
                    'Unresolved/invalid supplied tool calls')
            for call in message['tool_calls']:
                exact(call, ('id', 'type', 'function'), 'call')
                require(type(call['id']) is str and re.fullmatch('[a-zA-Z0-9_-]{1,80}', call['id'])
                        and call['id'] not in seen_calls and call['type'] == 'function', 'Invalid supplied call ID')
                exact(call['function'], ('name', 'arguments'), 'call function')
                fn = call['function']; require(fn['name'] in schemas and type(fn['arguments']) is dict,
                                             'Supply lossless JSON argument objects, not executable tools')
                schemas[fn['name']].validate(fn['arguments']); pending[call['id']] = fn['name']; seen_calls.add(call['id'])
        else:
            exact(message, ('role', 'content'), 'message'); require(not pending, 'Missing supplied tool result')
        require((type(message['content']) is str and len(message['content']) <= 8000)
                or (role == 'assistant' and 'tool_calls' in message and message['content'] is None), 'Invalid message content')
    require(not pending, 'Generation requires completed supplied tool results')
    # These are counterfactual/development inputs, not evidence of a spontaneous
    # tool episode. That stronger gate remains in the historical episode runner.


def validate_protocol(protocol, lock=None, *, sources=None):
    active_policy(); lock = load_lock() if lock is None else lock
    exact(protocol, ('schema_version', 'id', 'role', 'method', 'parent', 'generation', 'case', 'intervention',
                     'budget', 'design_sha256', 'source_sha256'), 'lab protocol')
    require(type(protocol['schema_version']) is int and protocol['schema_version'] == 1
            and protocol['role'] == 'practice' and protocol['method'] == 'forward-only-fixed-interventions'
            and protocol['parent'] == parent_identity(lock), 'Only original-parent exposed diagnostics are admitted')
    require(type(protocol['id']) is str and re.fullmatch('[a-z0-9_-]{1,80}', protocol['id']), 'Invalid study ID')
    digest(protocol['design_sha256'])
    require(protocol['source_sha256'] == (source_manifest() if sources is None else sources), 'Frozen producer source differs')
    require(canonical(protocol['generation']) == canonical(PROFILE), 'Frozen greedy CPU profile differs')
    validate_case(protocol['case']); validate_intervention(protocol['intervention'])
    budget = protocol['budget']
    exact(budget, ('worker_seconds', 'total_worker_seconds', 'prior_worker_seconds'), 'budget')
    require(type(budget['worker_seconds']) is int and budget['worker_seconds'] == 1800
            and type(budget['total_worker_seconds']) is int and 14400 <= budget['total_worker_seconds'] <= 21600
            and type(budget['prior_worker_seconds']) in (int, float) and math.isfinite(budget['prior_worker_seconds'])
            and budget['prior_worker_seconds'] >= 0
            and budget['prior_worker_seconds'] + len(ARMS) * 1800 <= budget['total_worker_seconds'],
            'All eight arms and prior charges must fit before any model worker')


def prepare(study_id, case, intervention, design, prior_seconds=0):
    value = {'schema_version': 1, 'id': study_id, 'role': 'practice', 'method': 'forward-only-fixed-interventions',
             'parent': parent_identity(load_lock()), 'generation': dict(PROFILE), 'case': case,
             'intervention': intervention, 'budget': {'worker_seconds': 1800, 'total_worker_seconds': 21600,
                                                    'prior_worker_seconds': prior_seconds},
             'design_sha256': sha256(design), 'source_sha256': source_manifest()}
    validate_protocol(value); return value


def render_prompt(tokenizer, protocol):
    case = protocol['case']; options = {'add_generation_prompt': True, 'enable_thinking': False}
    if case['tools']: options['tools'] = case['tools']
    rendered = tokenizer.apply_chat_template(case['messages'], tokenize=False, **options)
    ids = _ids(tokenizer.apply_chat_template(case['messages'], tokenize=True, **options))
    require(ids == tokenizer.encode(rendered, add_special_tokens=False) and 1 <= len(ids) <= 512,
            'Template/token identity or full input budget mismatch; no truncation')
    # Resolve both positions before admitting a generation.
    from wsp_interventions import FixedIntervention
    FixedIntervention(protocol['intervention'], 'base', len(ids))
    return ids, {'case': case, 'rendered': rendered, 'input_ids': ids, 'thinking': False, 'truncated': False,
                 'template_sha256': __import__('hashlib').sha256(tokenizer.chat_template.encode()).hexdigest(),
                 'supplied_tool_episode': bool(case['tools']), 'model_tool_calls_executed': False}


def plan(protocol):
    validate_protocol(protocol)
    return {'study_id': protocol['id'], 'arms': list(ARMS), 'generation_invocations_max': len(ARMS),
            'worst_case_worker_seconds': len(ARMS) * 1800, 'cpu_cores': 1, 'memory_gib': 10, 'gpu': False,
            'intervention': protocol['intervention'], 'training': False, 'weights_edited': False,
            'assessment': protocol['case'].get('assessment'),
            'weight_candidate_eligible': False, 'quality_evaluation': 'NOT_RUN',
            'scope': 'Fixed activation controls only. No reverse-transfer/semantic gate or candidate admission is inferred.',
            'launch': 'python scripts/wsp_lab.py run --protocol FILE --design FILE --baseline SNAPSHOT --deps SITE_PACKAGES --output NEW_DIRECTORY'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('catalog')
    p = sub.add_parser('prepare')
    for key in ('case', 'intervention', 'design', 'output'): p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--id', required=True); p.add_argument('--prior-worker-seconds', type=float, default=0)
    p = sub.add_parser('plan'); p.add_argument('--protocol', type=Path, required=True)
    p = sub.add_parser('run')
    for key in ('protocol', 'design', 'baseline', 'deps', 'output'): p.add_argument('--'+key, type=Path, required=True)
    p = sub.add_parser('compare')
    for key in ('manifest', 'trials'): p.add_argument('--'+key, type=Path, required=True)
    p = sub.add_parser('patch-structure'); p.add_argument('--patch', type=Path, required=True)
    p = sub.add_parser('replay-cycle2'); p.add_argument('--evidence', type=Path, required=True)
    args = parser.parse_args()
    active_policy()
    if args.command == 'catalog': result = read_json(ROOT / 'configs/wsp-lab-sources.json')
    elif args.command == 'prepare':
        result = prepare(args.id, read_json(args.case), read_json(args.intervention), args.design, args.prior_worker_seconds)
        publish_json(result, args.output)
    elif args.command == 'plan': result = plan(read_json(args.protocol))
    elif args.command == 'run':
        command = [sys.executable, str(ROOT / 'ops/wsp_lab54.py')]
        for key in ('protocol', 'design', 'baseline', 'deps', 'output'): command += ['--'+key, str(getattr(args, key))]
        raise SystemExit(subprocess.call(command))
    elif args.command == 'compare':
        from wsp_pairs import paired_summary
        result = paired_summary(read_json(args.manifest), read_json(args.trials))
    elif args.command == 'patch-structure':
        from direct_weights import validate_patch_structure
        validate_patch_structure(load_lock(), read_json(args.patch))
        result = {'numeric_structure_valid': True, 'baseline_authenticated': False,
                  'weight_candidate_eligible': False, 'quality_evaluation': 'NOT_RUN'}
    else:
        from replay_tool_cycle2_evidence import replay
        result = replay(args.evidence)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == '__main__': main()
