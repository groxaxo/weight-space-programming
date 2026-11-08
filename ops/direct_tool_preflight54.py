#!/usr/bin/env python3
"""Admitted tokenizer/parser preflight; zero model generations or weight edits."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from ops import direct_edit54 as host
import direct_tools as tools
from direct_weights import load_lock, publish_json, verify_baseline, read_authenticated_json
from pipeline import require, sha256


def inside():
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only preflight required')
    from transformers import AutoTokenizer
    lock = load_lock()
    verified = verify_baseline(lock, Path('/model'))
    publish_json(verified, Path('/out/baseline-verification.json'))
    tokenizer = AutoTokenizer.from_pretrained('/model', local_files_only=True, trust_remote_code=False)
    config = read_authenticated_json(verified['files']['config.json'])
    model_eos = config.get('text_config', config).get('eos_token_id')
    effective_eos = tools.forward.chat_stop_ids(SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=model_eos)), tokenizer)
    protocol = host.read_json(tools.PROTOCOL_FILE)
    require(effective_eos == protocol['stopping']['expected_eos_token_ids'], 'Authenticated EOS union differs from protocol')
    template_sha = tools.sha_bytes(tokenizer.chat_template.encode())
    require(template_sha == lock['model']['files']['chat_template.jinja']['sha256'], 'Tokenizer template differs from parent')
    values = {}
    synthetic = tools.analyze({'censored': False, 'text_with_special_tokens':
        '<tool_call>\n<function=run_tests>\n<parameter=target>\nparser\n</parameter>\n</function>\n</tool_call><|im_end|>'}, 'initial')
    require(synthetic['classification'] == 'expected_initial_call' and synthetic['parser_agreement'], 'Parser self-test failed')
    for phase in ('initial', 'continuation'):
        request = {'phase': phase}
        ids, proof = tools.prepare_prompt(tokenizer, request, {'analysis': synthetic} if phase == 'continuation' else None)
        require(ids.shape[1] <= tools.PROFILE['max_input_tokens'], 'Fixed full transcript exceeds declared context')
        require('<tool_response>' in proof['rendered'] if phase == 'continuation' else '<tools>' in proof['rendered'],
                'Template did not render expected tool schema/result delimiters')
        publish_json(proof, Path('/out') / (phase + '-prompt.json'))
        values[phase] = {'input_tokens': ids.shape[1], 'prompt': tools.evidence.artifact(Path('/out') / (phase + '-prompt.json')),
                        'call_id_tokenization_invariant': proof['harness_call_id_tokenization_invariant']}
    result = {'status': 'template_parser_preflight_passed', 'generation_invocations': 0,
              'model_loaded': False, 'training': False, 'weights_edited': False, 'quality_evaluation': 'NOT_RUN',
              'synthetic_call_only': True, 'template_sha256': template_sha,
              'effective_eos_token_ids': effective_eos, 'model_config_eos': model_eos,
              'tokenizer_eos': tokenizer.eos_token_id, 'first_eos_must_be_terminal': True,
              'protocol_sha256': sha256(tools.PROTOCOL_FILE), 'fixture_sha256': tools.FIXTURE_SHA256,
              'source_sha256': tools.evidence.SOURCE_AT_IMPORT, 'contexts': values}
    publish_json(result, Path('/out/preflight.json'))
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inside', action='store_true')
    for name in ('baseline', 'deps', 'output'): parser.add_argument('--' + name, type=Path)
    args = parser.parse_args()
    if args.inside:
        inside(); return
    require(os.uname().nodename == '4790', 'Only the .54 admission profile is supported')
    require(all((args.baseline, args.deps, args.output)), 'baseline, deps and output are required')
    require(not os.path.lexists(args.output), 'Refusing existing preflight evidence')
    args.output.mkdir(mode=0o700, parents=True)
    with host.WorkerLease(args.output) as lease:
        state = host.resources(args.output); publish_json(state, args.output / 'resources-before.json')
        host.admit(state, 2 * host.GIB)
        source = args.output / 'source'; hashes = host.freeze_source(source)
        request = args.output / 'request.json'; publish_json({'mode': 'tokenizer-only-preflight', 'generations': 0}, request)
        result_dir = args.output / 'worker-output'; result_dir.mkdir()
        name = 'wsp-direct-tool-preflight-' + args.output.name
        command = host.container_command(name, source.resolve(), args.deps.resolve(strict=True),
            host.mounted_files(load_lock(), args.baseline), result_dir.resolve(), request.resolve(),
            2 * host.GIB, tools.MODE)
        command = command[:command.index(host.IMAGE) + 1] + ['python', '-u', 'ops/direct_tool_preflight54.py', '--inside']
        publish_json({'command': command, 'source_sha256': hashes, 'generation_invocations': 0}, args.output / 'invocation.json')
        lease.launching(name, 2 * host.GIB)
        try:
            with (args.output / 'console.log').open('x') as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=900)
        except BaseException:
            subprocess.run(['docker', 'stop', '--time', '10', name], capture_output=True, timeout=30)
            lease.confirm_exit(); raise
        publish_json({'containers': lease.confirm_exit()}, args.output / 'container-inspect.json')
        publish_json({'exit_code': result.returncode, 'resources_after': host.resources(args.output),
                      'artifacts_retained': True}, args.output / 'execution.json')
        print((args.output / 'console.log').read_text())
        require(result.returncode == 0, 'Preflight failed; preserve all evidence')


if __name__ == '__main__': main()
