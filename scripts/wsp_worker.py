#!/usr/bin/env python3
"""One admitted fixed-intervention forward; never a trainer or final evaluator."""
from __future__ import annotations
import argparse
import io
import json
import os
from pathlib import Path
import signal
import sys
import time
import zipfile
import numpy as np

import direct_evidence as evidence
import direct_forward as forward
from direct_weights import load_lock, parent_identity, publish_json, read_bound_json, verify_baseline, assert_baseline_unchanged
from pipeline import require, sha256
import wsp_lab as lab
from wsp_interventions import ARMS, WIDTHS, FixedIntervention, apply_words

MODE = 'wsp-fixed-forward'


def read_pair(root, reference, spec):
    raw = evidence.bound_bytes(root, reference, maximum=128 * 1024)
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        entries = archive.infolist()
        require(len(entries) == 2 and {x.filename for x in entries} == {'before_bf16.npy', 'after_bf16.npy'}
                and all(x.compress_type == zipfile.ZIP_STORED and not x.flag_bits & 1 and x.file_size <= 20000 for x in entries),
                'Unsafe or incomplete intervention archive')
    with np.load(io.BytesIO(raw), allow_pickle=False, max_header_size=512) as archive:
        arrays = {name: archive[name].copy() for name in ('before_bf16', 'after_bf16')}
    require(all(a.dtype == np.dtype('<u2') and a.shape == (WIDTHS[spec['location']],)
                and not np.any((a & 0x7f80) == 0x7f80) for a in arrays.values()), 'Invalid intervention words')
    return arrays


def observe(protocol, arm, lock, verified, root, protocol_hash):
    import torch
    import transformers
    started = time.monotonic(); model = None; before = None; engine = None; hooks = []; arrays = {}
    report = {'mode': MODE, 'schema_version': 1, 'role': 'practice', 'arm': arm,
              'parent': parent_identity(lock), 'protocol_sha256': protocol_hash,
              'request_sha256': protocol_hash, 'design_sha256': protocol['design_sha256'],
              'lab_producer_sha256': dict(protocol['source_sha256']), 'training': False,
              'weights_edited': False, 'patch_applied': False, 'quality_improvement_claimed': False,
              'weight_candidate_eligible': False, 'quality_evaluation': 'NOT_RUN', 'complete': False,
              'parameter_postcheck_complete': False, 'results': [], 'generation': lab.PROFILE,
              'runtime': {'python': sys.version, 'torch': torch.__version__, 'transformers': transformers.__version__,
                          'numpy': np.__version__}, **evidence.bindings(root)}
    error = None
    try:
        model, tokenizer, loading, before = forward.load_text_model(lock, verified)
        evidence.validate_parameter_map(before)
        report.update(loading_info=loading, loaded_parameter_sha256_before=before)
        publish_json({**report}, root / 'loaded-parent-verification.json')
        report['loaded_parent_proof'] = evidence.artifact(root / 'loaded-parent-verification.json')
        ids, proof = lab.render_prompt(tokenizer, protocol)
        require(proof['template_sha256'] == lock['model']['files']['chat_template.jinja']['sha256'], 'Wrong official template')
        name = 'lab-' + arm
        publish_json(proof, root / (name + '-prompt.json'))
        tensor = torch.tensor([ids], dtype=torch.long)
        require(len(model.model.layers) == 32, 'Wrong language architecture')
        engine = FixedIntervention(protocol['intervention'], arm, len(ids)).install(model)
        def capture(layer):
            def hook(_module, args, output):
                key = f'layer_{layer}_input_bf16'
                if key in arrays: return
                require(not torch.is_grad_enabled(), 'Observer requires inference mode')
                for kind, value, width in (('input', args[0], 9216), ('output', output, 2560)):
                    require(value.dtype == torch.bfloat16 and tuple(value.shape) == (1, len(ids), width), 'Unexpected MLP shape/dtype')
                    arrays[f'layer_{layer}_{kind}_bf16'] = value[0, -1].detach().contiguous().view(torch.uint16).numpy().copy()
            return hook
        for layer, block in enumerate(model.model.layers): hooks.append(block.mlp.down_proj.register_forward_hook(capture(layer)))
        torch.manual_seed(lab.PROFILE['seed'])
        eos = forward.chat_stop_ids(model, tokenizer)
        require(eos == [248044, 248046], 'Authenticated model/tokenizer stop union differs')
        arguments = {'do_sample': False, 'max_new_tokens': 96, 'use_cache': True, 'logits_to_keep': 1,
                     'pad_token_id': tokenizer.eos_token_id, 'eos_token_id': eos}
        publish_json(arguments, root / 'generation-arguments.json')
        report['generation_arguments_proof'] = evidence.artifact(root / 'generation-arguments.json')
        with torch.inference_mode(), (root / (name + '-token-journal.jsonl')).open('x') as journal:
            output = model.generate(input_ids=tensor, attention_mask=torch.ones_like(tensor),
                                    streamer=forward.TokenJournal(journal), **arguments)
        require(engine.telemetry is not None, 'Declared intervention site was not observed')
        with (root / 'intervention.npz').open('xb') as handle:
            np.savez(handle, before_bf16=engine.before, after_bf16=engine.after)
        report['intervention_proof'] = evidence.artifact(root / 'intervention.npz')
        report['intervention'] = engine.telemetry
        with (root / (name + '-prefill.npz')).open('xb') as handle: np.savez(handle, **arrays)
        observation = evidence.artifact(root / (name + '-prefill.npz'))
        evidence.numeric_arrays(root, observation)
        generated = output[0, len(ids):].tolist()
        row = {'id': name, 'input_ids': ids, 'input_tokens': len(ids), 'output_ids': generated,
               'generated_tokens': len(generated), 'text': tokenizer.decode(generated, skip_special_tokens=True),
               'text_with_special_tokens': tokenizer.decode(generated, skip_special_tokens=False),
               'prompt_proof': evidence.artifact(root / (name + '-prompt.json')),
               'observation': observation['path'], 'observation_sha256': observation['sha256'],
               **forward.completion_status(generated, 96, model, tokenizer)}
        publish_json(row, root / (name + '-generation.json')); row['generation_sha256'] = sha256(root / (name + '-generation.json'))
        report['results'] = [row]; report['complete'] = True
    except BaseException as caught:
        error = caught; report['error'] = type(caught).__name__ + ': ' + str(caught)
    finally:
        signal.alarm(450)
        for hook in hooks:
            try: hook.remove()
            except BaseException as caught:
                report['hook_cleanup_error'] = type(caught).__name__ + ': ' + str(caught)
                if error is None: error = caught
        if engine is not None:
            try: engine.close()
            except BaseException as caught:
                report['hook_cleanup_error'] = type(caught).__name__ + ': ' + str(caught)
                if error is None: error = caught
        try:
            if arrays and not report.get('results'):
                with (root / 'partial-prefill.npz').open('xb') as handle: np.savez(handle, **arrays)
        except BaseException as caught:
            report['partial_save_error'] = type(caught).__name__ + ': ' + str(caught)
            if error is None: error = caught
        try:
            if model is not None and before is not None:
                after = {name: forward.parameter_sha256(model.get_parameter(name)) for name in before}
                report['loaded_parameter_sha256_after'] = after; evidence.validate_parameter_map(after)
                require(before == after, 'Model parameters changed during fixed activation intervention')
                report['parameters_unchanged'] = True
                assert_baseline_unchanged(verified); report['parameter_postcheck_complete'] = True
        except BaseException as caught:
            report['postcheck_error'] = type(caught).__name__ + ': ' + str(caught)
            if error is None: error = caught
        if error is not None: report['complete'] = False
        report['seconds'] = time.monotonic() - started
        publish_json(report, root / 'observation-result.json')
    if error is not None: raise error
    return report


def verify_result(root, protocol, protocol_hash, arm):
    from direct_weights import read_json
    lab.validate_protocol(protocol)
    report = read_json(root / 'worker.json')
    require(sha256(root / 'protocol.json') == protocol_hash
            and report.get('protocol_sha256') == protocol_hash and report.get('request_sha256') == protocol_hash
            and report.get('arm') == arm and report.get('mode') == MODE and report.get('role') == 'practice'
            and report.get('design_sha256') == protocol['design_sha256']
            and report.get('lab_producer_sha256') == protocol['source_sha256']
            and report.get('training') is False and report.get('weights_edited') is False
            and report.get('weight_candidate_eligible') is False and report.get('quality_evaluation') == 'NOT_RUN', 'Worker scope differs')
    lock = load_lock()
    require(report.get('parent') == parent_identity(lock) and report.get('patch_applied') is False
            and report.get('quality_improvement_claimed') is False, 'Wrong parent or quality scope')
    evidence.verify_common(root, MODE, report, lock=lock, source_hashes=protocol['source_sha256'])
    evidence.verify_observation(root, report, {'generation': lab.PROFILE, 'cases': [{'id': 'lab-'+arm}]})
    row = report['results'][0]; proof = evidence.bound_document(root, row['prompt_proof'])
    require(proof['case'] == protocol['case'] and proof['input_ids'] == row['input_ids'] and proof['truncated'] is False
            and proof['template_sha256'] == lock['model']['files']['chat_template.jinja']['sha256'], 'Prompt differs')
    args = evidence.bound_document(root, report['generation_arguments_proof'])
    require(args == {'do_sample': False, 'max_new_tokens': 96, 'use_cache': True, 'logits_to_keep': 1,
                     'pad_token_id': 248046, 'eos_token_id': [248044, 248046]}, 'Effective generation profile differs')
    vectors = read_pair(root, report['intervention_proof'], protocol['intervention'])
    after, stats = apply_words(vectors['before_bf16'], protocol['intervention'], arm)
    site = FixedIntervention(protocol['intervention'], arm, row['input_tokens'])
    require(np.array_equal(after, vectors['after_bf16'])
            and report['intervention'] == {**stats, 'layer': site.layer, 'location': site.spec['location'], 'position': site.position},
            'Applied intervention differs from fixed BF16 arithmetic')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('protocol', 'design', 'output'): parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--protocol-sha256', required=True); parser.add_argument('--arm', choices=ARMS, required=True)
    args = parser.parse_args()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not os.path.lexists(args.output), 'Fresh CPU worker required')
    def expired(_sig, _frame): raise TimeoutError('Fixed-forward active budget exceeded')
    signal.signal(signal.SIGALRM, expired); signal.alarm(1250)
    report = {'complete': False, 'training': False, 'weights_edited': False, 'quality_evaluation': 'NOT_RUN'}
    try:
        protocol, actual, raw = read_bound_json(args.protocol)
        require(actual == args.protocol_sha256 and sha256(args.design) == protocol['design_sha256'], 'Host-pinned inputs differ')
        lab.validate_protocol(protocol)
        with (args.output.parent / 'protocol.json').open('xb') as handle: handle.write(raw)
        lock = load_lock(); verified = verify_baseline(lock, Path('/model'))
        publish_json(verified, args.output.parent / 'baseline-verification.json')
        report = observe(protocol, args.arm, lock, verified, args.output.parent, actual)
    except BaseException as error:
        path = args.output.parent / 'observation-result.json'
        if path.is_file(): report.update(lab.read_json(path))
        report.update(complete=False, error=type(error).__name__ + ': ' + str(error)); raise
    finally:
        publish_json(report, args.output); signal.alarm(0)


if __name__ == '__main__': main()
