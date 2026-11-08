"""CPU-only verification of new direct-worker evidence (schema 2).

This checks execution integrity, never scientific merit or independent model quality.
Archived schema-1 evidence is preserved under its original producer, not upgraded.
"""
from __future__ import annotations

import io
import math
from pathlib import Path
import re
import zipfile

import numpy as np

from direct_weights import ROOT, parent_identity, sha_bytes
from pipeline import require, sha256

VERSION = 2
PRODUCER_FILES = tuple('scripts/' + name + '.py' for name in
                       ('pipeline', 'direct_weights', 'direct_forward', 'direct_probe',
                        'direct_causal', 'direct_evidence', 'direct_tools', 'coder_common',
                        'coder_sampling', 'evaluate_coder')) + ('ops/direct_edit54.py', 'ops/direct_tool_cycle54.py', 'ops/direct_tool_preflight54.py')
# Capture before model execution; later mutations cannot silently change attribution.
SOURCE_AT_IMPORT = {name: sha256(ROOT / name) for name in PRODUCER_FILES}


def expected_parameter_names():
    names = {'model.embed_tokens.weight', 'model.norm.weight'}  # tied output head
    common = ('input_layernorm.weight', 'post_attention_layernorm.weight',
              'mlp.gate_proj.weight', 'mlp.up_proj.weight', 'mlp.down_proj.weight')
    linear = ('A_log', 'conv1d.weight', 'dt_bias', 'in_proj_a.weight', 'in_proj_b.weight',
              'in_proj_qkv.weight', 'in_proj_z.weight', 'norm.weight', 'out_proj.weight')
    full = ('q_proj.weight', 'k_proj.weight', 'v_proj.weight', 'o_proj.weight',
            'q_norm.weight', 'k_norm.weight')
    for layer in range(32):
        prefix = f'model.layers.{layer}.'
        names.update(prefix + suffix for suffix in common)
        attention, suffixes = ('self_attn.', full) if layer % 4 == 3 else ('linear_attn.', linear)
        names.update(prefix + attention + suffix for suffix in suffixes)
    return names


def validate_parameter_map(values):
    require(type(values) is dict and set(values) == expected_parameter_names(),
            'Expected the exact 426 unique official language parameters')
    require(all(type(value) is str and re.fullmatch('[0-9a-f]{64}', value) for value in values.values()),
            'Invalid loaded parameter SHA-256')


def artifact(path):
    return {'path': path.name, 'sha256': sha256(path)}


def bindings(outdir):
    baseline = outdir / 'baseline-verification.json'
    return {'evidence_schema_version': VERSION, 'producer_sha256': dict(SOURCE_AT_IMPORT),
            'baseline_verification': artifact(baseline) if baseline.is_file() else None}


def bound_bytes(root, reference, *, maximum=4 * 1024 * 1024):
    require(type(reference) is dict and set(reference) == {'path', 'sha256'}, 'Missing artifact binding')
    name = reference['path']
    require(type(name) is str and Path(name).name == name and name not in ('', '.', '..'),
            'Unsafe evidence artifact path')
    path = root / name
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= maximum,
            'Missing or oversized evidence artifact: ' + name)
    with path.open('rb') as handle:
        raw = handle.read(maximum + 1)
    require(len(raw) <= maximum and sha_bytes(raw) == reference['sha256'],
            'Evidence artifact hash differs: ' + name)
    return raw


def bound_document(root, reference):
    from direct_weights import _json_bytes
    return _json_bytes(bound_bytes(root, reference))


def numeric_arrays(root, reference, *, applied_layers=()):
    raw = bound_bytes(root, reference, maximum=2 * 1024 * 1024)
    names = {f'layer_{layer}_{kind}_bf16' for layer in range(32) for kind in ('input', 'output')}
    names.update(f'layer_{layer}_applied_output_bf16' for layer in applied_layers)
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        entries = archive.infolist()
        require(len(entries) == len(names) and {entry.filename for entry in entries} == {n + '.npy' for n in names},
                'Duplicate, missing or undeclared observation arrays')
        require(all(entry.compress_type == zipfile.ZIP_STORED and not entry.flag_bits & 1
                    and 0 < entry.file_size <= 20000 for entry in entries)
                and sum(entry.file_size for entry in entries) <= 1024 * 1024, 'Unsafe observation archive')
    result = {}
    with np.load(io.BytesIO(raw), allow_pickle=False, max_header_size=512) as archive:
        for name in names:
            value = archive[name]
            width = 9216 if '_input_' in name else 2560
            require(value.dtype == np.dtype('<u2') and value.shape == (width,), 'Invalid observation dtype/shape')
            require(not np.any((value & 0x7f80) == 0x7f80), 'Non-finite BF16 observation')
            result[name] = value.copy()
    return result


def verify_baseline_proof(root, report, lock):
    baseline = bound_document(root, report.get('baseline_verification'))
    require(baseline.get('status') == 'baseline_verified' and baseline.get('parent') == parent_identity(lock),
            'Missing full official baseline authentication')
    files = baseline.get('files')
    require(type(files) is dict and set(files) == set(lock['model']['files']), 'Baseline proof lacks all locked files')
    for name, spec in lock['model']['files'].items():
        record = files[name]
        require(type(record) is dict and record.get('size') == spec['size'] and record.get('sha256') == spec['sha256'],
                'Baseline proof changed locked file: ' + name)
        identity = record.get('identity')
        require(type(identity) is dict and set(identity) == {'dev', 'ino', 'size', 'mtime_ns', 'ctime_ns'}
                and all(type(v) is int for v in identity.values()) and identity['size'] == spec['size'],
                'Baseline file generation is missing: ' + name)
    require(type(baseline.get('tensors')) is dict and set(baseline['tensors']) == set(lock['permitted_tensors']),
            'Baseline proof lacks permitted tensor checks')
    for name, expected in lock['permitted_tensors'].items():
        require(all(baseline['tensors'][name].get(k) == v for k, v in expected.items()),
                'Baseline tensor identity differs: ' + name)


def verify_common(root, mode, report, *, lock, source_hashes):
    require(report.get('evidence_schema_version') == VERSION, 'Worker evidence schema is missing or obsolete')
    expected = {name: source_hashes.get(name) for name in PRODUCER_FILES}
    require(all(type(value) is str and re.fullmatch('[0-9a-f]{64}', value) for value in expected.values())
            and report.get('producer_sha256') == expected, 'Worker producer differs from frozen source')
    verify_baseline_proof(root, report, lock)
    require(type(report.get('complete')) is bool, 'Worker completion must be explicit')
    if mode == 'verify':
        require(report['complete'] is True, 'Baseline verification is incomplete')
        return
    before = report.get('loaded_parameter_sha256_before')
    validate_parameter_map(before)
    require(before == report.get('loaded_parameter_sha256_after')
            and report.get('parameters_unchanged') is True
            and report.get('parameter_postcheck_complete') is True, 'Incomplete full parameter postcheck')
    loaded = bound_document(root, report.get('loaded_parent_proof'))
    require(loaded.get('loaded_parameter_sha256_before') == before
            and loaded.get('parent') == report['parent']
            and loaded.get('producer_sha256') == expected
            and loaded.get('baseline_verification') == report['baseline_verification']
            and loaded.get('loading_info') == report.get('loading_info'), 'Loaded-parent proof differs from worker')


def verify_observation(root, report, request):
    rows = report.get('results')
    require(type(rows) is list and [row.get('id') for row in rows] == [case['id'] for case in request['cases']]
            and report['complete'] is True, 'Observation case table is incomplete')
    for row in rows:
        raw = bound_document(root, {'path': row['id'] + '-generation.json',
                                    'sha256': row.get('generation_sha256')})
        require(raw == {k: v for k, v in row.items() if k != 'generation_sha256'},
                'Generation row differs from bound artifact')
        ids = row.get('output_ids')
        require(type(ids) is list and all(type(token) is int and token >= 0 for token in ids)
                and len(ids) == row.get('generated_tokens'), 'Invalid generation token counts')
        eos = row.get('eos_token_ids')
        require(type(eos) is list and bool(eos) and all(type(token) is int and token >= 0 for token in eos),
                'Missing generation EOS identity')
        ended = bool(ids) and ids[-1] in eos
        capped = len(ids) == request['generation']['max_new_tokens']
        require(len(ids) <= request['generation']['max_new_tokens']
                and row.get('hit_token_limit') is (capped and not ended)
                and row.get('reached_length_cap') is capped and row.get('censored') is (not ended)
                and row.get('finish_reason') == ('eos' if ended else 'length' if capped else 'unknown'),
                'Generation completion fields differ from raw tokens')
        numeric_arrays(root, {'path': row.get('observation'), 'sha256': row.get('observation_sha256')})


def verify_arms(root, mode, report, request):
    import direct_probe as probe
    import direct_causal as causal
    plans_doc = bound_document(root, report.get('tokenization_proof'))
    plans = plans_doc.get('plans') if mode == 'causal' else plans_doc
    require(type(plans) is dict, 'Missing tokenization plans')
    if mode == 'probe':
        specs = [{'id': name, 'context': name, 'arm': None} for name in probe.required_contexts(request)]
        specs += [{'id': arm['id'], 'context': 'baseline', 'arm': arm} for arm in request['arms']]
    else:
        specs = causal.specifications(request['stage'], request['winner_group'])
    rows = report.get('arms')
    require(type(rows) is list and len(rows) == len(specs)
            and [row.get('id') for row in rows if type(row) is dict] == [spec['id'] for spec in specs],
            'Missing, duplicated, reordered or unexpected requested arms')
    for row, spec in zip(rows, specs):
        require(row.get('status') in ('completed', 'NOT_RUN_budget', 'NOT_RUN_missing_contrast', 'NOT_RUN_replay_mismatch'),
                'Unknown arm completion status')
        if row['status'] != 'completed':
            require(set(row) == {'id', 'status'}, 'Skipped arm contains unexplained evidence')
            continue
        probe.validate_score(row)
        context = spec['context']
        plan = plans.get(context, {})
        require(row.get('context') == context and row.get('decision_position') == plan.get('decision_position')
                and row.get('input_tokens_per_sequence_max') == plan.get('max_forward_input_tokens')
                and all(row['branches'][label]['ids'] == plan.get('candidate_ids', {}).get(label) for label in probe.LABELS),
                'Arm differs from frozen tokenization plan')
        raw = bound_document(root, row.get('score_artifact'))
        if mode == 'probe':
            require(raw == {k: v for k, v in row.items() if k != 'score_artifact'}
                    and row.get('activation_arm') == spec['arm'], 'Probe raw score/arm differs')
            selected = range(spec['arm']['group'] * 8, spec['arm']['group'] * 8 + 8) if spec['arm'] else ()
        else:
            envelope = bound_document(root, row.get('causal_score_artifact'))
            require(envelope == {k: v for k, v in row.items() if k not in ('causal_score_artifact', 'replay')}
                    and row.get('activation_addition') == spec, 'Causal score/specification differs')
            require(all(row.get(k) == v for k, v in raw.items())
                    and row.get('capture_score_sha256') == row['score_artifact']['sha256'],
                    'Causal envelope differs from original capture score')
            selected = ()
            expected_layers = list(range(spec['group'] * 8, spec['group'] * 8 + 8)) if 'group' in spec else []
            stats = row.get('actual_interventions')
            require(type(stats) is list and [v.get('layer') for v in stats] == expected_layers,
                    'Causal intervention support differs')
            for value in stats:
                require(type(value.get('changed_entries')) is int and 0 <= value['changed_entries'] <= 2560
                        and all(type(value.get(k)) in (int, float) and math.isfinite(value[k]) and value[k] >= 0
                                for k in ('delta_l2', 'max_absolute_delta')), 'Invalid causal intervention magnitude')
            if spec['id'] in ('neutral-replay', 'no-op', 'semantic-replay'):
                replay = row.get('replay', {})
                tolerance = 0 if spec['id'] == 'no-op' else 1e-6
                require(replay.get('all_64_input_output_vectors_bit_identical') is True
                        and type(replay.get('max_token_logprob_difference')) in (int, float)
                        and 0 <= replay['max_token_logprob_difference'] <= tolerance,
                        'Missing exact donor/no-op replay proof')
        numeric_arrays(root, {'path': row.get('observation'), 'sha256': row.get('observation_sha256')},
                       applied_layers=selected)
    complete = all(row['status'] == 'completed' for row in rows)
    require(report['complete'] is complete, 'Completion claim differs from requested arms')
    if mode == 'probe':
        contrast = all(row['status'] == 'completed' for row in rows[:len(probe.required_contexts(request))])
        require(report.get('contrast_complete') is contrast and report.get('all_requested_arms_completed') is complete
                and report.get('required_contexts') == list(probe.required_contexts(request)), 'Probe completion summary differs')
        if request['schema_version'] == 2:
            require(report.get('diagnostic_gate') == probe.first_assertion_gate(rows), 'Probe verdict differs from raw scores')
    else:
        require(report.get('diagnostic_gate') == causal.diagnostic_gate(request['stage'], rows, request['winner_group']),
                'Causal verdict differs from raw scores')
