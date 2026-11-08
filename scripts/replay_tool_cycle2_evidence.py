#!/usr/bin/env python3
"""Replay Cycle 2's archived execution/development gate without loading a model.

Use the repository's reviewed source, never import source-text files from the
archive. This is not an independent paired model-quality evaluation and does not
freshly authenticate a multi-GB baseline on the reader's machine.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import json

import direct_tool_specificity as s
from direct_weights import load_lock, read_json
from pipeline import require, sha256


def replay(root):
    lock = load_lock()
    require(sha256(root / 'original/reproduction-gate.json') ==
            '824a514f9a8c9b2944387df1f88da48c6e14ab412ac4b0afdeef72d49489a77f',
            'Original rejected gate differs')
    stage1 = s.load_stage1(root / 'corrected', lock)
    protocol_hash = '628b5674bbd951e42f4c182c8678b3ea1ccdcb126b73efc2e02598f669301364'
    protocol, _ = s.load_protocol(root / 'frozen/configs/direct-tool-specificity-cycle2.protocol.json', protocol_hash, lock)
    require(protocol == s.protocol_document(stage1, lock), 'Recomputed protocol differs')
    conditions_path = root / 'engineering/engineering-02/compiler/conditions.json'
    conditions = read_json(conditions_path)
    require(sha256(conditions_path) == '324a91d0c81e5ee85b969bb9b95dffe559bcef9df2aa9d7668680ce2038af6a8',
            'Frozen conditions differ')
    saved_gate = read_json(root / 'specificity/specificity-gate.json')
    records = []
    for control in s.CONTROLS:
        area = root / 'specificity' / ('control-' + control) / 'worker-output'
        request, report = read_json(area / 'request.json'), read_json(area / 'worker.json')
        require(request['control'] == control, 'Control order differs')
        analysis = s.verify_specificity_evidence(area, report, request, protocol, protocol_hash, conditions, stage1, lock)
        require(report['loaded_parameter_sha256_before'] == stage1['parameters'] == report['loaded_parameter_sha256_after'],
                'Protected parent differs across fresh workers')
        records.append({'control': control, 'worker': report, 'analysis': analysis})
    gate = s.specificity_gate(records)
    require(gate['complete'] is True and gate['status'] == saved_gate['status'], 'Saved complete verdict differs')
    ledger = read_json(root / 'frozen/docs/experiments/direct-tool-finalization-cycle2-20261108/SPECIFICITY-QUALIFICATION-LEDGER.json')
    require(saved_gate['cumulative_worker_seconds'] == ledger['total_worker_seconds'] +
            sum(row['charged_worker_seconds'] for row in saved_gate['costs']), 'Cost arithmetic differs')
    require(saved_gate['cumulative_generation_invocations'] == 13 and saved_gate['cumulative_worker_seconds'] <= 21600,
            'Completed count or cumulative cap differs')
    cn = [record['worker']['results'][0]['output_ids'] for record in records if record['control'] in ('C', 'N')]
    return {**gate, 'recomputed_from_raw_evidence': True, 'stage1_manifest_sha256': stage1['manifest_sha256'],
            'official_generations_total': saved_gate['cumulative_generation_invocations'],
            'charged_worker_seconds': saved_gate['cumulative_worker_seconds'], 'parameter_count_per_worker': 426,
            'completion_and_neutral_output_ids_identical': cn[0] == cn[1],
            'verified_semantic_controls': [{'control': r['control'], 'classification': r['analysis']['classification'],
                                            'passed': r['analysis']['passed'],
                                            'generated_tokens': r['worker']['results'][0]['generated_tokens']}
                                           for r in records],
            'scope': 'Offline execution-integrity/development replay; no model loaded or new inference.',
            'independent_model_quality': 'NOT_RUN'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True, help='Already safely extracted evidence root')
    args = parser.parse_args()
    print(json.dumps(replay(args.evidence), indent=2, sort_keys=True))


if __name__ == '__main__': main()
