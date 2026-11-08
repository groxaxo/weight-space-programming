"""Paired exposed-development outcomes, exact-output locality and optional KL.

Metrics consume externally attributed/graded data. They do not certify the grader
or its provenance, create private cases, admit a candidate or replace final custody.
"""
from __future__ import annotations
import math
from collections import Counter
from pipeline import require, ValidationError


def validate_assessment(value):
    from coder_common import canonical
    require(type(value) is dict and set(value) == {'kind','expected','family','locality'}
            and value['kind'] in ('exact_text','json_exact')
            and type(value['family']) is str and 1 <= len(value['family']) <= 80
            and type(value['locality']) is bool, 'Invalid frozen diagnostic grader')
    require((value['kind'] == 'exact_text' and type(value['expected']) is str and len(value['expected']) <= 8000)
            or (value['kind'] == 'json_exact' and type(value['expected']) is dict and len(canonical(value['expected'])) <= 16384),
            'Only literal text or exact JSON objects are graded here; executable coding grading remains separate')


def assess_text(text, assessment):
    from coder_common import canonical
    from direct_weights import _json_bytes
    validate_assessment(assessment)
    require(type(text) is str and len(text) <= 65536, 'Invalid bounded model output')
    if assessment['kind'] == 'exact_text': return text.strip() == assessment['expected'].strip()
    try: return canonical(_json_bytes(text.encode())) == canonical(assessment['expected'])
    except (ValueError,TypeError,ValidationError): return False


def assess_records(protocol, records):
    """Grade verified raw development outputs against the pre-generation protocol."""
    import hashlib
    from coder_common import canonical
    assessment=protocol['case'].get('assessment')
    if assessment is None:
        return {'status':'NOT_RUN_no_frozen_assessment','quality_evaluation':'NOT_RUN'}
    validate_assessment(assessment)
    by_arm={r['arm']:r for r in records}
    if not {'base','treatment'} <= set(by_arm):
        return {'status':'NOT_RUN_missing_pair','quality_evaluation':'NOT_RUN'}
    def sha(value):return hashlib.sha256(canonical(value).encode()).hexdigest()
    base=by_arm['base'];row=base['results'][0]
    runtime=sha({'runtime':base['runtime'],'generation':protocol['generation'],'parent':protocol['parent'],
                 'producers':protocol['source_sha256'],
                 'cpu_reference':base.get('loading_info',{}).get('cpu_reference_kernels')})
    grader=sha({'assessment':assessment,'source':protocol['source_sha256']['scripts/wsp_pairs.py']})
    case={'id':protocol['case']['id'],'family':assessment['family'],'locality':assessment['locality'],
          'input_sha256':sha(row['input_ids'])}
    manifest={'schema_version':1,'role':'practice','runtime_sha256':runtime,'grader_sha256':grader,'cases':[case]}
    trials=[]
    for arm,label in (('base','base'),('treatment','candidate')):
        report=by_arm[arm];r=report['results'][0]
        require(report['runtime']==base['runtime'] and r['input_ids']==row['input_ids']
                and report.get('loading_info',{}).get('cpu_reference_kernels')==base.get('loading_info',{}).get('cpu_reference_kernels'),
                'Diagnostic pair runtime/input differs')
        trials.append({'id':case['id'],'arm':label,'input_sha256':case['input_sha256'],
            'runtime_sha256':runtime,'grader_sha256':grader,'passed':bool(report['complete'] and not r['censored']
                and assess_text(r['text'],assessment)), 'censored':r['censored'],'complete':report['complete'],
            'output_sha256':hashlib.sha256(r['text_with_special_tokens'].encode()).hexdigest(),
            'request_sha256':sha({'protocol_sha256':report['protocol_sha256'],'arm':arm,'input_ids':r['input_ids']})})
    return {**paired_summary(manifest,trials),'status':'development_pair_graded',
            'candidate_kind':'activation_treatment_only','grader_kind':assessment['kind'],
            'scope':'Frozen literal diagnostic outcome, not executable coding quality or independent acceptance.'}


def paired_summary(manifest, rows):
    require(type(manifest) is dict and set(manifest) == {'schema_version', 'role', 'runtime_sha256', 'grader_sha256', 'cases'}
            and type(manifest['schema_version']) is int and manifest['schema_version'] == 1
            and manifest['role'] == 'practice', 'Only frozen exposed-development manifests are accepted')
    def digest(value):
        return type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value)
    require(digest(manifest['runtime_sha256']) and digest(manifest['grader_sha256']), 'Invalid runtime/grader binding')
    cases = manifest['cases']
    require(type(cases) is list and 1 <= len(cases) <= 10000, 'Invalid fixed paired denominator')
    expected = {}
    for case in cases:
        require(type(case) is dict and set(case) == {'id', 'family', 'input_sha256', 'locality'}
                and type(case['id']) is str and bool(case['id']) and case['id'] not in expected
                and type(case['family']) is str and bool(case['family']) and digest(case['input_sha256'])
                and type(case['locality']) is bool, 'Invalid/duplicate frozen case')
        expected[case['id']] = case
    require(type(rows) is list and len(rows) <= 2 * len(cases), 'Invalid trial rows')
    pairs = {}
    for row in rows:
        require(type(row) is dict and set(row) == {'id', 'arm', 'input_sha256', 'runtime_sha256', 'grader_sha256',
                'passed', 'censored', 'complete', 'output_sha256', 'request_sha256'}, 'Unexpected trial fields')
        require(row['id'] in expected and row['arm'] in ('base', 'candidate'), 'Unknown paired identity')
        key = (row['id'], row['arm']); require(key not in pairs, 'Duplicate trial')
        require(row['input_sha256'] == expected[row['id']]['input_sha256']
                and row['runtime_sha256'] == manifest['runtime_sha256']
                and row['grader_sha256'] == manifest['grader_sha256']
                and digest(row['output_sha256']) and digest(row['request_sha256'])
                and all(type(row[k]) is bool for k in ('passed', 'censored', 'complete')), 'Unattributed paired trial')
        require(not row['passed'] or (row['complete'] and not row['censored']), 'Censored/incomplete trial cannot pass')
        pairs[key] = row
    rescues = regressions = base_passes = candidate_passes = unchanged = locality_n = 0
    families = Counter(); missing = []
    for case in cases:
        a, b = pairs.get((case['id'], 'base')), pairs.get((case['id'], 'candidate'))
        if a is None or b is None: missing.append(case['id'])
        pa, pb = bool(a and a['passed']), bool(b and b['passed'])
        base_passes += pa; candidate_passes += pb
        # A missing arm is charged as failure but never labelled an observed rescue.
        if a is not None and b is not None:
            rescues += not pa and pb; regressions += pa and not pb
            families[case['family']] += int(pb) - int(pa)
        if case['locality']:
            locality_n += 1
            unchanged += bool(a and b and a['complete'] and b['complete'] and not a['censored'] and not b['censored']
                              and a['output_sha256'] == b['output_sha256'])
    complete = not missing and all(r['complete'] and not r['censored'] for r in rows)
    return {'role': 'practice', 'denominator': len(cases), 'complete': complete, 'missing_pairs': missing,
            'base_passes': base_passes, 'candidate_passes': candidate_passes, 'rescues': rescues, 'regressions': regressions,
            'net_gain_pp': 100 * (candidate_passes - base_passes) / len(cases), 'family_net': dict(sorted(families.items())),
            'locality_denominator': locality_n, 'byte_identical_outputs': unchanged,
            'byte_identical_rate': unchanged / locality_n if locality_n else None,
            'quality_evaluation': 'NOT_RUN', 'weight_candidate_eligible': False,
            'scope': 'Declared attributed development-grade aggregation; grader authenticity requires its own evidence.'}


def token_kl(base_logprobs, candidate_logprobs):
    """Mean KL(base || candidate), requiring full normalized matched vocabulary rows."""
    import numpy as np
    a, b = np.asarray(base_logprobs, dtype=np.float64), np.asarray(candidate_logprobs, dtype=np.float64)
    require(a.ndim == 2 and a.shape == b.shape and 1 <= a.shape[0] <= 4096 and 2 <= a.shape[1] <= 300000
            and a.size <= 4_000_000 and np.isfinite(a).all() and np.isfinite(b).all(), 'Invalid bounded KL arrays')
    require(np.all(a <= 0) and np.all(b <= 0)
            and np.allclose(np.exp(a).sum(-1), 1, rtol=1e-6, atol=1e-8)
            and np.allclose(np.exp(b).sum(-1), 1, rtol=1e-6, atol=1e-8), 'KL requires full normalized vocabulary distributions')
    value = float(np.mean(np.sum(np.exp(a) * (a-b), axis=-1)))
    require(math.isfinite(value) and value >= -1e-10, 'Invalid KL result')
    return max(0.0, value)
