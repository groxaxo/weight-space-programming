"""Explicit native profiles; values are from the pinned Qwen3.5-4B model card."""
import math
from coder_common import check

MODEL_CARD = 'https://huggingface.co/Qwen/Qwen3.5-4B/blob/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/README.md'
QWEN35_INSTRUCT = {'temperature': 0.7, 'top_p': 0.8, 'top_k': 20, 'min_p': 0.0,
    'presence_penalty': 1.5, 'frequency_penalty': 0.0, 'repeat_penalty': 1.0, 'seed': 20261106}


def sampling_for_profile(name):
    check(name in ('greedy', 'qwen35-instruct'), 'Unknown native sampling profile')
    return dict(QWEN35_INSTRUCT) if name == 'qwen35-instruct' else None


def request_sampling(identity):
    return dict(identity.get('runtime', {}).get('sampling') or {'temperature': 0})


def effective_sampling(request, declared=None):
    if declared is None:
        temperature=request.get('temperature',0)
        check(type(temperature) in (float,int) and math.isfinite(temperature) and 0<=temperature<=2,
              'Invalid temperature')
        return {'temperature':temperature,'top_k':0,'top_p':0.95 if temperature>0 else 1.0,
            'min_p':0.0,'repeat_penalty':1.0,'presence_penalty':0.0,'frequency_penalty':0.0}
    check(declared==QWEN35_INSTRUCT, 'Unknown declared native sampling parameters')
    for key,expected in declared.items():
        if key in request:
            value=request[key]
            check(type(value) in (int,float) and math.isfinite(value)
                and math.isclose(value,expected,rel_tol=0,abs_tol=1e-8),
                'Request differs from the declared sampling profile: '+key)
    return dict(declared)


def verify_native_sampling(actual, expected):
    check(isinstance(actual,dict), 'Native engine omitted its effective sampling settings')
    for key,value in expected.items():
        observed=actual.get(key)
        check(type(observed) in (int,float) and math.isfinite(observed)
            and math.isclose(observed,value,rel_tol=0,abs_tol=1e-6),
            'Native engine did not apply the declared sampling setting: '+key)


def client_limits(identity):
    context,output=identity['limits']['context'],identity['limits']['output']
    check(type(context) is int and type(output) is int and 0<output<context, 'Invalid advertised client limits')
    return {'context':context,'input':context-output,'output':output}
