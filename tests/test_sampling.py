"""Prevent declared Qwen sampling/limits from being dropped along the client pipeline."""
import copy
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from coder_common import CoderError
from coder_sampling import (QWEN35_INSTRUCT, client_limits, effective_sampling,
    request_sampling, sampling_for_profile, verify_native_sampling)
from ops.replay_he9 import opencode_config, LIMITS
import test_gguf_backend as control


class SamplingTests(unittest.TestCase):
    def test_declared_profile_is_a_copy_and_cannot_silently_change_on_a_request(self):
        profile=sampling_for_profile('qwen35-instruct')
        self.assertEqual(effective_sampling({},profile),QWEN35_INSTRUCT)
        for key,value in [('temperature',0),('top_k',0),('presence_penalty',0),('seed',17),('top_p',True)]:
            with self.subTest(key=key),self.assertRaises(CoderError):effective_sampling({key:value},profile)
        profile['temperature']=0
        self.assertEqual(QWEN35_INSTRUCT['temperature'],0.7)
        with self.assertRaises(CoderError):effective_sampling({},profile)
        self.assertIsNone(sampling_for_profile('greedy'))

    def test_engine_report_must_confirm_all_effective_settings(self):
        verify_native_sampling(QWEN35_INSTRUCT,QWEN35_INSTRUCT)
        for key in QWEN35_INSTRUCT:
            dropped=dict(QWEN35_INSTRUCT);dropped.pop(key)
            with self.subTest(key=key),self.assertRaises(CoderError):verify_native_sampling(dropped,QWEN35_INSTRUCT)
        rounded={**QWEN35_INSTRUCT,'temperature':0.699999988079071}
        verify_native_sampling(rounded,QWEN35_INSTRUCT)

    def test_client_limits_and_sampling_follow_the_advertised_profile(self):
        identity={'limits':{'context':262144,'output':8192},'runtime':{'sampling':QWEN35_INSTRUCT}}
        self.assertEqual(client_limits(identity),{'context':262144,'input':253952,'output':8192})
        self.assertEqual(request_sampling(identity),QWEN35_INSTRUCT)
        self.assertEqual(request_sampling({}),{'temperature':0})
        config=opencode_config('unit')
        self.assertEqual(config['agent']['build']['temperature'],0.7)
        self.assertEqual(LIMITS['presence_penalty'],1.5)

    def test_legacy_greedy_defaults_remain_available(self):
        legacy=effective_sampling({'temperature':0})
        self.assertEqual((legacy['temperature'],legacy['top_k'],legacy['presence_penalty']),(0,0,0))
        self.assertEqual(effective_sampling({'temperature':0.6})['top_p'],0.95)
        with self.assertRaises(CoderError):effective_sampling({'temperature':float('nan')})

    def test_native_engine_cannot_ignore_presence_penalty_but_claim_the_same_profile(self):
        response=control.GgufBackendTests().result(generation_settings={**QWEN35_INSTRUCT,'presence_penalty':0})
        with control.native_control(response) as (url,_):
            backend=control.GgufBackendTests().backend(url)
            backend.identity['runtime']={'sampling':QWEN35_INSTRUCT}
            request={**control.GgufBackendTests().request(),**QWEN35_INSTRUCT}
            with self.assertRaisesRegex(CoderError,'presence_penalty'):backend.complete(request)


if __name__=='__main__':unittest.main()
