"""Policy controls for the separately declared native profile; fixtures are synthetic."""
import copy
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from compare_gguf_coder import compare_native
from coder_common import CoderError


class NativeGateTests(unittest.TestCase):
    def fixture(self,root):
        reports=[];rows=[]
        for label in ['base','candidate-step18']:
            path=root/(label+'.gguf');path.write_bytes(b'synthetic-policy-fixture-'+label.encode())
            h=hashlib.sha256(path.read_bytes()).hexdigest()
            reports.append({'identity':{'base_model':'unit','revision':'a'*40,'quantization':'Q8_0',
                'runtime':{'backend':'llama.cpp'},'artifact_sha256':h},
                'results':[{'id':str(i),'category':'code' if i<8 else 'tools','passed':True} for i in range(14)]})
            rows.append({'profile':label,'output':str(path),'bytes':path.stat().st_size,'sha256':h})
        conversion={'status':'converted_unpromoted','quantization':'Q8_0','model_id':'unit','revision':'a'*40,'conversions':rows}
        validated={'development':{'regressions':[],'improvements':[]},
            'humaneval':{'baseline':130,'candidate':131,'total':164,'aggregate_nonregression':True},
            'real_clients_passed':True,'validation_loss_improved':True}
        return reports,conversion,validated

    def test_saturated_development_requires_an_independent_functional_gain(self):
        with tempfile.TemporaryDirectory() as directory:
            reports,conversion,validated=self.fixture(Path(directory))
            with patch('compare_gguf_coder.acceptance',return_value=validated):
                result=compare_native(*reports,{'passed':130},{'passed':131},{},{},conversion)
                self.assertEqual(result['status'],'accepted_for_local_use')
                result=compare_native(*reports,{'passed':130},{'passed':130},{},{},conversion)
                self.assertEqual(result['status'],'not_promoted')

    def test_development_gain_cannot_promote_a_humaneval_tie(self):
        with tempfile.TemporaryDirectory() as directory:
            reports,conversion,validated=self.fixture(Path(directory))
            validated['development']['improvements']=['fixed development case']
            validated['humaneval'].update(candidate=130,strict_improvement=False)
            with patch('compare_gguf_coder.acceptance',return_value=validated):
                result=compare_native(*reports,{'passed':130},{'passed':130},{},{},conversion)
                self.assertEqual(result['status'],'not_promoted')

    def test_broader_gain_cannot_hide_development_regression_or_quality_floor_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            reports,conversion,validated=self.fixture(Path(directory))
            bad=copy.deepcopy(validated);bad['development']['regressions']=['a previously correct case']
            with patch('compare_gguf_coder.acceptance',return_value=bad):
                self.assertEqual(compare_native(*reports,{'passed':130},{'passed':140},{},{},conversion)['status'],'not_promoted')
            reports[1]['results'][0]['passed']=False;reports[1]['results'][1]['passed']=False
            with patch('compare_gguf_coder.acceptance',return_value=validated):
                self.assertEqual(compare_native(*reports,{'passed':130},{'passed':140},{},{},conversion)['status'],'not_promoted')

    def test_unmerged_lora_is_bound_to_the_same_verified_base_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);reports,conversion,validated=self.fixture(root)
            path=root/'adapter.gguf';path.write_bytes(b'synthetic-unmerged-adapter')
            h=hashlib.sha256(path.read_bytes()).hexdigest()
            conversion['conversions'][1]={'profile':'candidate-lora','kind':'lora','base_profile':'base',
                'output':str(path),'bytes':path.stat().st_size,'sha256':h}
            identity=reports[1]['identity'];identity.update(conversion_profile='candidate-lora',
                lora_artifact_sha256=h,artifact_sha256=reports[0]['identity']['artifact_sha256'])
            with patch('compare_gguf_coder.acceptance',return_value=validated):
                self.assertEqual(compare_native(*reports,{'passed':130},{'passed':131},{},{},conversion)['status'],'accepted_for_local_use')
                identity['artifact_sha256']='0'*64
                with self.assertRaises(CoderError):compare_native(*reports,{'passed':130},{'passed':131},{},{},conversion)


if __name__=='__main__':unittest.main()
