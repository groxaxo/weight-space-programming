"""Admitted engineering tests only: no official Qwen checkpoint, data or trainer."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'tests')]
import wsp_lab as lab
import wsp_worker as worker
import wsp_interventions as iv
from ops import wsp_lab54 as host
from direct_weights import load_lock,publish_json
from pipeline import ValidationError,sha256
import test_wsp_lab as fixture
from direct_evidence_fixtures import make_worker

ENABLED=os.environ.get('WSP_TEST_WSP_TORCH')=='1' and importlib.util.find_spec('torch') and importlib.util.find_spec('transformers')


@unittest.skipUnless(ENABLED,'Explicit bounded synthetic Torch runtime required')
class UnifiedWorkerRuntime(unittest.TestCase):
    def setUp(self):
        import torch
        torch.set_num_threads(1)
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)

    def execute(self,arm='treatment',location='output',failure=None):
        import torch
        protocol=fixture.protocol();protocol['intervention']=fixture.intervention(location)
        area=self.root/(arm+'-'+location+('-'+failure if failure else ''));area.mkdir()
        trusted=self.root/('trusted-'+area.name)
        make_worker(trusted,'verify')
        (area/'baseline-verification.json').write_bytes((trusted/'baseline-verification.json').read_bytes())
        publish_json(protocol,area/'protocol.json');digest=sha256(area/'protocol.json')
        parameter=torch.tensor([.125],dtype=torch.bfloat16)
        before={name:worker.forward.parameter_sha256(parameter) for name in lab.evidence.expected_parameter_names()}
        accesses=[]
        class Down(torch.nn.Module):
            def forward(self,value):return value[...,:2560].clone()
        blocks=[SimpleNamespace(mlp=SimpleNamespace(down_proj=Down())) for _ in range(32)]
        class Model:
            model=SimpleNamespace(layers=blocks)
            generation_config=SimpleNamespace(eos_token_id=248044)
            def get_parameter(self,name):accesses.append(name);return parameter
            def generate(self,input_ids,**kwargs):
                for i,block in enumerate(blocks):
                    values=torch.full((1,3,9216),float(i+1),dtype=torch.bfloat16)
                    block.mlp.down_proj(values)
                    if i==1 and failure in ('generation','partial-save'):raise TimeoutError('synthetic generation interrupted')
                    # Cached forwards must not overwrite the first-prefill record.
                    block.mlp.down_proj(torch.full((1,1,9216),-7.,dtype=torch.bfloat16))
                return torch.tensor([[1,2,3,9,248046]])
        model=Model();tokenizer=SimpleNamespace(eos_token_id=248046,decode=lambda *a,**k:'synthetic fixture')
        proof={'case':protocol['case'],'rendered':'synthetic fixture','input_ids':[1,2,3],
               'thinking':False,'truncated':False,'template_sha256':load_lock()['model']['files']['chat_template.jinja']['sha256'],
               'supplied_tool_episode':False,'model_tool_calls_executed':False}
        def prepare(*args):
            if failure=='preparation':raise TimeoutError('synthetic preparation interrupted')
            return [1,2,3],proof
        loading={'official_dtype_restorations':[],'cpu_reference_kernels':{'kind':'synthetic-engineering-only'}}
        real_close=worker.FixedIntervention.close
        def close(engine):
            real_close(engine)
            if failure=='cleanup':raise RuntimeError('synthetic cleanup error')
        with mock.patch.object(worker.forward,'load_text_model',return_value=(model,tokenizer,loading,before)), \
             mock.patch.object(lab,'render_prompt',side_effect=prepare), \
             mock.patch.object(worker,'assert_baseline_unchanged'),mock.patch.object(worker.signal,'alarm'), \
             mock.patch.object(worker.FixedIntervention,'close',new=close):
            if failure=='partial-save':
                with mock.patch.object(worker.np,'savez',side_effect=OSError('synthetic disk error')):
                    with self.assertRaises(TimeoutError):worker.observe(protocol,arm,load_lock(),{},area,digest)
                report=json.loads((area/'observation-result.json').read_text())
            elif failure:
                with self.assertRaises(RuntimeError if failure=='cleanup' else TimeoutError):
                    worker.observe(protocol,arm,load_lock(),{},area,digest)
                report=json.loads((area/'observation-result.json').read_text())
            else:
                report=worker.observe(protocol,arm,load_lock(),{},area,digest)
                publish_json(report,area/'worker.json')
                worker.verify_result(area,protocol,digest,arm)
        self.assertEqual(set(accesses),lab.evidence.expected_parameter_names())
        self.assertEqual(len(accesses),426)
        self.assertTrue(report['parameters_unchanged']);self.assertTrue(report['parameter_postcheck_complete'])
        self.assertTrue(all(not b.mlp.down_proj._forward_hooks and not b.mlp.down_proj._forward_pre_hooks for b in blocks))
        return protocol,report,area

    def test_eight_arms_roundtrip_evidence_and_frozen_bf16_controls(self):
        records=[]
        for arm in iv.ARMS:
            protocol,report,area=self.execute(arm);records.append(report)
            self.assertTrue(report['complete']);self.assertEqual(report['results'][0]['output_ids'],[9,248046])
        result=host.summarize(protocol,records)
        self.assertEqual(result['status'],'fixed_activation_records_complete')
        self.assertFalse(result['scientific_mechanism_accepted']);self.assertFalse(result['weight_candidate_eligible'])
        self.assertEqual(result['quality_evaluation'],'NOT_RUN')
        self.assertEqual(result['development_metrics']['base_passes'],1)
        self.assertEqual(result['development_metrics']['candidate_passes'],1)
        self.assertEqual(result['development_metrics']['candidate_kind'],'activation_treatment_only')

    def test_input_and_output_hooks_apply_only_to_declared_prefill_position(self):
        for location in ('input','output'):
            protocol,report,area=self.execute('treatment',location)
            vectors=worker.read_pair(area,report['intervention_proof'],protocol['intervention'])
            self.assertEqual(vectors['after_bf16'][[1,3]].tolist(),[0,0])
            self.assertEqual(report['intervention']['position'],2)
            self.assertEqual(report['intervention']['layer'],0)

    def test_preparation_generation_and_partial_save_errors_keep_all_426_postchecks(self):
        for failure in ('preparation','generation','partial-save','cleanup'):
            _,report,_=self.execute(failure=failure)
            self.assertFalse(report['complete'])
            if failure=='partial-save':self.assertIn('partial_save_error',report)

    def test_changed_numeric_archive_cannot_be_accepted_by_host(self):
        protocol,report,area=self.execute('base')
        with (area/'intervention.npz').open('ab') as handle:handle.write(b'tampered')
        with self.assertRaisesRegex(ValidationError,'hash differs'):worker.verify_result(area,protocol,sha256(area/'protocol.json'),'base')


@unittest.skipUnless(ENABLED,'Explicit bounded synthetic Torch runtime required')
class ActualQwenModuleRuntime(unittest.TestCase):
    def test_real_qwen35_mlp_noop_and_ablation_never_modify_parameters(self):
        import torch
        from transformers import Qwen3_5TextConfig,Qwen3_5ForCausalLM
        torch.set_num_threads(1);torch.manual_seed(20261109)
        # One random attention layer with the actual MLP widths. This is an
        # engineering model, not the official 4B checkpoint or a quality trial.
        config=Qwen3_5TextConfig(vocab_size=16,hidden_size=2560,intermediate_size=9216,
            num_hidden_layers=1,num_attention_heads=20,num_key_value_heads=5,head_dim=128,
            layer_types=['full_attention'],bos_token_id=1,eos_token_id=4,pad_token_id=0,
            tie_word_embeddings=False)
        model=Qwen3_5ForCausalLM(config).to(torch.bfloat16).eval()
        for p in model.parameters():p.requires_grad_(False)
        before={n:worker.forward.parameter_sha256(p) for n,p in model.named_parameters()}
        ids=torch.tensor([[1,2,3]])
        def forward(arm):
            engine=iv.FixedIntervention(fixture.intervention(),arm,3).install(model)
            try:
                with torch.inference_mode():result=model(input_ids=ids,use_cache=False,logits_to_keep=1).logits.clone()
                self.assertIsNotNone(engine.telemetry);return result,engine.telemetry
            finally:engine.close()
        a,_=forward('base');b,noop=forward('noop');c,treatment=forward('treatment')
        self.assertTrue(torch.equal(a,b));self.assertEqual(noop['changed_entries'],0)
        self.assertGreater(treatment['changed_entries'],0)
        self.assertEqual(before,{n:worker.forward.parameter_sha256(p) for n,p in model.named_parameters()})
        self.assertFalse(model.model.layers[0].mlp.down_proj._forward_hooks)


if __name__=='__main__':unittest.main()
