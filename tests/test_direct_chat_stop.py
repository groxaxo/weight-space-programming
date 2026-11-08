"""A generation must end at the FIRST configured assistant/chat EOS token."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'tests')]
import direct_forward as forward
import direct_evidence as evidence
import direct_tools as tools
import test_direct_discrepancies as fixture
from direct_evidence_fixtures import make_worker, write
from ops import direct_edit54 as host
from pipeline import ValidationError


class ChatStopRegression(unittest.TestCase):
    def test_generation_receives_the_same_model_and_tokenizer_eos_union_that_is_reported(self):
        value = fixture.AuditedBoundaryTests(methodName='runTest'); value.setUp()
        self.addCleanup(value.doCleanups)
        original = fixture._Model.generate
        options = {}
        def capture(model, *args, **kwargs):
            options.update(kwargs)
            return original(model, *args, **kwargs)
        with mock.patch.object(fixture._Model, 'generate', capture):
            report, _ = value.observe()
        self.assertEqual(options.get('eos_token_id'), [2, 4])
        self.assertEqual(options['eos_token_id'], report['results'][0]['eos_token_ids'])

    def test_completion_rejects_backend_tokens_past_first_chat_stop_instead_of_trimming(self):
        model = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=4))
        tokenizer = SimpleNamespace(eos_token_id=2)
        with self.assertRaisesRegex(ValidationError, 'first|past|internal'):
            forward.completion_status([3, 2, 7, 4], 96, model, tokenizer)

    def test_host_refuses_internal_stop_even_when_final_eos_hashes_and_all_flags_agree(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory); request, report, path = make_worker(output, 'observe')
            row = report['results'][0]
            row.update(output_ids=[3, 2, 3, 2], generated_tokens=4)
            raw = {k: v for k, v in row.items() if k != 'generation_sha256'}
            row['generation_sha256'] = write(output / (row['id'] + '-generation.json'), raw)['sha256']
            write(path, report)
            with self.assertRaisesRegex(ValidationError, 'first|past|internal'):
                host.worker_evidence(output, 'observe', request, 'f' * 64, request['lock_sha256'])

    def test_internal_chat_turns_are_explicit_harness_fault_before_tool_parsing(self):
        raw = ('<tool_call>{"name":"run_tests","arguments":{"target":"parser"}}</tool_call>'
               '<|im_end|>\n<|im_start|>user\n<tool_response>{"exit_code":1}</tool_response>'
               '<|im_end|>\n<|im_start|>assistant\n{"exit_code":1,"failed":2}<|endoftext|>')
        result = tools.analyze({'censored': False, 'text_with_special_tokens': raw}, 'continuation')
        self.assertEqual(result['classification'], 'STOP_conversation_control')
        self.assertTrue(result['parser_agreement'])
        self.assertIsNone(result['parsed_message'])

    def test_first_chat_eos_at_cap_remains_completed_with_one_terminal_stop(self):
        model = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=4))
        tokenizer = SimpleNamespace(eos_token_id=2)
        value = forward.completion_status([3] * 95 + [2], 96, model, tokenizer)
        self.assertFalse(value['censored']); self.assertFalse(value['hit_token_limit'])
        self.assertEqual(value['finish_reason'], 'eos')


if __name__ == '__main__': unittest.main()

import importlib.util
import os


@unittest.skipUnless(os.environ.get('WSP_TEST_DIRECT_TOOL_TORCH') == '1' and importlib.util.find_spec('torch'),
                     'explicit tiny chat-stop CPU runtime container required')
class TinyChatStopRuntime(unittest.TestCase):
    def test_transformers_generation_stops_at_chat_eos_when_model_default_is_endoftext(self):
        import torch
        from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM, LogitsProcessor, LogitsProcessorList
        torch.set_num_threads(1)
        config = Qwen3_5TextConfig(vocab_size=16, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2, head_dim=8,
            layer_types=['full_attention'], bos_token_id=1, eos_token_id=4, pad_token_id=0,
            tie_word_embeddings=False)
        model = Qwen3_5ForCausalLM(config).to(torch.bfloat16).eval()
        for parameter in model.parameters(): parameter.requires_grad_(False)
        class ForceChatEnd(LogitsProcessor):
            def __call__(self, ids, scores):
                result = torch.full_like(scores, float('-inf')); result[:, 2] = 0
                return result
        tokenizer = SimpleNamespace(eos_token_id=2)
        ids = torch.tensor([[1, 3]], dtype=torch.long)
        kwargs = {'input_ids': ids, 'attention_mask': torch.ones_like(ids), 'do_sample': False,
                  'max_new_tokens': 3, 'pad_token_id': 0,
                  'logits_processor': LogitsProcessorList([ForceChatEnd()])}
        with torch.inference_mode():
            unbounded = model.generate(**kwargs)[0, 2:].tolist()
            correct = model.generate(**kwargs, eos_token_id=forward.chat_stop_ids(model, tokenizer))[0, 2:].tolist()
        self.assertEqual(unbounded, [2, 2, 2])
        self.assertEqual(correct, [2])
        self.assertTrue(forward.completion_status(correct, 3, model, tokenizer)['first_eos_is_terminal'])
        with self.assertRaises(ValidationError): forward.completion_status(unbounded, 3, model, tokenizer)
