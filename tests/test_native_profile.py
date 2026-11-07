"""Maximum-context and quantized-cache contracts for the clean-VM HE9 replay."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import copy
import socket
import tempfile
import json

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'scripts')]
from ops import run54_gguf_phase as phase
from coder_common import CoderError
from serve_coder_gguf import NativeBackend
import test_gguf_backend as control


class NativeProfileTests(unittest.TestCase):
    def arguments(self):
        return SimpleNamespace(max_context=262144,max_output=8192,cache_type_k='q8_0',cache_type_v='q8_0',thinking=True)

    def test_pinned_model_maximum_context_is_supported_without_rope_extension(self):
        backend = NativeBackend({'served_model_name':'unit-model'},control.Tokenizer(),{},'http://127.0.0.1:1',262144,8192)
        self.assertEqual(backend.max_context,262144)
        with self.assertRaises(CoderError):
            NativeBackend({},control.Tokenizer(),{},'http://127.0.0.1:1',262145,8192)

    def test_native_process_uses_the_same_context_and_q8_caches_as_its_identity(self):
        args=self.arguments()
        settings=phase.inference_settings(args)
        self.assertEqual(settings['context'],262144)
        self.assertEqual(settings['output'],8192)
        identity={'runtime':{'launch_options':settings}}
        argv=phase.native_command(Path('/tmp/test-model.gguf'),identity)
        for flag,value in [('--ctx-size','262144'),('--cache-type-k','q8_0'),('--cache-type-v','q8_0')]:
            self.assertEqual(argv[argv.index(flag)+1],value)
        self.assertIn('--no-context-shift',argv)
        for key,value in [('max_context',262145),('max_output',262144),('cache_type_k','q4_0')]:
            broken=self.arguments();setattr(broken,key,value)
            with self.subTest(key=key),self.assertRaises(CoderError):phase.inference_settings(broken)

    def test_reasoning_opened_by_the_prompt_is_not_returned_as_answer_text(self):
        class Tokenizer(control.Tokenizer):
            def apply_chat_template(self,*args,**kwargs):
                self.thinking=kwargs['enable_thinking']
                return '<|im_start|>assistant\n<think>\n'
            def __call__(self,*args,**kwargs):return {'input_ids':[11,12]}
            def decode(self,*args,**kwargs):return 'Internal draft.\n</think>\nDone.<|im_end|>'
        tokenizer=Tokenizer()
        response=control.GgufBackendTests().result()
        with control.native_control(response) as (url,_):
            backend=NativeBackend({'served_model_name':'unit-model'},tokenizer,
                {'artifact_path':'/tmp/wsp-unit-model.gguf'},url,262144,8192,enable_thinking=True)
            result=backend.complete(control.GgufBackendTests().request())
        self.assertTrue(tokenizer.thinking)
        self.assertEqual(result['choices'][0]['message']['content'],'Done.')

    def test_compiled_coverage_follows_the_actual_mlp_targets_not_a_fixed_count(self):
        targets = ['model.language_model.layers.'+str(i)+'.mlp.'+name
                   for i in range(32) for name in ('gate_proj', 'up_proj', 'down_proj')]
        record = {'tensor_checks': [{'name': n+'.weight'} for n in targets],
            'verification': {'all_unedited_bytes_preserved': True, 'edited_tensors': 96}}
        phase.verify_compiled_targets(record, targets)
        for fault in ('missing', 'duplicate', 'unrelated', 'wrong_count'):
            broken = copy.deepcopy(record)
            if fault == 'missing': broken['tensor_checks'].pop()
            if fault == 'duplicate': broken['tensor_checks'][0] = broken['tensor_checks'][1]
            if fault == 'unrelated': broken['tensor_checks'][0]['name'] = 'model.vision.weight'
            if fault == 'wrong_count': broken['verification']['edited_tensors'] = 248
            with self.subTest(fault=fault), self.assertRaises(CoderError):
                phase.verify_compiled_targets(broken, targets)

    def test_native_port_probe_allows_time_wait_and_rejects_a_live_listener(self):
        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
            listener.bind(('127.0.0.1',0));listener.listen()
            address=listener.getsockname()
            with self.assertRaises(OSError):phase.verify_native_port_free(address)
            with socket.create_connection(address,timeout=3) as client:
                accepted,_=listener.accept()
                accepted.close()  # Active server close puts this port in TIME_WAIT.
                self.assertEqual(client.recv(1),b'')
        phase.verify_native_port_free(address)

    def test_failed_thinking_completion_keeps_opt_in_raw_evidence_without_relaxing_the_parser(self):
        class Tokenizer(control.Tokenizer):
            def apply_chat_template(self,*args,**kwargs):return '<|im_start|>assistant\n<think>\n'
            def __call__(self,*args,**kwargs):return {'input_ids':[11,12]}
            def decode(self,*args,**kwargs):return 'Unterminated internal draft.<|im_end|>'
        with tempfile.TemporaryDirectory() as directory, control.native_control(control.GgufBackendTests().result()) as (url,_):
            audit=Path(directory)/'private-trace'
            backend=NativeBackend({'served_model_name':'unit-model'},Tokenizer(),
                {'artifact_path':'/tmp/wsp-unit-model.gguf'},url,262144,8192,enable_thinking=True,audit_directory=audit)
            with self.assertRaisesRegex(CoderError,'thinking block'):
                backend.complete(control.GgufBackendTests().request())
            paths=list(audit.glob('*.json'))
            self.assertEqual(len(paths),1)
            record=json.loads(paths[0].read_text())
            self.assertEqual(record['generated_text'],'Unterminated internal draft.<|im_end|>')
            self.assertTrue(record['parser_input'].startswith('<think>\n'))
            self.assertIn('thinking block',record['parse_error'])


if __name__=='__main__':unittest.main()
