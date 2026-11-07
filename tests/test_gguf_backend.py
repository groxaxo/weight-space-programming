"""Native transport controls use an explicit fake model, never a quality score."""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
from pathlib import Path
import sys
import threading
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from serve_coder_gguf import NativeBackend
from coder_common import CoderError


class Tokenizer:
    def apply_chat_template(self,*args,**kwargs):return 'exact-template'
    def __call__(self,text,add_special_tokens):
        if text!='exact-template' or add_special_tokens:raise ValueError('Tokenizer contract changed')
        return {'input_ids':[11,12]}
    def decode(self,tokens,skip_special_tokens):return 'Native transport control.'


@contextmanager
def native_control(response):
    requests=[]
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*_args):pass
        def do_POST(self):
            requests.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
            blob=json.dumps(response).encode()
            self.send_response(200);self.send_header('Content-Length',str(len(blob)));self.end_headers();self.wfile.write(blob)
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:yield 'http://127.0.0.1:'+str(server.server_port),requests
    finally:server.shutdown();server.server_close();thread.join(timeout=5)


class GgufBackendTests(unittest.TestCase):
    def result(self,**changes):
        return dict({'model':'/tmp/wsp-unit-model.gguf','truncated':False,'tokens':[21,22],
                     'tokens_predicted':2,'stop_type':'eos'},**changes)

    def request(self):return {'model':'unit-model','messages':[{'role':'user','content':'Hello'}],'max_tokens':8,'temperature':0}

    def backend(self,url):
        return NativeBackend({'served_model_name':'unit-model'},Tokenizer(),
                             {'artifact_path':'/tmp/wsp-unit-model.gguf'},url,512,16)

    def test_exact_token_ids_and_usage_survive_real_native_http_transport(self):
        with native_control(self.result()) as (url,requests):
            result=self.backend(url).complete(self.request())
        self.assertEqual(requests[0]['prompt'],[11,12])
        self.assertEqual(requests[0]['n_predict'],8)
        self.assertFalse(requests[0]['cache_prompt'])
        self.assertTrue(requests[0]['return_tokens'])
        self.assertEqual(result['usage'],{'prompt_tokens':2,'completion_tokens':2,'total_tokens':4})
        self.assertEqual(result['choices'][0]['finish_reason'],'stop')

    def test_native_limit_is_not_reported_as_a_successful_stop(self):
        with native_control(self.result(tokens=[21]*8,tokens_predicted=8,stop_type='limit')) as (url,_):
            result=self.backend(url).complete(self.request())
        self.assertEqual(result['choices'][0]['finish_reason'],'length')

    def test_declared_instruct_sampling_reaches_the_native_engine(self):
        sampling={'temperature':0.7,'top_p':0.8,'top_k':20,'min_p':0.0,
            'presence_penalty':1.5,'frequency_penalty':0.0,'repeat_penalty':1.0,'seed':20261106}
        with native_control(self.result(generation_settings=sampling)) as (url,requests):
            backend=self.backend(url)
            backend.identity['runtime']={'sampling':sampling}
            request={**self.request(),**sampling}
            backend.complete(request)
        self.assertEqual({k:requests[0][k] for k in sampling},sampling)

    def test_wrong_loaded_artifact_and_silent_context_truncation_fail_closed(self):
        for changes in [{'model':'/tmp/wrong-model.gguf'},{'truncated':True}]:
            with self.subTest(changes=changes),native_control(self.result(**changes)) as (url,_):
                with self.assertRaises(CoderError):self.backend(url).complete(self.request())

    def test_native_lora_must_be_explicitly_requested_and_observed_active(self):
        for active in [False,True]:
            response=self.result(generation_settings={'lora':[{'id':0,'scale':1.0}] if active else []})
            with self.subTest(active=active),native_control(response) as (url,requests):
                backend=self.backend(url)
                backend.identity.update(lora_artifact_path='/tmp/unit-lora.gguf',lora_id=0,lora_scale=1.0)
                if active:self.assertEqual(backend.complete(self.request())['choices'][0]['finish_reason'],'stop')
                else:
                    with self.assertRaisesRegex(CoderError,'not actively applied'):backend.complete(self.request())
                self.assertEqual(requests[0]['lora'],[{'id':0,'scale':1.0}])


if __name__=='__main__':unittest.main()
