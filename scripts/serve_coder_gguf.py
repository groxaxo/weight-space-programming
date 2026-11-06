#!/usr/bin/env python3
"""Use a native GGUF completion backend with the existing OpenAI/tool protocol.

The authenticated HF tokenizer supplies exact input token IDs. llama.cpp only
generates tokens; the established strict Qwen tool parser and HTTP/SSE handler
remain responsible for the client-facing protocol.
"""
from __future__ import annotations
import argparse
from http.server import HTTPServer
import importlib.metadata
import json
from pathlib import Path
import time
from urllib.parse import urlparse
from urllib.request import Request, urlopen
import uuid

from coder_common import check, digest, fingerprint, load_config, loads, parse_completion, verify_model_proof
from serve_coder import handler_for, validate_request


class NativeBackend:
    def __init__(self, cfg, tokenizer, identity, native_url, max_context=16384, max_output=2048):
        parsed=urlparse(native_url)
        check(parsed.scheme=='http' and parsed.hostname in ('127.0.0.1','localhost'), 'Native backend must be loopback')
        check(512<=max_context<=32768 and 1<=max_output<max_context,'Invalid native server limits')
        self.cfg,self.tokenizer,self.identity=cfg,tokenizer,identity
        self.native_url=native_url.rstrip('/')
        self.max_context,self.max_output=max_context,max_output

    def complete(self, request):
        name=self.cfg['served_model_name']
        messages,tools,limit,temperature=validate_request(request,name,self.max_context,self.max_output)
        rendered=self.tokenizer.apply_chat_template(messages,tools=tools or None,
            tokenize=False,add_generation_prompt=True,enable_thinking=False)
        ids=self.tokenizer(rendered,add_special_tokens=False)['input_ids']
        check(isinstance(ids,list) and ids and all(type(i) is int and i>=0 for i in ids),'Invalid tokenizer result')
        count=len(ids)
        check(count+limit<=self.max_context,'Context limit exceeded; compress the conversation')
        payload={'prompt':ids,'n_predict':limit,'temperature':temperature,'top_k':0,
            'top_p':0.95 if temperature>0 else 1.0,'min_p':0.0,'repeat_penalty':1.0,
            'presence_penalty':0.0,'frequency_penalty':0.0,'cache_prompt':False,
            'stream':False,'return_tokens':True,'stop':['<|im_end|>','<|endoftext|>']}
        if self.identity.get('lora_artifact_path'):
            payload['lora']=[{'id':self.identity['lora_id'],'scale':self.identity['lora_scale']}]
        started=time.monotonic()
        with urlopen(Request(self.native_url+'/completion',data=json.dumps(payload).encode(),
                     headers={'Content-Type':'application/json'}),timeout=400) as response:
            result=loads(response.read())
        check(Path(result['model']).resolve()==Path(self.identity['artifact_path']).resolve(),'Wrong native model on serving port')
        check(result.get('truncated') is False,'Native backend truncated the prompt')
        if self.identity.get('lora_artifact_path'):
            active=result.get('generation_settings',{}).get('lora',[])
            check(any(a.get('id')==self.identity['lora_id'] and a.get('scale')==self.identity['lora_scale'] for a in active),
                  'Native adapter was not actively applied')
        generated=result['tokens'];n=result['tokens_predicted']
        check(isinstance(generated,list) and all(type(i) is int and i>=0 for i in generated),'Missing native output token IDs')
        check(type(n) is int and 0<=n<=limit and len(generated)<=n,'Invalid native output-token count')
        raw=self.tokenizer.decode(generated,skip_special_tokens=False)
        message=parse_completion(raw,tools)
        finish='tool_calls' if message.get('tool_calls') else ('length' if n>=limit or result.get('stop_type')=='limit' else 'stop')
        return {'id':'chatcmpl-'+uuid.uuid4().hex,'object':'chat.completion','created':int(time.time()),
            'model':name,'choices':[{'index':0,'message':message,'finish_reason':finish}],
            'usage':{'prompt_tokens':count,'completion_tokens':n,'total_tokens':count+n},
            'wsp':{'identity':self.identity,'inference_seconds':time.monotonic()-started,
                   'native_timings':result.get('timings')}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--model-proof',type=Path,required=True)
    parser.add_argument('--identity',type=Path,required=True)
    parser.add_argument('--native-url',default='http://127.0.0.1:18085')
    parser.add_argument('--port',type=int,default=18084)
    parser.add_argument('--max-context',type=int,default=16384)
    parser.add_argument('--max-output',type=int,default=2048)
    args=parser.parse_args();cfg=load_config(args.config)
    identity=loads(args.identity.read_text());proof=loads(args.model_proof.read_text())
    check(identity['base_model']==proof['model_id']==cfg['model_id'] and
          identity['revision']==proof['revision']==cfg['revision'],'Different model identities')
    check(identity['quantization']=='Q8_0','Unexpected GGUF quantization')
    artifact=Path(identity['artifact_path'])
    check(artifact.is_file() and not artifact.is_symlink() and digest(artifact)==identity['artifact_sha256'],
          'GGUF artifact changed')
    if identity.get('lora_artifact_path'):
        lora=Path(identity['lora_artifact_path'])
        check(lora.is_file() and not lora.is_symlink() and digest(lora)==identity['lora_artifact_sha256'],
              'Native LoRA artifact changed')
    snapshot=Path(proof['snapshot'])
    check(snapshot.name==cfg['revision'],'Tokenizer snapshot is not pinned')
    verify_model_proof(cfg, snapshot, args.model_proof)
    protocol_files=[Path(__file__),Path(__file__).with_name('serve_coder.py'),Path(__file__).with_name('coder_common.py'),
                    Path(__file__).resolve().parents[1]/'ops/run54_gguf_phase.py']
    identity['protocol_sha256']=fingerprint({'files':{p.name:digest(p) for p in protocol_files},
        'context':args.max_context,'output':args.max_output,'cache_prompt':False,
        'chat_template':proof['files']['chat_template.jinja']['sha256']})
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(str(snapshot),local_files_only=True,trust_remote_code=False)
    identity['runtime']['tokenizer']={n:importlib.metadata.version(n) for n in ('transformers','tokenizers')}
    identity['chat_stop_token_ids']=[248044,248046]
    check(tokenizer.convert_tokens_to_ids('<|im_end|>')==248046 and
          tokenizer.convert_tokens_to_ids('<|endoftext|>')==248044,'Unexpected chat stop token IDs')
    backend=NativeBackend(cfg,tokenizer,identity,args.native_url,args.max_context,args.max_output)
    with urlopen(args.native_url+'/health',timeout=5) as response:
        check(loads(response.read()).get('status')=='ok','Native model is not ready')
    server=HTTPServer(('127.0.0.1',args.port),handler_for(backend))
    print(json.dumps({'status':'ready','identity':identity}),flush=True)
    try:server.serve_forever()
    finally:server.server_close()


if __name__=='__main__':main()
