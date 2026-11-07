#!/usr/bin/env python3
"""Supervise a native Q8 profile on .54 using the existing GPU/Bonsai journal."""
from datetime import datetime, timezone
import argparse
import json
import os
from pathlib import Path
import socket
import signal
import subprocess
import sys
import time
from urllib.request import urlopen

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from coder_common import adapter_identity,allocate_device,check,digest,fingerprint,load_config,loads,write_json
from ops.run54_gpu_phase import gpu_window,restore,stop_child
from coder_sampling import sampling_for_profile

CPP=Path('/home/op/src/llama.cpp-prismml-mtp/src/build/bin')
IMAGE='python:3.12-slim@sha256:4840d514c583a4d3d8cad3940e68dee6f0f6f3635c12fcd0a3d74c21a5b93a54'


def inference_settings(args):
    settings={'context':getattr(args,'max_context',16384),'output':getattr(args,'max_output',2048),
        'cache_type_k':getattr(args,'cache_type_k','f16'),'cache_type_v':getattr(args,'cache_type_v','f16'),
        'thinking':getattr(args,'thinking',False),'slots':1,'gpu_layers':'all','flash_attention':'on',
        'threads':4,'threads_batch':4,'context_shift':False,'cache_prompt':False,'warmup':False,'seed':20261106,
        'sampling_profile':getattr(args,'sampling_profile','greedy')}
    check(type(settings['context']) is int and 512<=settings['context']<=262144,
          'Context exceeds the pinned Qwen3.5-4B supported maximum')
    check(type(settings['output']) is int and 1<=settings['output']<settings['context'],'Invalid output allowance')
    check(settings['cache_type_k'] in ('f16','q8_0') and settings['cache_type_v'] in ('f16','q8_0'),'Unsupported KV cache type')
    check(type(settings['thinking']) is bool,'Invalid thinking flag')
    sampling_for_profile(settings['sampling_profile'])
    check(settings['sampling_profile']!='qwen35-instruct' or not settings['thinking'],
          'The instruct sampling profile requires non-thinking mode')
    return settings


def native_command(artifact,identity):
    settings=identity['runtime']['launch_options']
    command=[str(CPP/'llama-server'),'--model',str(artifact),'--host','127.0.0.1','--port','18085',
        '--ctx-size',str(settings['context']),'--parallel','1','--n-gpu-layers','all',
        '--threads','4','--threads-batch','4','--flash-attn','on','--no-context-shift','--no-warmup',
        '--cache-type-k',settings['cache_type_k'],'--cache-type-v',settings['cache_type_v'],'--seed','20261106']
    if identity.get('lora_artifact_path'):command.extend(['--lora',identity['lora_artifact_path']])
    return command


def verify_compiled_targets(compiled, targets):
    names=[row['name'] for row in compiled['tensor_checks']]
    expected={name+'.weight' for name in targets}
    check(targets and len(expected)==len(targets) and len(names)==len(set(names))
          and set(names)==expected and compiled['verification']['all_unedited_bytes_preserved'] is True
          and compiled['verification']['edited_tensors']==len(targets),
          'Incomplete language-layer compilation proof')


def verify_native_port_free(address=('127.0.0.1',18085)):
    # Match the native HTTP listener's reuse semantics. A recently closed
    # connection in TIME_WAIT is not an active serving process. SO_REUSEADDR
    # still refuses an address held by a live listener; never stop that owner.
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        probe.bind(address)


def wait_ready(url,process,seconds=300):
    deadline=time.monotonic()+seconds
    while True:
        check(process.poll() is None,'Serving process exited during startup')
        try:
            with urlopen(url,timeout=3) as response:return loads(response.read())
        except OSError:
            check(time.monotonic()<deadline,'Serving startup timed out')
            time.sleep(2)


def worker(command,output,name,seconds,servers,accepted=(0,)):
    with (output/name).open('x') as log:
        process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        try:
            deadline=time.monotonic()+seconds
            while process.poll() is None:
                check(all(p.poll() is None for p in servers),'A serving process exited during evaluation')
                check(time.monotonic()<deadline,'Evaluation worker timed out')
                time.sleep(1)
            check(process.returncode in accepted,'Evaluation worker failed; inspect '+name)
        finally:stop_child(process)


def selected_identity(args,cfg):
    record=loads(args.conversion.read_text())
    check(record['status']=='converted_unpromoted' and record['quantization']=='Q8_0','Incomplete Q8 conversion')
    check(record['model_id']==cfg['model_id'] and record['revision']==cfg['revision'],'Different conversion model')
    rows=[r for r in record['conversions'] if r['profile']==args.profile]
    check(len(rows)==1,'Unknown or duplicate conversion profile')
    selected=rows[0];kind=selected.get('kind','model')
    check(kind in ('model','lora'),'Unknown native artifact kind')
    def verified_artifact(row):
        path=Path(row['output'])
        check(path.is_file() and not path.is_symlink() and path.stat().st_size==row['bytes']
              and digest(path)==row['sha256'],'GGUF artifact changed')
        return path
    model_entry=selected
    lora=None
    if kind=='lora':
        bases=[r for r in record['conversions'] if r['profile']==selected['base_profile']]
        check(len(bases)==1 and bases[0].get('kind','model')=='model','Missing native LoRA base')
        model_entry=bases[0];lora=verified_artifact(selected)
    artifact=verified_artifact(model_entry)
    binary=CPP/'llama-server'
    libraries=sorted({p.resolve() for p in CPP.glob('*.so*') if p.is_file()})
    settings=inference_settings(args)
    runtime={'backend':'llama.cpp','server_sha256':digest(binary),
        'shared_libraries':{p.name:digest(p) for p in libraries},
        'launch_options':settings}
    sampling=sampling_for_profile(settings['sampling_profile'])
    if sampling is not None:runtime['sampling']=sampling
    identity={'identity_schema':2,'model_proof_sha256':fingerprint(loads(args.model_proof.read_text())),
        'limits':{'context':settings['context'],'output':settings['output']},'base_model':cfg['model_id'],'revision':cfg['revision'],'adapter':None,
        'quantization':'Q8_0','runtime':runtime,'artifact_path':str(artifact.resolve()),
        'artifact_sha256':model_entry['sha256'],'conversion_manifest_sha256':digest(args.conversion),
        'conversion_profile':args.profile}
    if args.profile!='base':
        if kind=='lora':
            provenance=Path(selected['export_record'])
            check(provenance.is_file() and digest(provenance)==selected['export_record_sha256'],'Export record changed')
            exported=loads(provenance.read_text());adapter=Path(exported['source_adapter'])
            check(exported['status']=='exported_unpromoted' and exported['factor_dtype']=='F32' and
                  exported['artifact_sha256']==selected['sha256'] and exported['model_id']==cfg['model_id'] and
                  exported['revision']==cfg['revision'],'Different LoRA export')
            training_hash=exported['training_result_sha256']
        else:
            provenance=args.conversion.parent/'compile-record.json'
            check(provenance.is_file() and digest(provenance)==record['compile_record_sha256'],'Missing verified compile record')
            compiled=loads(provenance.read_text());adapter=Path(compiled['source_adapter'])
            verify_compiled_targets(compiled,loads((adapter.parent/'adapter-targets.json').read_text()))
            training_hash=record['compiler_provenance']['training_result_sha256']
        result=loads((adapter.parent/'training-result.json').read_text())
        run=loads((adapter.parent/'run-manifest.json').read_text())
        completed=loads((adapter.parent/'completed-run-manifest.json').read_text())
        check(result['status']=='trained_unpromoted' and result['smoke_only'] is False and
              completed['status']=='complete' and completed['exit_code']==0 and
              run['model_id']==cfg['model_id'] and run['model_revision']==cfg['revision'],'Invalid source adapter run')
        check(digest(adapter.parent/'training-result.json')==training_hash,
              'Different source training record')
        for name,expected in result['adapter_files'].items():
            check(Path(name).name==name and (adapter/name).is_file() and not (adapter/name).is_symlink()
                  and digest(adapter/name)==expected,'Source adapter artifact changed')
        identity.update(adapter=str(adapter),adapter_sha256=result['adapter_files']['adapter_model.safetensors'],
            adapter_config_sha256=result['adapter_files']['adapter_config.json'],adapter_provenance_sha256=digest(provenance))
        identity.update(adapter_identity(result))
        if lora is not None:
            targets=loads((adapter.parent/'adapter-targets.json').read_text())
            check(exported['tensor_count']==2*len(targets),'Incomplete FP32 native adapter')
            identity.update(adapter_application='native-f32-lora-on-q8_0',lora_artifact_path=str(lora.resolve()),
                lora_artifact_sha256=selected['sha256'],lora_id=0,lora_scale=1.0)
        else:identity['adapter_application']='compiled-bf16-then-q8_0'
    return artifact,identity


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['evaluate','benchmark','serve'],default='evaluate')
    parser.add_argument('--conversion',type=Path,required=True)
    parser.add_argument('--profile',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/qwen35-4b-lr5e-6.json')
    parser.add_argument('--model-proof',type=Path,default=ROOT/'reports/model-inputs.json')
    parser.add_argument('--allow-pause-bonsai',action='store_true')
    parser.add_argument('--clients',action='store_true')
    parser.add_argument('--project',action='store_true')
    parser.add_argument('--image',default=IMAGE)
    parser.add_argument('--max-context',type=int,default=16384)
    parser.add_argument('--max-output',type=int,default=2048)
    parser.add_argument('--cache-type-k',choices=['f16','q8_0'],default='f16')
    parser.add_argument('--cache-type-v',choices=['f16','q8_0'],default='f16')
    parser.add_argument('--thinking',action='store_true')
    parser.add_argument('--sampling-profile',choices=['greedy','qwen35-instruct'],default='greedy')
    parser.add_argument('--audit-generations',action='store_true',help='Write private request/token evidence for a controlled run')
    args=parser.parse_args();args.output=args.output.absolute();args.conversion=args.conversion.resolve()
    check(os.uname().nodename=='4790','This supervisor is for .54 only')
    check(not args.output.exists(),'Refusing existing native phase output')
    args.output.mkdir(parents=True)
    cfg=load_config(args.config)
    write_json(args.output/'phase.json',{'phase':args.phase,'profile':args.profile,'quantization':'Q8_0',
        'started_at_utc':datetime.now(timezone.utc).isoformat(),'conversion':str(args.conversion)})
    def terminate(signum,_frame):raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,terminate);signal.signal(signal.SIGINT,terminate)
    native=front=None
    try:
        artifact,identity=selected_identity(args,cfg)
        with gpu_window(args.output,args.allow_pause_bonsai):
            allocate_device(True, wait_seconds=60)
            verify_native_port_free()
            command=native_command(artifact,identity)
            write_json(args.output/'native-command.json',{'argv':command,'identity_before_frontend':identity})
            with (args.output/'native.log').open('x') as native_log,(args.output/'server.log').open('x') as front_log:
                native_env={k:v for k,v in os.environ.items() if not k.startswith('LLAMA_ARG_')}
                native=subprocess.Popen(command,cwd=ROOT,env=native_env,stdout=native_log,stderr=subprocess.STDOUT)
                try:
                    info=wait_ready('http://127.0.0.1:18085/health',native)
                    check(info.get('status')=='ok','Native model is not healthy')
                    if identity.get('lora_artifact_path'):
                        with urlopen('http://127.0.0.1:18085/lora-adapters',timeout=5) as response:loaded=loads(response.read())
                        check(any(Path(a['path']).resolve()==Path(identity['lora_artifact_path']) and a['id']==identity['lora_id']
                                  for a in loaded),'Expected native LoRA is not loaded')
                    write_json(args.output/'identity.json',identity)
                    front_env=dict(os.environ,CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                                   OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
                    frontend_command=[str(ROOT/'.venv-train/bin/python'),'-u','scripts/serve_coder_gguf.py',
                        '--config',str(args.config),'--model-proof',str(args.model_proof),'--identity',str(args.output/'identity.json'),
                        '--max-context',str(identity['limits']['context']),'--max-output',str(identity['limits']['output'])]
                    if args.thinking:frontend_command.append('--thinking')
                    if args.audit_generations:frontend_command.extend(['--audit-directory',str(args.output/'generations')])
                    front=subprocess.Popen(frontend_command,
                        cwd=ROOT,env=front_env,stdout=front_log,stderr=subprocess.STDOUT)
                    ready=wait_ready('http://127.0.0.1:18084/health',front)
                    check(ready['wsp']['artifact_sha256']==identity['artifact_sha256'],'Unexpected frontend identity')
                    write_json(args.output/'server-ready.json',ready)
                    python=str(ROOT/'.venv-train/bin/python');servers=[native,front]
                    if args.phase=='serve':
                        while True:
                            check(all(p.poll() is None for p in servers),'Serving process exited')
                            time.sleep(1)
                    if args.phase=='evaluate':
                        worker([python,'-u','scripts/evaluate_coder.py','--image',args.image,'--output',str(args.output/'evaluation.json')],args.output,'evaluation.log',1800,servers)
                    if args.clients:
                        worker([python,'-u','ops/smoke_clients.py','--image',args.image,'--output',str(args.output/'clients')],args.output,'clients.log',1000,servers,(0,2))
                    if args.project:
                        worker([python,'-u','ops/project_clients.py','--image',args.image,'--output',str(args.output/'project-clients')],args.output,'project-clients.log',2100,servers,(0,2))
                    if args.phase=='benchmark':
                        worker([python,'-u','scripts/evaluate_humaneval.py','--image',args.image,'--output',str(args.output/'humaneval.json')],args.output,'humaneval.log',7200,servers)
                finally:
                    if front is not None:stop_child(front)
                    if native is not None:stop_child(native)
    except BaseException as error:
        if args.phase=='serve' and isinstance(error,SystemExit) and error.code in (130,143):
            write_json(args.output/'phase-completed.json',{'status':'stopped','phase':args.phase})
            return
        write_json(args.output/'phase-failed.json',{'status':'failed','error':f'{type(error).__name__}: {error}'})
        raise
    write_json(args.output/'phase-completed.json',{'status':'complete','phase':args.phase})


if __name__=='__main__':main()
