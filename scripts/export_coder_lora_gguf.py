#!/usr/bin/env python3
"""Export an exact FP32 LoRA with factor-aware Qwen hybrid-head permutation.

The installed llama.cpp converter is imported read-only. Its permutation method
is patched in this process only: row permutations affect B, column permutations
affect A, preserving P_row @ (B @ A) @ P_column without forming a dense update.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import runpy
import shutil
import sys
import tempfile
import numpy as np

from coder_common import check,digest,load_config,loads,verify_model_proof,write_json
from compile_coder import validate_adapter
from pipeline import publish_directory,sync_directory


def reorder_factors(a,b,dim,num_k_heads,num_v_per_k,head_dim,reorder):
    check(len(a.shape)==len(b.shape)==2,'Only matrix LoRA permutations are supported')
    dim=dim+2 if dim<0 else dim
    check(dim in (0,1),'Invalid LoRA permutation axis')
    if dim==0:b=reorder(b,0,num_k_heads,num_v_per_k,head_dim)
    else:a=reorder(a,1,num_k_heads,num_v_per_k,head_dim)
    return a,b


def install_reorder_patch(qwen):
    owners=[value for value in vars(qwen).values() if isinstance(value,type) and '_reorder_v_heads' in value.__dict__]
    check(len(owners)==1,'Unexpected converter permutation implementation')
    owner=owners[0];original=owner._reorder_v_heads;counts={'rows':0,'columns':0}
    def patched(tensor,dim,num_k_heads,num_v_per_k,head_dim):
        getter=getattr(tensor,'get_lora_A_B',None)
        if callable(getter):
            a,b=getter()
            a,b=reorder_factors(a,b,dim,num_k_heads,num_v_per_k,head_dim,original)
            counts['rows' if dim in (0,-2) else 'columns']+=1
            return type(tensor)(a,b)
        return original(tensor,dim,num_k_heads,num_v_per_k,head_dim)
    owner._reorder_v_heads=staticmethod(patched)
    return counts


def verify_permutations(targets, counts):
    """Require only the hybrid transforms present in this adapter's targets."""
    row_names = ('in_proj_qkv', 'in_proj_z', 'in_proj_b', 'in_proj_a')
    expected_rows = sum('.linear_attn.' in n and n.rsplit('.', 1)[-1] in row_names for n in targets)
    expected_columns = sum('.linear_attn.' in n and n.endswith('.out_proj') for n in targets)
    check(counts == {'rows': expected_rows, 'columns': expected_columns},
          'Native adapter permutation coverage differs from its target modules')


def main():
    from method_policy import blocked_legacy_operation
    blocked_legacy_operation('Trained LoRA export')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--llama-source',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--model-proof',type=Path,required=True)
    parser.add_argument('--training',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();cfg=load_config(args.config)
    check(os.environ.get('CUDA_VISIBLE_DEVICES') in ('','-1'),'Export must run with CUDA hidden')
    check(not os.path.lexists(args.output),'Refusing existing export output')
    proof=loads(args.model_proof.read_text());base=Path(proof['snapshot'])
    verify_model_proof(cfg,base,args.model_proof)
    result=loads((args.training/'training-result.json').read_text())
    run=loads((args.training/'run-manifest.json').read_text())
    completed=loads((args.training/'completed-run-manifest.json').read_text())
    check(completed['status']=='complete' and completed['exit_code']==0 and result['status']=='trained_unpromoted'
          and not result['smoke_only'] and run['model_id']==cfg['model_id'] and run['model_revision']==cfg['revision'],
          'Incomplete or different source adapter')
    adapter=args.training/'adapter'
    for name,expected in result['adapter_files'].items():
        path=adapter/name
        check(Path(name).name==name and path.is_file() and not path.is_symlink() and digest(path)==expected,
              'Source adapter artifact changed')
    validate_adapter(loads((adapter/'adapter_config.json').read_text()))
    targets=loads((args.training/'adapter-targets.json').read_text())
    source=args.llama_source.resolve();converter=source/'convert_lora_to_gguf.py'
    original_hashes={str(p.relative_to(source)):digest(p) for p in
                     [converter,source/'conversion/qwen.py']}
    sys.path[:0]=[str(source),str(source/'gguf-py')]
    from conversion import qwen
    counts=install_reorder_patch(qwen)
    args.output=args.output.absolute();args.output.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.'+args.output.name+'.building-',dir=args.output.parent))
    try:
        artifact=stage/'adapter-F32.gguf'
        sys.argv=[str(converter),'--base',str(base),'--outtype','f32','--outfile',str(artifact),str(adapter)]
        try:runpy.run_path(str(converter),run_name='__main__')
        except SystemExit as error:check(error.code in (None,0),'Upstream converter failed')
        check(artifact.is_file() and not artifact.is_symlink(),'Missing exported GGUF adapter')
        from gguf import GGUFReader
        reader=GGUFReader(str(artifact))
        check(len(reader.tensors)==2*len(targets),'Export changed the LoRA tensor count')
        check(all(t.data.dtype==np.dtype('float32') and np.isfinite(t.data).all() for t in reader.tensors),
              'Expected finite FP32 GGUF factors')
        verify_permutations(targets, counts)
        del reader
        record={'schema_version':1,'status':'exported_unpromoted','created_at_utc':datetime.now(timezone.utc).isoformat(),
            'model_id':cfg['model_id'],'revision':cfg['revision'],'source_adapter':str(adapter.resolve()),
            'training_result_sha256':digest(args.training/'training-result.json'),
            'adapter_files':result['adapter_files'],'tensor_count':2*len(targets),'factor_dtype':'F32',
            'permutations':counts,'converter_source_sha256':original_hashes,'patch_sha256':digest(Path(__file__)),
            'artifact':artifact.name,'artifact_bytes':artifact.stat().st_size,'artifact_sha256':digest(artifact),
            'scope':'Exact unmerged factors with native layout permutations; requires fresh paired inference acceptance.'}
        check(all(digest(source/n)==h for n,h in original_hashes.items()),'Protected converter source changed')
        write_json(stage/'export.json',record)
        for p in stage.iterdir():
            with p.open('rb') as f:os.fsync(f.fileno())
        sync_directory(stage);publish_directory(stage,args.output);sync_directory(args.output.parent)
        print(json.dumps(record,indent=2),flush=True)
    finally:
        if stage.exists():shutil.rmtree(stage)


if __name__=='__main__':main()
