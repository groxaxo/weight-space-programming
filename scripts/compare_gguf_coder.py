#!/usr/bin/env python3
"""Apply the declared Q8 contract, including a strict paired HumanEval gain."""
import argparse
import json
from pathlib import Path
import re
from coder_common import check,digest,loads,write_json
from compare_coder import acceptance


def compare_native(base,candidate,base_human,candidate_human,clients,training,conversion):
    # Reuse every frozen-suite, full-result, client, training and runtime integrity check.
    validated=acceptance(base,candidate,base_human,candidate_human,clients,training)
    check(conversion['status']=='converted_unpromoted' and conversion['quantization']=='Q8_0','Unknown conversion profile')
    check(conversion['model_id']==base['identity']['base_model'] and conversion['revision']==base['identity']['revision'],
          'Conversion base identity mismatch')
    for report,default_label in [(base,'base'),(candidate,'candidate-step18')]:
        identity=report['identity']
        check(identity['quantization']=='Q8_0' and identity['runtime'].get('backend')=='llama.cpp','Need a paired native Q8 profile')
        label=identity.get('conversion_profile',default_label)
        rows=[r for r in conversion['conversions'] if r['profile']==label]
        check(len(rows)==1,'Unknown converted artifact')
        key='lora_artifact_sha256' if rows[0].get('kind')=='lora' else 'artifact_sha256'
        check(rows[0]['sha256']==identity[key],'Different converted artifact')
        if key=='lora_artifact_sha256':
            bases=[r for r in conversion['conversions'] if r['profile']==rows[0]['base_profile']]
            check(len(bases)==1 and identity['artifact_sha256']==bases[0]['sha256']==base['identity']['artifact_sha256'],
                  'Native LoRA uses a different base artifact')
        path=Path(rows[0]['output'])
        check(path.is_file() and not path.is_symlink() and path.stat().st_size==rows[0]['bytes'] and
              digest(path)==rows[0]['sha256'],'Converted artifact changed')
        check(re.fullmatch('[a-f0-9]{64}',identity[key]) is not None,'Invalid artifact checksum')
    dev=validated['development']
    code=[r for r in candidate['results'] if r['category']=='code']
    tools=[r for r in candidate['results'] if r['category']=='tools']
    floor=(not dev['regressions'] and sum(r['passed'] for r in code)/len(code)>=0.8 and all(r['passed'] for r in tools))
    gain=bool(dev['improvements']) or candidate_human['passed']>base_human['passed']
    accepted=(floor and gain and candidate_human['passed']>base_human['passed'] and
              validated['real_clients_passed'] and validated['validation_loss_improved'])
    profile='native-Q8_0-unmerged-F32-v2' if candidate['identity'].get('lora_artifact_path') else 'native-Q8_0-compiled-v1'
    return {'status':'accepted_for_local_use' if accepted else 'not_promoted','format_profile':profile,
        'development':{'quality_floor_passed':floor,'regressions':dev['regressions'],'improvements':dev['improvements']},
        'humaneval':validated['humaneval'],'real_clients_passed':validated['real_clients_passed'],
        'validation_loss_improved':validated['validation_loss_improved'],'functional_gain':gain,
        'scope':'Complete paired Python benchmarks and isolated real-client edits. HumanEval is public and used for selection; frontier equivalence is unproven.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['baseline','candidate','baseline-humaneval','candidate-humaneval','clients','training','conversion','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    paths=[args.baseline,args.candidate,args.baseline_humaneval,args.candidate_humaneval,args.clients,args.training,args.conversion]
    result=compare_native(*[loads(p.read_text()) for p in paths])
    result['input_sha256']={str(p):digest(p) for p in paths}
    result['contract_sha256']=digest(Path(__file__).resolve().parents[1]/'docs/native-acceptance.md')
    write_json(args.output,result)
    print(json.dumps(result,indent=2))
    raise SystemExit(0 if result['status']=='accepted_for_local_use' else 2)


if __name__=='__main__':main()
