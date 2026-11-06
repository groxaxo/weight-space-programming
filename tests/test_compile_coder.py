"""Golden BF16 arithmetic, sharded byte preservation and atomic failure controls."""
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from compile_coder import to_bf16, compile_checkpoint, verify_compiled
from coder_common import CoderError
from pipeline import read_header

TARGET = 'model.language_model.layers.0.proj'


def write_tensors(path, tensors):
    header = {'__metadata__': {'format': 'pt'}}
    payload = bytearray()
    for name, dtype, shape, raw in tensors:
        start = len(payload);payload.extend(raw)
        header[name] = {'dtype': dtype, 'shape': shape, 'data_offsets': [start,len(payload)]}
    text = json.dumps(header).encode()
    text += b' ' * (-len(text) % 8)
    path.write_bytes(struct.pack('<Q',len(text))+text+payload)


def fixture(root, fault=None):
    base=root/'base';adapter=root/'adapter';base.mkdir();adapter.mkdir()
    write_tensors(base/'one.safetensors', [(TARGET+'.weight','BF16',[2,2],struct.pack('<4H',0x3f80,0x4000,0x4040,0x4080)),
                  ('untouched.norm','F32',[2],struct.pack('<2f',7,8))])
    write_tensors(base/'two.safetensors', [('untouched.vision','BF16',[2],struct.pack('<2H',0x4100,0x4180))])
    index={TARGET+'.weight':'one.safetensors','untouched.norm':'one.safetensors','untouched.vision':'two.safetensors'}
    (base/'model.safetensors.index.json').write_text(json.dumps({'weight_map':index}))
    (base/'config.json').write_text('{}')
    (adapter/'adapter_config.json').write_text(json.dumps({'peft_type':'LORA','bias':'none','r':1,'lora_alpha':2}))
    a=np.array([[1,0]],dtype='<f4');b=np.array([[2],[4]],dtype='<f4')
    if fault=='shape':b=np.ones((2,2),dtype='<f4')
    if fault=='nonfinite':a[0,0]=np.inf
    entries=[('base_model.model.'+TARGET+'.lora_A.weight','F32',list(a.shape),a.tobytes()),
             ('base_model.model.'+TARGET+'.lora_B.weight','F32',list(b.shape),b.tobytes())]
    if fault=='unexpected':entries.append(('unexpected','F32',[1],struct.pack('<f',1)))
    write_tensors(adapter/'adapter_model.safetensors',entries)
    files={p.name:{'size':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in base.iterdir()}
    return base,adapter,files


class CompileCoderTests(unittest.TestCase):
    def test_bf16_rounds_halfway_values_to_even_and_rejects_nonfinite(self):
        values=np.array([0x3f808000,0x3f818000,0xbf808000,0xbf818000],dtype='<u4').view('<f4')
        self.assertEqual(to_bf16(values),struct.pack('<4H',0x3f80,0x3f82,0xbf80,0xbf82))
        for value in [np.nan,np.inf,np.finfo(np.float32).max]:
            with self.subTest(value=value),self.assertRaises(CoderError):to_bf16([value])

    @unittest.skipUnless(sys.platform.startswith('linux'),'Atomic checkpoint publication requires Linux')
    def test_sharded_compile_matches_golden_matrix_and_preserves_every_other_byte(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);base,adapter,files=fixture(root);output=root/'compiled'
            record=compile_checkpoint(base,adapter,[TARGET],files,output)
            header,start=read_header(output/'one.safetensors');a,b=header[TARGET+'.weight']['data_offsets']
            self.assertEqual((output/'one.safetensors').read_bytes()[start+a:start+b],
                             struct.pack('<4H',0x40a0,0x4000,0x4130,0x4080))
            self.assertTrue(record['verification']['all_unedited_bytes_preserved'])
            self.assertEqual((base/'two.safetensors').read_bytes(),(output/'two.safetensors').read_bytes())
            with self.assertRaises(CoderError):compile_checkpoint(base,adapter,[TARGET],files,output)
            self.assertEqual(verify_compiled(base,output,record)['edited_tensors'],1)

    @unittest.skipUnless(sys.platform.startswith('linux'),'Atomic checkpoint publication requires Linux')
    def test_bad_adapters_never_publish_or_leave_staging_directories(self):
        for fault in ['shape','nonfinite','unexpected']:
            with self.subTest(fault=fault),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);base,adapter,files=fixture(root,fault);output=root/'compiled'
                with self.assertRaises(CoderError):compile_checkpoint(base,adapter,[TARGET],files,output)
                self.assertFalse(output.exists())
                self.assertFalse(list(root.glob('.compiled.building-*')))
                self.assertTrue(all(hashlib.sha256((base/n).read_bytes()).hexdigest()==v['sha256'] for n,v in files.items()))

    @unittest.skipUnless(sys.platform.startswith('linux'),'Atomic checkpoint publication requires Linux')
    def test_unchanged_region_tamper_fails_even_if_whole_file_digest_is_rewritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);base,adapter,files=fixture(root);output=root/'compiled'
            record=compile_checkpoint(base,adapter,[TARGET],files,output)
            header,start=read_header(output/'one.safetensors')
            offset=start+header['untouched.norm']['data_offsets'][0]
            data=bytearray((output/'one.safetensors').read_bytes());data[offset]^=1
            (output/'one.safetensors').write_bytes(data)
            record['output_file_sha256']['one.safetensors']=hashlib.sha256(data).hexdigest()
            with self.assertRaisesRegex(CoderError,'Unedited bytes'):
                verify_compiled(base,output,record)


if __name__=='__main__':unittest.main()
