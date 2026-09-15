import sys
from pathlib import Path
import unittest
import json
import tempfile
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from hopper_fp8_export import encode_e4m3, pack_weight, export

class ExportTest(unittest.TestCase):
    def test_literal_codes_and_ties(self):
        x=np.array([0.,-0.,1.,-1.,448.,500.,2**-9,1.0625,1.1875],np.float32)
        self.assertEqual(encode_e4m3(x).tolist(),[0,128,56,184,126,126,1,56,58])
    def test_reject_nonfinite(self):
        for x in (float('nan'),float('inf')):
            with self.assertRaises(ValueError):encode_e4m3(np.array([x]))
    def test_column_major_kxn_and_scale(self):
        x=np.array([[448,0],[-448,224]],np.float32)
        packed,scale=pack_weight(x)
        self.assertEqual(scale,1.)
        self.assertEqual(packed.tolist(),[126,254,0,118])
    def test_zero_scale_is_finite(self):
        packed,scale=pack_weight(np.zeros((16,8),np.float32))
        self.assertEqual(scale,1.)
        self.assertFalse(packed.any())
    def test_bundle_stores_one_representation_and_rejects_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'source';source.mkdir();out=root/'out'
            (source/'manifest.json').write_text(json.dumps(dict(backend='piece_transformer',
                dtype='fp16',d_model=256,ff_dim=1024,num_layers=4,seq_len=57,
                output_dim=24,activation='relu')))
            for i in range(3):
                np.ones((256,768),dtype='<f2').tofile(source/f'block{i}_attn_qkv_weight_hxk.fp16')
            np.ones(256,dtype='<f2').tofile(source/'block0_ln1_gamma.fp16')
            for name in ('piece_positions.u16','piece_mask.u8','piece_types.u8'):
                (source/name).write_bytes(b'\x01\x00')
            result=export(source,out)
            self.assertEqual(len(result['files']),7)
            for name in ('piece_positions.u16','piece_mask.u8','piece_types.u8'):
                self.assertEqual((out/name).read_bytes(),(source/name).read_bytes())
            self.assertFalse((out/'block0_attn_qkv_weight_hxk.fp16').exists())
            self.assertEqual((out/'block0_attn_qkv_weight_hxk.e4m3').stat().st_size,256*768)
            self.assertEqual((out/'block0_ln1_gamma.fp16').read_bytes(),(source/'block0_ln1_gamma.fp16').read_bytes())
            with self.assertRaises(FileExistsError):export(source,out)

if __name__=='__main__':unittest.main()
