"""Factor permutations must equal the corresponding independent dense transform."""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from export_coder_lora_gguf import reorder_factors
from coder_common import CoderError


def dense_reorder(tensor,axis,k,v,head):
    shape=list(tensor.shape)
    split=shape[:axis]+[k,v,head]+shape[axis+1:]
    permutation=list(range(len(split)))
    permutation[axis],permutation[axis+1]=permutation[axis+1],permutation[axis]
    return tensor.reshape(split).transpose(permutation).copy().reshape(shape)


class LoraPermutationTests(unittest.TestCase):
    def test_row_and_column_factors_match_dense_permutations(self):
        a=(np.arange(36,dtype=np.float32).reshape(3,12)-17)/8
        b=(np.arange(36,dtype=np.float32).reshape(12,3)-11)/4
        for axis in [0,1,-1,-2]:
            with self.subTest(axis=axis):
                changed_a,changed_b=reorder_factors(a,b,axis,2,3,2,dense_reorder)
                expected=dense_reorder(b@a,axis%2,2,3,2)
                np.testing.assert_array_equal(changed_b@changed_a,expected)

    def test_invalid_axis_and_nonmatrix_factors_fail(self):
        a=np.ones((3,12),dtype=np.float32);b=np.ones((12,3),dtype=np.float32)
        with self.assertRaises(CoderError):reorder_factors(a,b,2,2,3,2,dense_reorder)
        with self.assertRaises(CoderError):reorder_factors(a[None],b[None],0,2,3,2,dense_reorder)


if __name__=='__main__':unittest.main()
