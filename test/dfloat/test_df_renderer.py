import unittest
from tinygrad.dtype import dtypes
from tinygrad.helpers import Target
from tinygrad.renderer.cstyle import CUDARenderer
from tinygrad.uop.ops import UOp, Ops

class TestDFRenderer(unittest.TestCase):
  def test_df16_add_uses_helper(self):
    r=CUDARenderer(Target(device="CUDA", renderer="CUDA", arch="sm_61", interface="MOCK"))
    a,b=UOp.const(dtypes.df16,1.5),UOp.const(dtypes.df16,-2.0)
    src=r.render([a,b,UOp(Ops.ADD,dtypes.df16,(a,b))])
    kernel=src[src.index('extern "C"'):]
    self.assertIn("df16_add(98304,-131072)",kernel)
    self.assertNotIn("98304+-131072",kernel)

  def test_unsupported_never_falls_back(self):
    r=CUDARenderer(Target(device="CUDA", renderer="CUDA", arch="sm_61", interface="MOCK"))
    a=UOp.const(dtypes.df16,1.0)
    with self.assertRaisesRegex(RuntimeError, "native floating fallback is forbidden"):
      r.render([a,UOp(Ops.TRUNC,dtypes.df16,(a,))])

  def test_exp2_uses_integer_lut_helper(self):
    r=CUDARenderer(Target(device="CUDA", renderer="CUDA", arch="sm_61", interface="MOCK"))
    a=UOp.const(dtypes.df16,1.0)
    src=r.render([a,UOp(Ops.EXP2,dtypes.df16,(a,))])
    self.assertIn("df16_exp2(65536)",src[src.index('extern "C"'):])
    self.assertIn("dft_exp_f3",src)

if __name__ == "__main__": unittest.main()
