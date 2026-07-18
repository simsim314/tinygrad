import unittest

from tinygrad import Tensor, dtypes


def df16_raw(values, device):
  return Tensor(values,dtype=dtypes.int32,device=device).bitcast(dtypes.df16)


def df32_raw(values, device):
  return Tensor(values,dtype=dtypes.int64,device=device).bitcast(dtypes.df32)


class TestDFCPUCUDA(unittest.TestCase):
  def assert_same(self, cpu, cuda, storage_dtype):
    self.assertEqual(cpu.bitcast(storage_dtype).numpy().tolist(),cuda.bitcast(storage_dtype).numpy().tolist())

  def test_df16_basic_and_saturation(self):
    a=[0,1,-1,65536,-65536,2147483647,-2147483648,1073741824,-1073741824]
    b=[1,-1,65536,98304,-147456,2147483647,-2147483648,-196608,196608]
    ca,cb,ga,gb=df16_raw(a,"CPU"),df16_raw(b,"CPU"),df16_raw(a,"CUDA"),df16_raw(b,"CUDA")
    for cop,gop in ((ca+cb,ga+gb),(ca-cb,ga-gb),(ca*cb,ga*gb),(ca/(cb+cb.const_like(1)),ga/(gb+gb.const_like(1)))):
      self.assert_same(cop,gop,dtypes.int32)
    self.assert_same(ca.square().sqrt(),ga.square().sqrt(),dtypes.int32)

  def test_df32_basic_and_saturation(self):
    lim=(1<<63)-1
    a=[0,1,-1,1<<32,-(1<<32),lim,-lim-1,1<<61,-(1<<61)]
    b=[1,-1,1<<32,3<<31,-5<<30,lim,-lim-1,-3<<32,3<<32]
    ca,cb,ga,gb=df32_raw(a,"CPU"),df32_raw(b,"CPU"),df32_raw(a,"CUDA"),df32_raw(b,"CUDA")
    for cop,gop in ((ca+cb,ga+gb),(ca-cb,ga-gb),(ca*cb,ga*gb),(ca/(cb+cb.const_like(1)),ga/(gb+gb.const_like(1)))):
      self.assert_same(cop,gop,dtypes.int64)
    self.assert_same(ca.square().sqrt(),ga.square().sqrt(),dtypes.int64)

  def test_fixed_reduction_tree(self):
    values=[2147483647,2147483647,-2147483648,-2147483648,65536,-65536,98304,-147456,1,-1]
    cpu,gpu=df16_raw(values,"CPU"),df16_raw(values,"CUDA")
    self.assert_same(cpu.sum(),gpu.sum(),dtypes.int32)
    self.assert_same(cpu.max(),gpu.max(),dtypes.int32)


if __name__ == "__main__": unittest.main()
