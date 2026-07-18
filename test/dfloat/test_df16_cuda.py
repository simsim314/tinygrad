import os, unittest
from tinygrad import Tensor, dtypes, nn
from tinygrad.dtype import least_upper_dtype, sum_acc_dtype, to_storage_scalar

DEVICE = os.getenv("DFLOAT_TEST_DEVICE", "CUDA")

def raw16(values, shape=None):
  t = Tensor(values, dtype=dtypes.int32, device=DEVICE).bitcast(dtypes.df16)
  return t.reshape(shape) if shape is not None else t

def bits16(t): return t.bitcast(dtypes.int32).numpy().tolist()
def raw32(values, shape=None):
  t = Tensor(values, dtype=dtypes.int64, device=DEVICE).bitcast(dtypes.df32)
  return t.reshape(shape) if shape is not None else t
def bits32(t): return t.bitcast(dtypes.int64).numpy().tolist()

class TestDFloatDType(unittest.TestCase):
  def test_layout_and_promotion(self):
    self.assertEqual(dtypes.df16.itemsize, 4)
    self.assertEqual(dtypes.df32.itemsize, 8)
    self.assertEqual(least_upper_dtype(dtypes.df16, dtypes.df16), dtypes.df16)
    self.assertEqual(least_upper_dtype(dtypes.df16, dtypes.df32), dtypes.df32)
    self.assertEqual(sum_acc_dtype(dtypes.df16), dtypes.df32)
    with self.assertRaises(TypeError): least_upper_dtype(dtypes.df16, dtypes.float32)

  def test_constant_storage(self):
    self.assertEqual(to_storage_scalar(1.5, dtypes.df16), 98304)
    t = Tensor([0.0, 1.0, 1.5, -2.25], dtype=dtypes.df16, device=DEVICE)
    self.assertEqual(bits16(t), [0, 65536, 98304, -147456])
    self.assertEqual(to_storage_scalar(2**-17, dtypes.df16), 1)
    self.assertEqual(to_storage_scalar(-(2**-17), dtypes.df16), -1)
    self.assertEqual(to_storage_scalar(2**-33, dtypes.df32), 1)

class TestDFloatCUDA(unittest.TestCase):
  def test_elementwise_raw_bits(self):
    a=raw16([65536,98304,-131072,2147483647]); b=raw16([131072,-131072,32768,1])
    self.assertEqual(bits16(a+b), [196608,-32768,-98304,2147483647])
    self.assertEqual(bits16(a-b), [-65536,229376,-163840,2147483646])
    self.assertEqual(bits16(a*b), [131072,-196608,-65536,32768])
    self.assertEqual(bits16(a/b), [32768,-49152,-262144,2147483647])

  def test_sqrt(self):
    self.assertEqual(bits16(raw16([0,65536,262144,589824]).sqrt()), [0,65536,131072,196608])
    expected=[0,4294967296,6074000999,8589934592,12884901888,199032864766430]
    for _ in range(20): self.assertEqual(bits32(raw32([0,4294967296,8589934592,17179869184,38654705664,9223372036854775807]).sqrt()),expected)

  def test_exp2_raw_bits_and_repeatability(self):
    x=raw16([-131072,-65536,-32768,0,32768,65536,196608,917504])
    expected=None
    for _ in range(20):
      got=bits16(x.exp2())
      if expected is None: expected=got
      self.assertEqual(got,expected)
    self.assertEqual(expected,[16384,32768,46340,65536,92682,131072,524286,1073720349])

  def test_wide_matmul_and_repeatability(self):
    a=raw16([65536,131072,-65536,32768,98304,262144],(2,3))
    b=raw16([131072,65536,65536,-65536,32768,131072],(3,2))
    expected=[[229376,-196608],[294912,458752]]
    for _ in range(20): self.assertEqual(bits16(a@b), expected)

  def test_rmsnorm_class_uses_df32_core(self):
    norm=nn.RMSNorm(4,eps=1e-8)
    norm.weight=raw16([65536]*4)
    x=raw16([65536,131072,-65536,32768,98304,-32768,196608,65536],(2,4))
    expected=[[52429,104858,-52429,26214],[55609,-18536,111218,37073]]
    for _ in range(20): self.assertEqual(bits16(norm(x)),expected)

if __name__ == "__main__": unittest.main()
