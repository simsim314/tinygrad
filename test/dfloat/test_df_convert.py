import unittest
from tinygrad import Tensor, dtypes
from extra.dfloat import tensor_to_df16, df16_checksum

class TestDFConvert(unittest.TestCase):
  def test_ieee_sources(self):
    expected=[0,65536,98304,-147456,1,-1,2147483647,-2147483648]
    values=[0.0,1.0,1.5,-2.25,2**-17,-2**-17,40000.0,-40000.0]
    for dtype in (dtypes.float16,dtypes.float32,dtypes.float64):
      got=tensor_to_df16(Tensor(values,dtype=dtype),device="CUDA").bitcast(dtypes.int32).numpy().tolist()
      self.assertEqual(got,expected)

  def test_checksum_is_raw_bits(self):
    a=tensor_to_df16(Tensor([1.0,-2.0],dtype=dtypes.float32),device="CUDA")
    b=tensor_to_df16(Tensor([1.0,-2.0],dtype=dtypes.float16),device="CUDA")
    self.assertEqual(df16_checksum(a),df16_checksum(b))

if __name__ == "__main__": unittest.main()
