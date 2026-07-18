import hashlib, unittest
import numpy as np

from tinygrad import Tensor, dtypes
from tinygrad.llm.gguf import _fp16_times_int_to_fp16_bits, ggml_data_to_tensor


class TestDeterministicQ6Load(unittest.TestCase):
  def test_integer_fp16_multiplier_matches_reference(self):
    rng=np.random.default_rng(123)
    finite=rng.integers(0,0x7c00,20000,dtype=np.uint16)
    # Explicitly cover signed zero, subnormal, normal, halfway, and overflow boundaries.
    scales=np.concatenate((np.array([0,0x8000,1,0x03ff,0x0400,0x3c00,0x3c01,0x7bff],dtype=np.uint16),finite))
    multipliers=np.concatenate((np.array([0,-1,1,3,2047,4096,-4096,2],dtype=np.int32),
                                rng.integers(-4096,4097,len(finite),dtype=np.int32)))
    got=_fp16_times_int_to_fp16_bits(Tensor(scales),Tensor(multipliers)).numpy()
    with np.errstate(over="ignore",invalid="ignore"):
      reference=(scales.view(np.float16)*multipliers.astype(np.float32)).astype(np.float16).view(np.uint16)
    np.testing.assert_equal(got,reference)

  def test_q6_cpu_decode_then_upload_preserves_canonical_fp16(self):
    rng=np.random.default_rng(456)
    blocks=rng.integers(0,256,size=(3,210),dtype=np.uint8)
    blocks[:,-2:]=np.array([0x00,0x3c],dtype=np.uint8)  # exact scale 1.0
    raw=Tensor(blocks.flatten(),dtype=dtypes.uint8,device="CPU")
    deterministic=ggml_data_to_tensor(raw,3*256,14,deterministic_q6_fp16=True).contiguous().realize()
    native=ggml_data_to_tensor(raw,3*256,14).cast(dtypes.float16).contiguous().realize()
    expected=deterministic.bitcast(dtypes.uint16).numpy()
    native_bits=native.bitcast(dtypes.uint16).numpy()
    # The legacy two-multiply route can produce -0 depending on which integer factor is zero.
    # Q6 canonical storage deliberately normalizes every exact zero weight to +0.
    np.testing.assert_equal(np.bitwise_and(expected,0x7fff),np.bitwise_and(native_bits,0x7fff))
    uploaded=deterministic.to("CUDA").contiguous().realize().bitcast(dtypes.uint16).numpy()
    np.testing.assert_equal(uploaded,expected)
    self.assertEqual(hashlib.sha256(expected.astype("<u2",copy=False).tobytes()).hexdigest(),
                     "e3cd97eac8ad48beda79a682a86d30ea43d9bf3c0562c4bb604cdd88d7e9c4c8")


if __name__ == "__main__": unittest.main()
