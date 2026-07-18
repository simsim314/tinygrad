import hashlib, unittest
import numpy as np

from tinygrad import Tensor, dtypes
from tinygrad.llm.gguf import _float32_to_fp16_bits, _fp16_times_int_to_fp16_bits, ggml_data_to_tensor


class TestDeterministicQ6Load(unittest.TestCase):
  def test_integer_float32_to_fp16_matches_reference(self):
    rng=np.random.default_rng(789)
    source=rng.integers(0,2**32,100000,dtype=np.uint32)
    got=_float32_to_fp16_bits(Tensor(source)).numpy()
    with np.errstate(over="ignore",invalid="ignore"):
      reference=source.view(np.float32).astype(np.float16).view(np.uint16)
    finite_nonzero=(((source >> 23) & 0xff) != 0xff) & ((source & 0x7fffffff) != 0)
    np.testing.assert_equal(got[finite_nonzero],reference[finite_nonzero])
    special=np.array([0,0x80000000,0x7f800000,0xff800000,0x7fc00001,1,0x33800000,0x33000000],dtype=np.uint32)
    np.testing.assert_equal(_float32_to_fp16_bits(Tensor(special)).numpy(),
                            np.array([0,0,0x7c00,0xfc00,0x7e00,0,1,0],dtype=np.uint16))

  def test_float32_loader_decodes_on_cpu_then_uploads_fp16(self):
    source=np.array([0x3f800000,0xbf800000,0x33800000,0x33000000,0x477fe000,0x80000000],dtype=np.uint32)
    raw=Tensor(source.view(np.uint8),dtype=dtypes.uint8,device="CPU")
    decoded=ggml_data_to_tensor(raw,len(source),0,deterministic_f32_fp16=True).contiguous().realize()
    self.assertEqual(decoded.dtype,dtypes.float16)
    expected=np.array([0x3c00,0xbc00,0x0001,0x0000,0x7bff,0x0000],dtype=np.uint16)
    np.testing.assert_equal(decoded.bitcast(dtypes.uint16).numpy(),expected)
    np.testing.assert_equal(decoded.to("CUDA").bitcast(dtypes.uint16).numpy(),expected)

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
