import unittest
from tinygrad import Tensor, dtypes
from extra.dfloat import tensor_to_df16, df16_checksum, convert_state_dict_df16, convert_state_dict_storage, precompute_freqs_cis_df16, DF16Linear, DF16Embedding

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

  def test_tied_weights_remain_tied(self):
    weight=Tensor([1.0,-2.0],dtype=dtypes.float32)
    got=convert_state_dict_df16({"embedding":weight,"output":weight},device="CUDA")
    self.assertIs(got["embedding"],got["output"])

  def test_streamed_fp16_weight_classes(self):
    lin=DF16Linear(2,2,bias=False)
    lin.weight=Tensor([[1.0,-2.0],[0.5,0.25]],dtype=dtypes.float16,device="CUDA")
    x=tensor_to_df16(Tensor([[1.5,-0.5]],dtype=dtypes.float32),device="CUDA")
    self.assertEqual(lin(x).bitcast(dtypes.int32).numpy().tolist(),[[163840,40960]])
    emb=DF16Embedding(3,2)
    emb.weight=Tensor([[1.0,2.0],[-1.0,0.5],[3.0,-2.0]],dtype=dtypes.float16,device="CUDA")
    self.assertEqual(emb(Tensor([2,0],dtype=dtypes.int32,device="CUDA")).bitcast(dtypes.int32).numpy().tolist(),[[196608,-131072],[65536,131072]])

  def test_compact_storage_progress_path(self):
    shared=Tensor([1.0,-2.0],dtype=dtypes.float32)
    got=convert_state_dict_storage({"layers.0.a":shared,"layers.0.b":shared,"layers.1.a":Tensor([3.0])},
                                   dtype=dtypes.float16,device="CUDA",verbose=True)
    self.assertIs(got["layers.0.a"],got["layers.0.b"])
    self.assertEqual(got["layers.1.a"].dtype,dtypes.float16)

  def test_integer_only_rope_table(self):
    expected=[[[65536,0],[65536,0],[65536,0],[65536,0]],
              [[35409,55147],[65209,6541],[65533,655],[65536,66]],
              [[-27273,59592],[64230,13017],[65523,1311],[65536,131]],
              [[-64880,9248],[62610,19364],[65507,1966],[65536,197]]]
    for _ in range(3):
      got=precompute_freqs_cis_df16(8,4,10000,"CUDA").bitcast(dtypes.int32).numpy().reshape(4,4,2).tolist()
      self.assertEqual(got,expected)

if __name__ == "__main__": unittest.main()
