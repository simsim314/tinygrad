import math, unittest

from tinygrad import Context, Tensor, dtypes
from tinygrad.nn.state import get_state_dict, load_state_dict
from extra.dfloat import convert_state_dict_df16, precompute_freqs_cis_df16
from extra.models.llama import Transformer


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

  def test_df16_transcendentals(self):
    values=[-655360,-589824,-98305,-98304,-65537,-65536,-1,0,1,65535,65536,98304,589824,655360]
    cpu,gpu=df16_raw(values,"CPU"),df16_raw(values,"CUDA")
    self.assert_same(cpu.exp2(),gpu.exp2(),dtypes.int32)
    positive=df16_raw([1,32768,65535,65536,65537,98304,131072,1048576,2147483647],"CPU")
    positive_gpu=df16_raw([1,32768,65535,65536,65537,98304,131072,1048576,2147483647],"CUDA")
    self.assert_same(positive.log2(),positive_gpu.log2(),dtypes.int32)
    self.assert_same(cpu.sin(),gpu.sin(),dtypes.int32)

  def test_df32_transcendentals(self):
    unit=1<<32
    values=[-16*unit,-10*unit,-unit-1,-unit,-1,0,1,unit-1,unit,unit+(unit//2),10*unit,16*unit]
    cpu,gpu=df32_raw(values,"CPU"),df32_raw(values,"CUDA")
    self.assert_same(cpu.exp2(),gpu.exp2(),dtypes.int64)
    positive=df32_raw([1,unit//2,unit-1,unit,unit+1,2*unit,16*unit,(1<<63)-1],"CPU")
    positive_gpu=df32_raw([1,unit//2,unit-1,unit,unit+1,2*unit,16*unit,(1<<63)-1],"CUDA")
    self.assert_same(positive.log2(),positive_gpu.log2(),dtypes.int64)
    self.assert_same(cpu.sin(),gpu.sin(),dtypes.int64)

  def test_one_block_llama_raw_logits(self):
    args=dict(dim=8,hidden_dim=16,n_heads=2,n_layers=1,norm_eps=1e-5,vocab_size=32,
              n_kv_heads=2,max_context=8,jit=False,disable_kv_cache=True)
    with Context(DEV="CPU"):
      Tensor.manual_seed(1234)
      base=Transformer(**args)
      state={k:v for k,v in get_state_dict(base).items() if k != "freqs_cis"}

    def run(device):
      with Context(DEV=device): model=Transformer(**args)
      load_state_dict(model,convert_state_dict_df16(state,device=device),verbose=False,strict=False)
      model.freqs_cis=precompute_freqs_cis_df16(4,16,10000,device)
      tokens=Tensor([[1,7,3]],dtype=dtypes.int32,device=device)
      return model.forward(tokens,0,math.nan,0,0.0,0.0,0.0).realize().bitcast(dtypes.int32).numpy().tolist()

    self.assertEqual(run("CPU"),run("CUDA"))


if __name__ == "__main__": unittest.main()
