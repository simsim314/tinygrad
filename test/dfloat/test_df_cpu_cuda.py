import math, unittest

from tinygrad import Context, Tensor, dtypes
from tinygrad.nn.state import get_state_dict, load_state_dict
from extra.dfloat import convert_state_dict_df16, precompute_freqs_cis_df16
from extra.dfloat_attestation import AttestationRecorder
from extra.dfloat_attestation_schema import residual_plan
from extra.dfloat_attestation_ops import attested_matmul, attested_rmsnorm, attested_softmax
from extra.dfloat_attestation_schema import matmul_plan, rmsnorm_plan, softmax_plan
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

  def test_same_tensor_witness_attestation(self):
    raw=[-2147483648,-98304,-1,0,1,65536,98304,2147483647]
    attestations=[]
    for device in ("CPU","CUDA"):
      left=df16_raw(raw,device)
      right=df16_raw(list(reversed(raw)),device)
      wide=left.cast(dtypes.df32)+right.cast(dtypes.df32)
      mask=(wide > wide.const_like(2147483647<<16)).cast(dtypes.uint8)
      out=(left+right).contiguous().realize()
      values={"left_input":left,"right_input":right,"wide_sum":wide,"saturation_mask":mask,"df16_output":out}
      recorder=AttestationRecorder(step=4)
      recorder.record_module(residual_plan("layers.0.residual"),values,
                             input_names=("left_input","right_input"),output_names=("df16_output",))
      attestations.append(recorder.json())
    self.assertEqual(attestations[0],attestations[1])

  def test_witnessed_operations_match_and_attest(self):
    results=[]
    for device in ("CPU","CUDA"):
      x=df16_raw([65536,-32768,98304,16384,-65536,131072,49152,-81920],device).reshape(2,4)
      w=df16_raw([65536,32768,-65536,16384,49152,-32768,81920,65536,16384,98304,-49152,32768],device).reshape(3,4).T
      mm_plan=matmul_plan("test.linear",4)
      mm,mm_values=attested_matmul(x,w,mm_plan)
      self.assertEqual(mm.bitcast(dtypes.int32).numpy().tolist(),x.dot(w).bitcast(dtypes.int32).numpy().tolist())
      norm_plan=rmsnorm_plan("test.norm",3)
      norm,norm_values=attested_rmsnorm(mm,df16_raw([65536,98304,32768],device),1e-5,norm_plan)
      from tinygrad import nn
      reference=nn.RMSNorm(3,1e-5)
      reference.weight=df16_raw([65536,98304,32768],device)
      self.assertEqual(norm.bitcast(dtypes.int32).numpy().tolist(),reference(mm).bitcast(dtypes.int32).numpy().tolist())
      soft_plan=softmax_plan("test.softmax",3)
      soft,soft_values=attested_softmax(norm,soft_plan)
      self.assertEqual(soft.bitcast(dtypes.int32).numpy().tolist(),norm.softmax(-1).bitcast(dtypes.int32).numpy().tolist())
      recorder=AttestationRecorder(2)
      recorder.record_module(mm_plan,mm_values,input_names=("activation_input","weight_df16"),output_names=("output_df16",))
      recorder.record_module(norm_plan,norm_values,input_names=("input_df16",),output_names=("norm_output_df16",))
      recorder.record_module(soft_plan,soft_values,input_names=("scores_df16",),output_names=("probabilities_df16",))
      results.append(recorder.json())
    self.assertEqual(results[0],results[1])


if __name__ == "__main__": unittest.main()
