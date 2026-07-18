"""Compact V2 DF attestation using GPU tensor samples and keyed XOR sketches."""
from __future__ import annotations

from dataclasses import dataclass
import functools, hashlib, struct, sys, time, unicodedata
from typing import Mapping, Sequence

from tinygrad import Device, Tensor, UOp, dtypes
from tinygrad.device import Buffer
from tinygrad.engine.realize import run_linear
from tinygrad.uop import Ops
from tinygrad.uop.ops import KernelInfo

from extra.dfloat_attestation import (DTYPE_ITEMSIZE, ZERO_SHA256, RootReference, TensorDType, TensorRole,
  canonical_json_bytes, ordered_root, require_digest, sha256_frame, tensor_dtype_id)


@dataclass(frozen=True)
class DeviceDigest:
  device: str
  buffer: Buffer


def _device_digest_bytes(digest:bytes,device:str) -> DeviceDigest:
  words=struct.unpack(">8I",require_digest(digest,"digest"))
  return DeviceDigest(device,Buffer(device,8,dtypes.uint32,initial_value=struct.pack("<8I",*words)))


def _device_digest_to_bytes(digest:DeviceDigest) -> bytes:
  return struct.pack(">8I",*(int(word) for word in digest.buffer.numpy().tolist()))


def _low32_uop(value:UOp,dtype) -> UOp:
  if dtype in (dtypes.df16,dtypes.int32): return value.bitcast(dtypes.uint32)
  if dtype == dtypes.df32: return value.bitcast(dtypes.uint64).bitwise_and(0xffffffff).cast(dtypes.uint32)
  if dtype == dtypes.float16: return value.bitcast(dtypes.uint16).cast(dtypes.uint32)
  if dtype == dtypes.uint8: return value.cast(dtypes.uint32)
  raise TypeError(f"in-kernel witness does not support {dtype}")


def instrumented_cuda_realize_many(tensors:Sequence[Tensor],selector:DeviceDigest,ordinals:Sequence[int]) -> list[DeviceDigest]:
  """Realize tensors together after adding eight witness stores to each tensor's producer CALL."""
  if len(tensors) != len(ordinals): raise ValueError("tensor/ordinal count mismatch")
  if not tensors: return []
  tensors=[tensor.contiguous() for tensor in tensors]
  linear,var_vals=tensors[0].linear_with_vars(*tensors[1:])
  witnesses=[DeviceDigest(selector.device,Buffer(selector.device,8,dtypes.uint32,preallocate=True)) for _ in tensors]
  targets={tensor.uop.buffer:(tensor,ordinal,witness) for tensor,ordinal,witness in zip(tensors,ordinals,witnesses)}
  replacement:dict[UOp,UOp]={}
  found:set[Buffer]=set()
  for call in linear.src:
    if call.op is not Ops.CALL or call.src[0].op is not Ops.SINK: continue
    sink=call.src[0]
    extra_args:list[UOp]=[]
    store_replacements:dict[UOp,UOp]={}
    for output_slot,actual in enumerate(call.src[1:]):
      if (target:=targets.get(actual.buffer)) is None: continue
      tensor,ordinal,witness=target
      output_param=next((node for node in sink.toposort() if node.op is Ops.PARAM and node.arg.slot == output_slot),None)
      if output_param is None: continue
      witness_slot=len(call.src)-1+len(extra_args)
      selector_slot=witness_slot+1
      witness_param=UOp.placeholder((8,),dtypes.uint32,witness_slot)
      selector_param=UOp.placeholder((8,),dtypes.uint32,selector_slot)
      for store in (node for node in sink.toposort() if node.op is Ops.STORE and node.src[0].src[0] is output_param):
        index,value=store.src[0].src[1],_low32_uop(store.src[1],tensor.dtype)
        taps=[witness_param[lane].store(value,index.cast(dtypes.uint32).eq((selector_param[lane]+ordinal)%tensor.numel())) for lane in range(8)]
        store_replacements[store]=UOp.group(store,*taps)
      extra_args.extend((UOp.from_buffer(witness.buffer),UOp.from_buffer(selector.buffer)))
      found.add(tensor.uop.buffer)
    if not store_replacements: continue
    # Stores are direct children of END barriers in a linearized kernel. Rewrite
    # those barrier children explicitly; generic substitution does not cross END.
    new_sink=sink.replace(src=tuple(end.replace(src=tuple(store_replacements.get(node,node) for node in end.src))
                                    if end.op is Ops.END else store_replacements.get(end,end) for end in sink.src))
    replacement[call]=call.replace(src=(new_sink,*call.src[1:],*extra_args))
  missing=set(targets)-found
  if missing: raise RuntimeError(f"could not locate {len(missing)} producer store(s) for in-kernel witnesses")
  # CALLs are direct children of LINEAR. UOp.substitute intentionally does not enter
  # CALLs, so replace those children explicitly while preserving launch order.
  run_linear(linear.replace(src=tuple(replacement.get(call,call) for call in linear.src)),var_vals=var_vals)
  return witnesses


def instrumented_cuda_realize(tensor:Tensor,selector:DeviceDigest,ordinal:int) -> tuple[Tensor,DeviceDigest]:
  """Single-tensor compatibility wrapper around the batched producer instrumentation."""
  tensor=tensor.contiguous()
  return tensor,instrumented_cuda_realize_many([tensor],selector,[ordinal])[0]


def _digest_words(digest:bytes, bits:int) -> tuple[int, ...]:
  digest=require_digest(digest,"selector_digest")
  return struct.unpack("<8I" if bits == 32 else "<4Q",digest)


_SHA256_INITIAL=(0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19)
_SHA256_ROUND=(
  0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
  0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
  0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
  0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
  0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
  0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
  0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
  0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2)


def _rotr32(value, count:int):
  return value.rshift(count).bitwise_or(value.lshift(32-count))


def _sha256_compress(state:list, block:list) -> list:
  """One standard SHA-256 compression block over big-endian uint32 message words."""
  schedule=list(block)
  for index in range(16,64):
    s0=_rotr32(schedule[index-15],7).bitwise_xor(_rotr32(schedule[index-15],18)).bitwise_xor(schedule[index-15].rshift(3))
    s1=_rotr32(schedule[index-2],17).bitwise_xor(_rotr32(schedule[index-2],19)).bitwise_xor(schedule[index-2].rshift(10))
    schedule.append(schedule[index-16]+s0+schedule[index-7]+s1)
  a,b,c,d,e,f,g,h=state
  for index in range(64):
    big1=_rotr32(e,6).bitwise_xor(_rotr32(e,11)).bitwise_xor(_rotr32(e,25))
    choose=e.bitwise_and(f).bitwise_xor(e.bitwise_not().bitwise_and(g))
    temporary1=h+big1+choose+_SHA256_ROUND[index]+schedule[index]
    big0=_rotr32(a,2).bitwise_xor(_rotr32(a,13)).bitwise_xor(_rotr32(a,22))
    majority=a.bitwise_and(b).bitwise_xor(a.bitwise_and(c)).bitwise_xor(b.bitwise_and(c))
    temporary2=big0+majority
    h,g,f,e,d,c,b,a=g,f,e,d+temporary1,c,b,a,temporary1+temporary2
  return [original+updated for original,updated in zip(state,(a,b,c,d,e,f,g,h))]


def digest_tensor(digest:bytes, device:str) -> Tensor:
  """Upload conventional SHA-256 digest bytes as eight big-endian uint32 state words."""
  return Tensor(struct.unpack(">8I",require_digest(digest,"digest")),dtype=dtypes.uint32,device=device).realize()


def digest_tensor_bytes(digest:Tensor) -> bytes:
  words=digest.cast(dtypes.uint32).contiguous().realize().to("CPU").tolist()
  return struct.pack(">8I",*(int(word) for word in words))


@functools.lru_cache(maxsize=None)
def _sha256_one_thread_kernel(out:UOp, prior:UOp, values:UOp, tag:UOp) -> UOp:
  """Build the portable scalar SHA kernel once; custom-kernel placeholders are structural cache keys."""
  out,prior,values,tag=(x.flatten() for x in (out,prior,values,tag))
  initial=[prior[0].const_like(value) for value in _SHA256_INITIAL]
  first=[prior[i] for i in range(8)]+[values[i] for i in range(8)]
  second=[tag[i] for i in range(8)]+[prior[0].const_like(0x80000000)]+[prior[0].const_like(0)]*6+[prior[0].const_like(96*8)]
  state=_sha256_compress(_sha256_compress(initial,first),second)
  stores=UOp.group(*(out[i].store(state[i]) for i in range(8)))
  return stores.sink(arg=KernelInfo(name="dfat_sha256_one_thread",opts_to_apply=()))


_CUDA_SHA256_SOURCE=r'''
extern "C" __device__ __forceinline__ unsigned int rotr32(unsigned int x, unsigned int n) { return (x >> n) | (x << (32-n)); }
extern "C" __global__ void dfat_sha256_one_thread(unsigned int *out, const unsigned int *prior,
                                                   const unsigned int *sketch, const unsigned int *tag) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  const unsigned int initial[8]={0x6a09e667u,0xbb67ae85u,0x3c6ef372u,0xa54ff53au,
    0x510e527fu,0x9b05688cu,0x1f83d9abu,0x5be0cd19u};
  const unsigned int k[64]={
    0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
    0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
    0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
    0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
    0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
    0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
    0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
    0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u};
  unsigned int state[8]; for (int i=0;i<8;i++) state[i]=initial[i];
  for (int block=0;block<2;block++) {
    unsigned int w[64];
    if (block==0) { for (int i=0;i<8;i++) w[i]=prior[i]; for (int i=0;i<8;i++) w[i+8]=sketch[i]; }
    else {
      for (int i=0;i<8;i++) w[i]=tag[i]; w[8]=0x80000000u;
      for (int i=9;i<15;i++) w[i]=0u; w[15]=768u;
    }
    for (int i=16;i<64;i++) {
      unsigned int s0=rotr32(w[i-15],7)^rotr32(w[i-15],18)^(w[i-15]>>3);
      unsigned int s1=rotr32(w[i-2],17)^rotr32(w[i-2],19)^(w[i-2]>>10);
      w[i]=w[i-16]+s0+w[i-7]+s1;
    }
    unsigned int a=state[0],b=state[1],c=state[2],d=state[3],e=state[4],f=state[5],g=state[6],h=state[7];
    for (int i=0;i<64;i++) {
      unsigned int s1=rotr32(e,6)^rotr32(e,11)^rotr32(e,25);
      unsigned int ch=(e&f)^((~e)&g), t1=h+s1+ch+k[i]+w[i];
      unsigned int s0=rotr32(a,2)^rotr32(a,13)^rotr32(a,22);
      unsigned int maj=(a&b)^(a&c)^(b&c), t2=s0+maj;
      h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    state[0]+=a; state[1]+=b; state[2]+=c; state[3]+=d;
    state[4]+=e; state[5]+=f; state[6]+=g; state[7]+=h;
  }
  for (int i=0;i<8;i++) out[i]=state[i];
}
'''


@functools.lru_cache(maxsize=None)
def _cuda_sha256_runtime(device_name:str):
  device=Device[device_name]
  return device.runtime("dfat_sha256_one_thread",device.compiler.compile(_CUDA_SHA256_SOURCE))


def _cuda_fused_witness_source(kind:str) -> str:
  ctype,bits,load={
    "u8":("unsigned char",32,"((unsigned int)data[index])"),
    "u16":("unsigned short",32,"((unsigned int)data[index])"),
    "u32":("unsigned int",32,"data[index]"),
    "u64":("unsigned long long",64,"data[index]"),
  }[kind]
  lanes=256//bits
  constants=",".join(f"0x{value:08x}u" for value in _SHA256_ROUND)
  selector="prior[group & 7u]" if bits == 32 else "(((unsigned long long)prior[(group & 3u)*2u])<<32)|prior[(group & 3u)*2u+1u]"
  shared_type="unsigned int" if bits == 32 else "unsigned long long"
  split="for (int i=0;i<8;i++) sketch[i]=shared_values[0][i];" if bits == 32 else \
    "for (int i=0;i<4;i++) { sketch[2*i]=(unsigned int)(shared_values[0][i]>>32); sketch[2*i+1]=(unsigned int)shared_values[0][i]; }"
  return f'''
extern "C" __device__ __forceinline__ unsigned int dfat_rotr(unsigned int x,unsigned int n) {{ return (x>>n)|(x<<(32-n)); }}
extern "C" __device__ __forceinline__ void dfat_sha(unsigned int *out,const unsigned int *prior,
                                                     const unsigned int *sketch,const unsigned int *tag) {{
  const unsigned int initial[8]={{0x6a09e667u,0xbb67ae85u,0x3c6ef372u,0xa54ff53au,0x510e527fu,0x9b05688cu,0x1f83d9abu,0x5be0cd19u}};
  const unsigned int k[64]={{{constants}}}; unsigned int state[8]; for(int i=0;i<8;i++) state[i]=initial[i];
  for(int block=0;block<2;block++) {{ unsigned int w[64];
    if(block==0) {{ for(int i=0;i<8;i++) w[i]=prior[i]; for(int i=0;i<8;i++) w[i+8]=sketch[i]; }}
    else {{ for(int i=0;i<8;i++) w[i]=tag[i]; w[8]=0x80000000u; for(int i=9;i<15;i++) w[i]=0u; w[15]=768u; }}
    for(int i=16;i<64;i++) {{ unsigned int s0=dfat_rotr(w[i-15],7)^dfat_rotr(w[i-15],18)^(w[i-15]>>3);
      unsigned int s1=dfat_rotr(w[i-2],17)^dfat_rotr(w[i-2],19)^(w[i-2]>>10); w[i]=w[i-16]+s0+w[i-7]+s1; }}
    unsigned int a=state[0],b=state[1],c=state[2],d=state[3],e=state[4],f=state[5],g=state[6],h=state[7];
    for(int i=0;i<64;i++) {{ unsigned int s1=dfat_rotr(e,6)^dfat_rotr(e,11)^dfat_rotr(e,25),ch=(e&f)^((~e)&g);
      unsigned int t1=h+s1+ch+k[i]+w[i],s0=dfat_rotr(a,2)^dfat_rotr(a,13)^dfat_rotr(a,22),maj=(a&b)^(a&c)^(b&c),t2=s0+maj;
      h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2; }}
    state[0]+=a;state[1]+=b;state[2]+=c;state[3]+=d;state[4]+=e;state[5]+=f;state[6]+=g;state[7]+=h;
  }} for(int i=0;i<8;i++) out[i]=state[i];
}}
extern "C" __global__ void dfat_fused_witness(unsigned int *out,const {ctype} *data,const unsigned int *prior,
                                               unsigned int count,unsigned int tag0,unsigned int tag1,unsigned int tag2,unsigned int tag3,
                                               unsigned int tag4,unsigned int tag5,unsigned int tag6,unsigned int tag7) {{
  __shared__ {shared_type} shared_values[256][{lanes}]; unsigned int thread=threadIdx.x;
  {shared_type} local[{lanes}]={{0}}; unsigned int groups=(count+{lanes-1}u)/{lanes}u;
  for(unsigned int group=thread;group<groups;group+=256u) {{
    {shared_type} key=({selector})^group; unsigned int offset=(unsigned int)(key&{lanes-1}u),reverse=(unsigned int)((key>>{lanes.bit_length()-1})&1u);
    for(unsigned int lane=0;lane<{lanes}u;lane++) {{ unsigned int source=reverse?((offset-lane)&{lanes-1}u):((lane-offset)&{lanes-1}u);
      unsigned int index=group*{lanes}u+source; {shared_type} value=index<count?{load}:0; local[lane]^=value; }}
  }} for(int lane=0;lane<{lanes};lane++) shared_values[thread][lane]=local[lane]; __syncthreads();
  for(unsigned int stride=128u;stride;stride>>=1) {{ if(thread<stride) for(int lane=0;lane<{lanes};lane++) shared_values[thread][lane]^=shared_values[thread+stride][lane]; __syncthreads(); }}
  if(thread==0) {{ unsigned int sketch[8],tag[8]={{tag0,tag1,tag2,tag3,tag4,tag5,tag6,tag7}}; {split} dfat_sha(out,prior,sketch,tag); }}
}}
extern "C" __global__ void dfat_sha_buffer(unsigned int *out,const unsigned int *prior,const unsigned int *sketch,
                                            unsigned int tag0,unsigned int tag1,unsigned int tag2,unsigned int tag3,
                                            unsigned int tag4,unsigned int tag5,unsigned int tag6,unsigned int tag7) {{
  if(blockIdx.x==0 && threadIdx.x==0) {{ unsigned int tag[8]={{tag0,tag1,tag2,tag3,tag4,tag5,tag6,tag7}}; dfat_sha(out,prior,sketch,tag); }}
}}
'''


@functools.lru_cache(maxsize=None)
def _cuda_fused_witness_runtime(device_name:str,kind:str):
  device=Device[device_name]
  return device.runtime("dfat_fused_witness",device.compiler.compile(_cuda_fused_witness_source(kind)))


@functools.lru_cache(maxsize=None)
def _cuda_sha_buffer_runtime(device_name:str):
  device=Device[device_name]
  return device.runtime("dfat_sha_buffer",device.compiler.compile(_cuda_fused_witness_source("u32")))


def cuda_fused_tensor_witness(previous:DeviceDigest,tensor:Tensor,tag_digest:bytes) -> DeviceDigest:
  dtype=tensor_dtype_id(tensor)
  kind={TensorDType.DF32:"u64",TensorDType.DF16:"u32",TensorDType.FLOAT16:"u16",
        TensorDType.INT32:"u32",TensorDType.UINT8:"u8"}[dtype]
  # Contiguous slices of the realized token trace are zero-copy Buffer views.
  if not tensor.uop._base_buffer_is_realized() or tensor.uop.contiguous_view_offset() is None:
    tensor=tensor.contiguous().realize()
  device_name=previous.device
  output=DeviceDigest(device_name,Buffer(device_name,8,dtypes.uint32,preallocate=True))
  buffers=(output.buffer._buf,tensor.uop.buffer.get_buf(device_name),previous.buffer._buf)
  tag_words=struct.unpack(">8I",require_digest(tag_digest,"tag_digest"))
  _cuda_fused_witness_runtime(device_name,kind)(*buffers,vals=(int(tensor.numel()),*tag_words),global_size=(1,1,1),local_size=(256,1,1),wait=False)
  return output


def cuda_sha_buffer_witness(previous:DeviceDigest,sketch:DeviceDigest,tag_digest:bytes) -> DeviceDigest:
  if previous.device != sketch.device: raise ValueError("digest buffers must share a device")
  output=DeviceDigest(previous.device,Buffer(previous.device,8,dtypes.uint32,preallocate=True))
  tag_words=struct.unpack(">8I",require_digest(tag_digest,"tag_digest"))
  _cuda_sha_buffer_runtime(previous.device)(output.buffer._buf,previous.buffer._buf,sketch.buffer._buf,vals=tag_words,
                                            global_size=(1,1,1),local_size=(1,1,1),wait=False)
  return output


def sha256_device_witness(previous:Tensor, sketch:Tensor, tag_digest:Tensor) -> Tensor:
  """Compute SHA256(previous || sketch || tag_digest), keeping its state on the same device.

  All arguments contain eight conventional big-endian SHA/message uint32 words. The 96-byte
  message always occupies two SHA-256 blocks including the standard FIPS 180-4 padding.
  """
  if previous.shape != (8,) or sketch.shape != (8,) or tag_digest.shape != (8,): raise ValueError("SHA witness inputs must be uint32[8]")
  previous,sketch,tag_digest=(x.cast(dtypes.uint32).flatten().contiguous().realize() for x in (previous,sketch,tag_digest))
  output=Tensor.empty(8,dtype=dtypes.uint32,device=previous.device)
  if str(previous.device).split(":",1)[0] == "CUDA":
    output.realize()
    device_name=str(previous.device)
    _cuda_sha256_runtime(device_name)(*(x.uop.buffer.get_buf(device_name) for x in (output,previous,sketch,tag_digest)),wait=False)
    return output
  return output.custom_kernel(previous,sketch,tag_digest,fxn=_sha256_one_thread_kernel)[0].realize()


def _tag_tensor(tag:bytes, device:str) -> Tensor:
  return digest_tensor(hashlib.sha256(tag).digest(),device)


def _raw_words(tensor:Tensor, *, force_bits:int|None=None) -> tuple[Tensor,int,TensorDType]:
  dtype=tensor_dtype_id(tensor)
  value=tensor.contiguous().realize().flatten()
  if dtype == TensorDType.DF32:
    words,bits=value.bitcast(dtypes.uint64),64
  elif dtype == TensorDType.DF16:
    words,bits=value.bitcast(dtypes.uint32),32
  elif dtype == TensorDType.FLOAT16:
    words,bits=value.bitcast(dtypes.uint16).cast(dtypes.uint32),32
  elif dtype == TensorDType.INT32:
    words,bits=value.bitcast(dtypes.uint32),32
  elif dtype == TensorDType.UINT8:
    words,bits=value.cast(dtypes.uint32),32
  else: raise TypeError(f"fast attestation does not support {dtype}")
  if force_bits == 32 and bits == 64: words,bits=words.bitwise_and(0xffffffff).cast(dtypes.uint32),32
  return words,bits,dtype


def _small_tensor_bytes(tensor:Tensor, bits:int) -> bytes:
  expected_dtype=dtypes.uint32 if bits == 32 else dtypes.uint64
  value=tensor.cast(expected_dtype).contiguous().realize().to("CPU").contiguous().realize()
  raw=value.data().cast("B").tobytes()
  itemsize=bits//8
  if sys.byteorder == "big": raw=b"".join(raw[i:i+itemsize][::-1] for i in range(0,len(raw),itemsize))
  return raw


def sample_tensor_256(tensor:Tensor, previous:bytes) -> tuple[bytes,tuple[int,...]]:
  """Gather eight SHA-selected low-32-bit tensor values and copy only 256 bits."""
  words,_,_= _raw_words(tensor,force_bits=32)
  count=int(words.numel())
  if count <= 0: raise ValueError("cannot sample an empty tensor")
  indices=tuple(selector%count for selector in _digest_words(previous,32))
  index=Tensor(indices,dtype=dtypes.int32,device=words.device)
  sampled=words[index].contiguous().realize()
  return _small_tensor_bytes(sampled,32),indices


def tensor_xor_sketch_256_device(tensor:Tensor, previous:Tensor) -> Tensor:
  """Cover every element with a digest-keyed shuffle/XOR and return uint32[8] on-device."""
  words,bits,_=_raw_words(tensor)
  lanes=256//bits
  count=int(words.numel())
  device=str(previous.device)
  if words.device is None: words=words.to(device)
  if count <= 0: return Tensor.zeros(8,dtype=dtypes.uint32,device=device).realize()
  groups=(count+lanes-1)//lanes
  padded=groups*lanes
  if padded != count: words=words.pad(((0,padded-count),),value=0)
  grouped=words.reshape(groups,lanes)

  if previous.shape != (8,): raise ValueError("previous digest must be uint32[8]")
  if bits == 32: selectors=previous.cast(dtypes.uint32)
  else:
    selectors=Tensor.stack(*(previous[2*i].cast(dtypes.uint64).lshift(32).bitwise_or(previous[2*i+1].cast(dtypes.uint64)) for i in range(4)))
  word_dtype=dtypes.uint32 if bits == 32 else dtypes.uint64
  group=Tensor.arange(groups,dtype=dtypes.int32).to(device)
  group_word=group.cast(word_dtype)
  key=selectors[group%lanes].bitwise_xor(group_word)
  mask=lanes-1
  lane_bits=lanes.bit_length()-1
  offset=key.bitwise_and(mask).cast(dtypes.int32).reshape(groups,1)
  output_lane=Tensor.arange(lanes,dtype=dtypes.int32).to(device).reshape(1,lanes).expand(groups,lanes)
  forward=(output_lane-offset).bitwise_and(mask)
  reverse=(offset-output_lane).bitwise_and(mask)
  source=key.rshift(lane_bits).bitwise_and(1).cast(dtypes.bool).reshape(groups,1).where(reverse,forward)
  current=grouped.gather(1,source).contiguous().realize()
  width=groups
  while width > 1:
    if width & 1:
      current=current.pad(((0,1),(0,0)),value=0)
      width+=1
    paired=current.reshape(width//2,2,lanes)
    current=paired[:,0,:].bitwise_xor(paired[:,1,:]).contiguous().realize()
    width//=2
  sketch=current[0].contiguous().realize()
  if bits == 32: return sketch.cast(dtypes.uint32).contiguous().realize()
  return Tensor.stack(*(part for word in sketch for part in (word.rshift(32).cast(dtypes.uint32),word.cast(dtypes.uint32)))).contiguous().realize()


def tensor_xor_sketch_256(tensor:Tensor, previous:bytes) -> bytes:
  """Compatibility wrapper which copies the finished 256-bit sketch to the host."""
  sketch=tensor_xor_sketch_256_device(tensor,digest_tensor(previous,str(tensor.device)))
  return b"".join(struct.pack(">I",word) for word in sketch.to("CPU").tolist())


def _tensor_metadata(name:str, role:TensorRole, tensor:Tensor, mode:str) -> bytes:
  dtype=tensor_dtype_id(tensor)
  shape=[int(x) for x in tensor.shape]
  elements=1
  for dim in shape: elements*=dim
  return canonical_json_bytes({"name":unicodedata.normalize("NFC",name),"role":int(role),"dtype":int(dtype),
    "shape":shape,"elements":elements,"byte_length":elements*DTYPE_ITEMSIZE[dtype],"mode":mode})


@dataclass(frozen=True)
class FastLayerRecord:
  name: str
  root: Tensor|DeviceDigest


@dataclass(frozen=True)
class DeviceRootReference:
  root: bytes|Tensor|DeviceDigest


def _token_contribution(step:int,input_token:int,selected_token:int) -> bytes:
  return hashlib.sha256(struct.pack("<Qii",step,input_token,selected_token)).digest()


def _token_tag(step:int) -> bytes:
  return canonical_json_bytes({"domain":"DFAT-FAST-TOKEN-V3","step":step})


class FastAttestationRecorder:
  """Serial device chain with eight deterministic low-word taps written by each producer kernel."""
  def __init__(self,step:int,seed:bytes|Tensor|DeviceDigest,weights:dict[str,bytes],tag_cache:dict[tuple[str,bytes],Tensor|DeviceDigest]):
    self.step,self.seed=step,seed
    if isinstance(seed,bytes): require_digest(seed,"step_seed")
    elif isinstance(seed,Tensor) and seed.shape != (8,): raise ValueError("device step seed must be uint32[8]")
    self._chain:Tensor|DeviceDigest|None=seed if isinstance(seed,(Tensor,DeviceDigest)) else None
    self._seed_bytes:bytes|None=seed if isinstance(seed,bytes) else None
    self._tensor_index=0
    self.weights=weights
    self._tag_cache=tag_cache
    self.layers:list[FastLayerRecord]=[]
    self._pending:list[tuple[Tensor|DeviceDigest,bytes]]=[]
    self._pending_cuda:list[tuple[Tensor,bytes,int]]=[]
    self.input_token:int|None=None
    self.selected_token:int|None=None
    self._final_cache:Tensor|DeviceDigest|None=None
    self.graph_snapshot_seconds=0.0
    self.layer_attestation_seconds=0.0
    self._pending_token_trace:tuple[list[str],Tensor]|None=None
    self._layer_only_token_trace=False

  def set_token_io(self, *, input_token:int|None=None, selected_token:int|None=None):
    if input_token is not None: self.input_token=int(input_token)
    if selected_token is not None: self.selected_token=int(selected_token)

  def _ensure_device(self,device:str) -> Tensor|DeviceDigest:
    if self._chain is None:
      assert self._seed_bytes is not None
      self._chain=_device_digest_bytes(self._seed_bytes,device) if device.split(":",1)[0] == "CUDA" else digest_tensor(self._seed_bytes,device)
    if str(self._chain.device) != device: raise ValueError(f"attestation chain device {self._chain.device} differs from tensor device {device}")
    return self._chain

  def _constant(self,digest:bytes,device:str) -> Tensor|DeviceDigest:
    key=(device,digest)
    if key not in self._tag_cache:
      self._tag_cache[key]=_device_digest_bytes(digest,device) if device.split(":",1)[0] == "CUDA" else digest_tensor(digest,device)
    return self._tag_cache[key]

  def _advance(self,contribution:Tensor|DeviceDigest,tag:bytes) -> Tensor|DeviceDigest:
    device=str(contribution.device)
    chain=self._ensure_device(device)
    tag_digest=hashlib.sha256(tag).digest()
    if isinstance(chain,DeviceDigest):
      if not isinstance(contribution,DeviceDigest): raise TypeError("CUDA contribution must be a raw device digest")
      self._chain=cuda_sha_buffer_witness(chain,contribution,tag_digest)
    else:
      tag_tensor=self._constant(tag_digest,device)
      if not isinstance(contribution,Tensor) or not isinstance(tag_tensor,Tensor): raise TypeError("CPU digest type mismatch")
      self._chain=sha256_device_witness(chain,contribution,tag_tensor)
    return self._chain

  def _advance_tensor(self,tensor:Tensor,tag:bytes) -> Tensor|DeviceDigest:
    device=str(tensor.device or Device.DEFAULT)
    chain=self._ensure_device(device)
    tag_digest=hashlib.sha256(tag).digest()
    if isinstance(chain,DeviceDigest): self._chain=cuda_fused_tensor_witness(chain,tensor,tag_digest)
    else:
      tensor=tensor.contiguous().realize()
      tag_tensor=self._constant(tag_digest,str(tensor.device))
      if not isinstance(tag_tensor,Tensor): raise TypeError("CPU tag digest type mismatch")
      self._chain=sha256_device_witness(chain,tensor_xor_sketch_256_device(tensor,chain),tag_tensor)
    return self._chain

  def weight_reference(self,name:str,tensor:Tensor) -> DeviceRootReference:
    if name not in self.weights:
      self.weights[name]=sha256_frame("DFAT-FAST-WEIGHT-REFERENCE-V2",(_tensor_metadata(name,TensorRole.WEIGHT,tensor,"reference"),))
    return DeviceRootReference(self.weights[name])

  def capture_tensor(self,name:str,tensor:Tensor,role:TensorRole=TensorRole.STATE) -> Tensor:
    """Write eight selected tensor words from inside its producer kernel; defer hashing to layer end."""
    index=self._tensor_index
    self._tensor_index+=1
    tensor=tensor.contiguous()
    chain=self._ensure_device(str(tensor.device))
    metadata=_tensor_metadata(name,role,tensor,"producer-kernel-low32x8")+struct.pack("<QI",self.step,index)
    if isinstance(chain,DeviceDigest):
      self._pending_cuda.append((tensor,metadata,index))
      return tensor
    else:
      tensor=tensor.realize()
      words,_,_=_raw_words(tensor,force_bits=32)
      selector=chain.cast(dtypes.uint32)
      indices=((selector+index)%tensor.numel()).cast(dtypes.int32)
      witness=words[indices].cast(dtypes.uint32).contiguous().realize()
    self._pending.append((witness,metadata))
    return tensor

  # Compatibility for callers while migrating to the return-value API.
  def snapshot_tensor(self,name:str,tensor:Tensor,role:TensorRole=TensorRole.STATE):
    return self.capture_tensor(name,tensor,role)

  def materialize_pending(self):
    """Schedule all pending CUDA producers together, retaining only their eight-word device witnesses."""
    if not self._pending_cuda: return
    chain=self._ensure_device(str(self._pending_cuda[0][0].device))
    if not isinstance(chain,DeviceDigest): raise TypeError("pending CUDA tensors require a CUDA digest")
    tensors,metadata,ordinals=zip(*self._pending_cuda)
    witnesses=instrumented_cuda_realize_many(tensors,chain,ordinals)
    self._pending.extend(zip(witnesses,metadata))
    self._pending_cuda.clear()

  def flush_snapshots(self):
    self.materialize_pending()
    for witness,metadata in self._pending: self._advance(witness,metadata)
    self._pending.clear()

  def root_reference(self,root:bytes|Tensor|DeviceDigest) -> DeviceRootReference: return DeviceRootReference(root)

  def combine_roots(self,name:str,roots:Sequence[bytes|Tensor|DeviceDigest]) -> Tensor|DeviceDigest:
    if not roots: raise ValueError("cannot combine an empty root list")
    device=next(str(root.device) for root in roots if isinstance(root,(Tensor,DeviceDigest)))
    state=self._constant(hashlib.sha256((name+".seed").encode()).digest(),device)
    for index,root in enumerate(roots):
      value=root if isinstance(root,(Tensor,DeviceDigest)) else self._constant(root,device)
      tag=hashlib.sha256(f"{name}.{index}".encode()).digest()
      if isinstance(state,DeviceDigest):
        if not isinstance(value,DeviceDigest): raise TypeError("CUDA combined root type mismatch")
        state=cuda_sha_buffer_witness(state,value,tag)
      else:
        tag_tensor=self._constant(tag,device)
        if not isinstance(value,Tensor) or not isinstance(tag_tensor,Tensor): raise TypeError("CPU combined root type mismatch")
        state=sha256_device_witness(state,value,tag_tensor)
    return state

  def commit_tensor(self,name:str,value,role:TensorRole=TensorRole.STATE) -> Tensor|DeviceDigest:
    index=self._tensor_index
    self._tensor_index+=1
    if isinstance(value,(RootReference,DeviceRootReference)):
      root=value.root
      if isinstance(root,(Tensor,DeviceDigest)): contribution=root
      else:
        if self._chain is None: raise ValueError("a byte reference cannot precede the first tensor witness")
        contribution=self._constant(root,str(self._chain.device))
      metadata=canonical_json_bytes({"name":name,"role":int(role),"mode":"reference","step":self.step,"index":index})
    elif isinstance(value,(bytes,bytearray,memoryview)):
      raw=bytes(value)
      if self._chain is None: raise ValueError("a byte witness cannot precede the first tensor witness")
      contribution=self._constant(hashlib.sha256(raw).digest(),str(self._chain.device))
      metadata=canonical_json_bytes({"name":name,"role":int(role),"mode":"bytes-sha256","length":len(raw),"step":self.step,"index":index})
    else:
      if value.device is None:
        if self._chain is None: raise ValueError("a device-less tensor cannot be the first witness")
        value=value+Tensor.zeros(value.shape,dtype=value.dtype,device=self._chain.device)
      value=value.contiguous().realize()
      device=str(self._chain.device) if value.device is None else str(value.device)
      metadata=_tensor_metadata(name,role,value,"full-keyed-shuffle-xor256")+struct.pack("<QI",self.step,index)
      return self._advance_tensor(value,metadata)
    return self._advance(contribution,metadata)

  def record_module(self,plan,values:Mapping[str,object],*,input_names:Sequence[str],output_names:Sequence[str]):
    if tuple(values) != plan.witnesses:
      raise ValueError(f"{plan.name} witness order mismatch: expected {plan.witnesses}, got {tuple(values)}")
    for name,value in values.items(): self.commit_tensor(f"{plan.name}.{name}",value)
    assert self._chain is not None
    return self._advance(self._constant(plan.spec_root,str(self._chain.device)),canonical_json_bytes(
      {"domain":"DFAT-FAST-MODULE-END-V3","name":plan.name,"witnesses":list(plan.witnesses)}))

  def record_layer(self,name:str,tensor:Tensor):
    tensor=self.capture_tensor(name,tensor,TensorRole.OUTPUT)
    self.flush_snapshots()
    assert self._chain is not None
    self.layers.append(FastLayerRecord(unicodedata.normalize("NFC",name),self._chain))
    return self._chain

  def attest_layer_output(self,name:str,tensor:Tensor) -> Tensor:
    """Run the ordinary producer unchanged, then chain one full-output XOR/SHA attestation kernel."""
    index=self._tensor_index
    self._tensor_index+=1
    tensor=tensor.contiguous().realize()
    metadata=_tensor_metadata(name,TensorRole.OUTPUT,tensor,"layer-output-keyed-xor256")+struct.pack("<QI",self.step,index)
    self._advance_tensor(tensor,metadata)
    assert self._chain is not None
    self.layers.append(FastLayerRecord(unicodedata.normalize("NFC",name),self._chain))
    return tensor

  def attest_layer_outputs(self,outputs:Sequence[tuple[str,Tensor]],terminal:Tensor) -> Tensor:
    """Materialize the ordinary graph in one scheduler pass, then enqueue ordered layer-output attestations."""
    retained=[tensor.contiguous() for _,tensor in outputs]
    terminal=terminal.contiguous()
    device=str(terminal.device or Device.DEFAULT)
    graph_started=time.monotonic()
    terminal.realize(*retained)
    Device[device].synchronize()
    self.graph_snapshot_seconds=time.monotonic()-graph_started
    attestation_started=time.monotonic()
    for (name,_),tensor in zip(outputs,retained):
      index=self._tensor_index
      self._tensor_index+=1
      metadata=_tensor_metadata(name,TensorRole.OUTPUT,tensor,"layer-output-keyed-xor256")+struct.pack("<QI",self.step,index)
      self._advance_tensor(tensor,metadata)
      assert self._chain is not None
      self.layers.append(FastLayerRecord(unicodedata.normalize("NFC",name),self._chain))
    Device[device].synchronize()
    self.layer_attestation_seconds=time.monotonic()-attestation_started
    return terminal

  def attest_token_trace(self,outputs:Sequence[tuple[str,Tensor]],terminal:Tensor) -> Tensor:
    """Prepare one immutable token trace with the logits graph; hashing is deferred until token selection."""
    if not outputs: raise ValueError("token trace requires at least one layer output")
    if self._pending_token_trace is not None: raise ValueError("token trace is already pending")
    names=[name for name,_ in outputs]
    trace=Tensor.stack(*(tensor for _,tensor in outputs)).contiguous()
    terminal=terminal.contiguous()
    device=str(terminal.device or Device.DEFAULT)
    graph_started=time.monotonic()
    terminal.realize(trace)
    Device[device].synchronize()
    self.graph_snapshot_seconds=time.monotonic()-graph_started
    self._pending_token_trace=(names,trace)
    self._layer_only_token_trace=True
    return terminal

  def accept_token_trace(self,names:Sequence[str],trace:Tensor,graph_seconds:float):
    """Register the already-realized packed trace returned by the JIT model graph."""
    if self._pending_token_trace is not None: raise ValueError("token trace is already pending")
    if trace.shape[0] != len(names): raise ValueError("trace layer count mismatch")
    if not trace.uop._base_buffer_is_realized(): raise ValueError("JIT token trace must be realized")
    self.graph_snapshot_seconds=float(graph_seconds)
    self._pending_token_trace=([unicodedata.normalize("NFC",name) for name in names],trace)
    self._layer_only_token_trace=True

  def finalize_token_trace(self):
    """After the new token exists, attest zero-copy views of its already-realized immutable trace."""
    if self._pending_token_trace is None: raise ValueError("no token trace is pending")
    names,trace=self._pending_token_trace
    device=str(trace.device or Device.DEFAULT)
    attestation_started=time.monotonic()
    for layer_index,name in enumerate(names):
      tensor=trace[layer_index]
      index=self._tensor_index
      self._tensor_index+=1
      metadata=_tensor_metadata(name,TensorRole.OUTPUT,tensor,"token-trace-layer-xor256")+struct.pack("<QI",self.step,index)
      if layer_index == len(names)-1:
        if self.input_token is None or self.selected_token is None: raise ValueError("token I/O must exist before final layer attestation")
        metadata+=canonical_json_bytes({"input_token":self.input_token,"selected_token":self.selected_token})
      self._advance_tensor(tensor,metadata)
      assert self._chain is not None
      self.layers.append(FastLayerRecord(unicodedata.normalize("NFC",name),self._chain))
    Device[device].synchronize()
    self.layer_attestation_seconds=time.monotonic()-attestation_started
    self._pending_token_trace=None

  def capture_layer(self,name:str,tensor:Tensor) -> Tensor:
    tensor=self.capture_tensor(name,tensor,TensorRole.OUTPUT)
    self.flush_snapshots()
    assert self._chain is not None
    self.layers.append(FastLayerRecord(unicodedata.normalize("NFC",name),self._chain))
    return tensor

  @property
  def internal_root_device(self) -> Tensor|DeviceDigest:
    if self._chain is None: raise ValueError("empty attestation step")
    return self._chain

  @property
  def final_root_device(self) -> Tensor|DeviceDigest:
    if self.input_token is None or self.selected_token is None: raise ValueError(f"token step {self.step} is incomplete")
    if self._pending or self._pending_cuda or self._pending_token_trace is not None:
      raise ValueError(f"token step {self.step} has unflushed layer snapshots")
    if self._final_cache is not None: return self._final_cache
    chain=self.internal_root_device
    if self._layer_only_token_trace:
      self._final_cache=chain
      return self._final_cache
    contribution=self._constant(_token_contribution(self.step,self.input_token,self.selected_token),str(chain.device))
    tag=hashlib.sha256(_token_tag(self.step)).digest()
    if isinstance(chain,DeviceDigest):
      if not isinstance(contribution,DeviceDigest): raise TypeError("CUDA token contribution type mismatch")
      self._final_cache=cuda_sha_buffer_witness(chain,contribution,tag)
    else:
      tag_tensor=self._constant(tag,str(chain.device))
      if not isinstance(contribution,Tensor) or not isinstance(tag_tensor,Tensor): raise TypeError("CPU token digest type mismatch")
      self._final_cache=sha256_device_witness(chain,contribution,tag_tensor)
    return self._final_cache


class FastAttestationSession:
  def __init__(self):
    self.weights:dict[str,bytes]={}
    self.kv_roots:dict[tuple[int,str],bytes|Tensor|DeviceDigest]={}
    self.steps:list[FastAttestationRecorder]=[]
    self._tag_cache:dict[tuple[str,bytes],Tensor|DeviceDigest]={}

  def recorder(self,step:int) -> FastAttestationRecorder:
    if self.steps and step <= self.steps[-1].step: raise ValueError("attestation steps must be strictly increasing")
    seed=self.steps[-1].final_root_device if self.steps else ZERO_SHA256
    recorder=FastAttestationRecorder(step,seed,self.weights,self._tag_cache)
    self.steps.append(recorder)
    return recorder

  def artifact(self,metadata:Mapping[str,object],generated_text:str) -> dict[str,object]:
    for device in {root.device for step in self.steps for root in [step.internal_root_device,step.final_root_device] if isinstance(root,DeviceDigest)}:
      Device[device].synchronize()
    def root_bytes(root:bytes|Tensor|DeviceDigest) -> bytes:
      if isinstance(root,bytes): return require_digest(root,"root")
      return _device_digest_to_bytes(root) if isinstance(root,DeviceDigest) else digest_tensor_bytes(root)
    step_json=[]
    previous=ZERO_SHA256
    step_roots=[]
    for step in self.steps:
      seed=root_bytes(step.seed)
      internal=root_bytes(step.internal_root_device)
      layers=[{"name":layer.name,"root":root_bytes(layer.root).hex()} for layer in step.layers]
      final=root_bytes(step.final_root_device)
      if seed != previous: raise ValueError(f"step {step.step} device seed mismatch")
      step_json.append({"step":step.step,"seed_root":seed.hex(),"input_token":step.input_token,"selected_token":step.selected_token,
        "internal_chain_root":internal.hex(),"layers":layers,"token_combined_root":final.hex()})
      previous=final
      step_roots.append(final)
    ordered_steps=ordered_root("DFAT-FAST-ORDERED-STEPS-V2",step_roots)
    names=sorted(self.weights)
    weight_manifest=sha256_frame("DFAT-FAST-WEIGHT-MANIFEST-V2",
      (b"".join(sha256_frame("DFAT-FAST-NAMED-WEIGHT-V2",(name.encode("utf-8"),self.weights[name])) for name in names),))
    run_root=sha256_frame("DFAT-FAST-RUN-V2",(canonical_json_bytes(metadata),weight_manifest,ordered_steps,generated_text.encode("utf-8")))
    document:dict[str,object]={"version":3,"mode":"gpu-full-xor-sha256-v3","metadata":dict(metadata),
      "weights":{name:self.weights[name].hex() for name in names},"weight_manifest_root":weight_manifest.hex(),
      "ordered_steps_root":ordered_steps.hex(),"steps":step_json,
      "generated_text":generated_text,"run_root":run_root.hex()}
    document["document_sha256"]=sha256_frame("DFAT-FAST-DOCUMENT-V2",(canonical_json_bytes(document),)).hex()
    return document

  @staticmethod
  def text_artifact(document:Mapping[str,object]) -> str:
    lines=[str(document["generated_text"]),"",f"run_sha256 {document['run_root']}",f"document_sha256 {document['document_sha256']}"]
    for step in document["steps"]:
      lines.append(f"token_step {step['step']} combined {step['token_combined_root']} internal {step['internal_chain_root']}")
      for layer in step["layers"]: lines.append(f"  layer {layer['name']} {layer['root']}")
    return "\n".join(lines)+"\n"


def _json_digest(value:object,name:str) -> bytes:
  if not isinstance(value,str): raise ValueError(f"{name} must be hexadecimal")
  try: return require_digest(bytes.fromhex(value),name)
  except ValueError as error: raise ValueError(f"invalid {name}") from error


def verify_fast_artifact(document:Mapping[str,object]) -> bool:
  if document.get("version") != 3 or document.get("mode") != "gpu-full-xor-sha256-v3": raise ValueError("not a fast V3 artifact")
  unsigned=dict(document)
  claimed=_json_digest(unsigned.pop("document_sha256",None),"document_sha256")
  if sha256_frame("DFAT-FAST-DOCUMENT-V2",(canonical_json_bytes(unsigned),)) != claimed: raise ValueError("document hash mismatch")
  steps=document.get("steps")
  if not isinstance(steps,list): raise ValueError("steps must be a list")
  metadata=document.get("metadata")
  if not isinstance(metadata,dict): raise ValueError("metadata must be a map")
  layer_only=metadata.get("attestation") == "transformer-layer-output-xor-sha256-v3"
  seed=ZERO_SHA256
  roots=[]
  for ordinal,step in enumerate(steps):
    if not isinstance(step,dict) or int(step.get("step",-1)) != ordinal: raise ValueError("step order mismatch")
    if _json_digest(step.get("seed_root"),"seed_root") != seed: raise ValueError(f"step {ordinal} seed mismatch")
    internal=_json_digest(step.get("internal_chain_root"),"internal_chain_root")
    layers=step.get("layers")
    if not isinstance(layers,list): raise ValueError("layers must be a list")
    for layer in layers:
      if not isinstance(layer,dict) or not isinstance(layer.get("name"),str): raise ValueError("invalid layer record")
      _json_digest(layer.get("root"),"layer_root")
    if layer_only: combined=internal
    else:
      contribution=_token_contribution(ordinal,int(step["input_token"]),int(step["selected_token"]))
      combined=hashlib.sha256(internal+contribution+hashlib.sha256(_token_tag(ordinal)).digest()).digest()
    if combined != _json_digest(step.get("token_combined_root"),"token_combined_root"): raise ValueError("token root mismatch")
    roots.append(combined)
    seed=combined
  weights=document.get("weights")
  if not isinstance(weights,dict): raise ValueError("weights must be a map")
  names=sorted(weights)
  manifest=sha256_frame("DFAT-FAST-WEIGHT-MANIFEST-V2",(b"".join(
    sha256_frame("DFAT-FAST-NAMED-WEIGHT-V2",(name.encode("utf-8"),_json_digest(weights[name],f"weight[{name}]"))) for name in names),))
  if manifest != _json_digest(document.get("weight_manifest_root"),"weight_manifest_root"): raise ValueError("weight manifest mismatch")
  ordered=ordered_root("DFAT-FAST-ORDERED-STEPS-V2",roots)
  if ordered != _json_digest(document.get("ordered_steps_root"),"ordered_steps_root"): raise ValueError("ordered steps mismatch")
  text=document.get("generated_text")
  if not isinstance(metadata,dict) or not isinstance(text,str): raise ValueError("invalid metadata or text")
  run=sha256_frame("DFAT-FAST-RUN-V2",(canonical_json_bytes(metadata),manifest,ordered,text.encode("utf-8")))
  if run != _json_digest(document.get("run_root"),"run_root"): raise ValueError("run root mismatch")
  return True
