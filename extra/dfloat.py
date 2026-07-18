"""Correctness-first host conversion utilities for deterministic tinygrad tensors."""
from __future__ import annotations
import hashlib
from tinygrad import Tensor, dtypes
import math
import re

SUPPORTED_WEIGHT_DTYPES = (dtypes.float16, dtypes.bfloat16, dtypes.float32, dtypes.float64)

def tensor_to_df16(src:Tensor, device:str|None=None) -> Tensor:
  """Convert an IEEE tensor to Q15.16 raw storage on CPU, then copy it to device."""
  if src.dtype == dtypes.df16: return src.to(device) if device is not None else src
  if src.dtype not in SUPPORTED_WEIGHT_DTYPES:
    raise TypeError(f"DF16 weight conversion supports F16/BF16/F32/F64, got {src.dtype}")
  import numpy as np
  values = src.numpy().astype(np.float64, copy=False)
  scaled = values * np.float64(65536.0)
  # Loading policy inherited from the prior implementation: nearest, ties away from zero.
  rounded = np.where(scaled >= 0, np.floor(scaled + 0.5), np.ceil(scaled - 0.5))
  rounded = np.where(np.isnan(scaled), 0, rounded)
  raw = np.clip(rounded, np.float64(-2147483648), np.float64(2147483647)).astype(np.int32)
  raw_tensor = Tensor(raw, dtype=dtypes.int32, device="CPU").reshape(src.shape)
  if device is not None: raw_tensor = raw_tensor.to(device)
  return raw_tensor.bitcast(dtypes.df16)

def df16_raw_bytes(t:Tensor) -> bytes:
  if t.dtype != dtypes.df16: raise TypeError(f"expected DF16 tensor, got {t.dtype}")
  raw = t.bitcast(dtypes.int32).contiguous().realize()
  return raw.to("CPU").contiguous().realize().data().cast("B").tobytes()

def df16_checksum(t:Tensor) -> str: return hashlib.sha256(df16_raw_bytes(t)).hexdigest()

def convert_state_dict_df16(state:dict[str,Tensor], device:str|None=None) -> dict[str,Tensor]:
  ret, converted = {}, {}
  for name,t in state.items():
    key=id(t)
    if key in converted: ret[name]=converted[key]; continue
    value = tensor_to_df16(t, device=device) if t.dtype in SUPPORTED_WEIGHT_DTYPES else t.to(device) if device is not None else t
    if device is not None: value=value.realize()
    ret[name]=converted[key]=value
  return ret

def safe_load_df16(path:str, device:str|None=None) -> dict[str,Tensor]:
  """Load ordinary Safetensors, then explicitly convert supported IEEE weights."""
  from tinygrad.nn.state import safe_load
  return convert_state_dict_df16(safe_load(path), device=device)

def convert_state_dict_storage(state:dict[str,Tensor], dtype=dtypes.float16, device:str|None=None, verbose=False) -> dict[str,Tensor]:
  """Keep model weights compact; conversion to DF16 happens in the consuming CUDA graph.

  When verbose, progress covers the actual cast/copy/realize work rather than the
  later (and nearly free) replacement of already-realized model parameters.
  """
  from tinygrad.helpers import GlobalCounters, tqdm
  ret, converted = {}, {}
  unique_total=len({id(t) for t in state.values()})
  layer_totals:dict[int,int]={}
  for name,t in state.items():
    if (m:=re.match(r"layers\.(\d+)\.",name)) is not None:
      layer=int(m.group(1))
      layer_totals[layer]=layer_totals.get(layer,0)+1
  layer_done={layer:0 for layer in layer_totals}
  weight_progress=tqdm(total=unique_total, desc="realizing compact weights", disable=not verbose)
  layer_progress=tqdm(total=len(layer_totals), desc="realized transformer layers", disable=not verbose)
  for name,t in state.items():
    key=id(t)
    if key not in converted:
      value=t.cast(dtype).contiguous()
      if device is not None: value=value.to(device).realize()
      converted[key]=value
      weight_progress.set_description(f"realizing compact weights ({GlobalCounters.mem_used/1e9:.2f} GB GPU)")
      weight_progress.update(1)
    if (m:=re.match(r"layers\.(\d+)\.",name)) is not None:
      layer=int(m.group(1))
      layer_done[layer]+=1
      if layer_done[layer] == layer_totals[layer]:
        layer_progress.update(1)
    ret[name]=converted[key]
  weight_progress.update(close=True)
  layer_progress.update(close=True)
  return ret

class DF16Linear:
  def __init__(self,in_features:int,out_features:int,bias=True,storage_dtype=dtypes.float16):
    bound=1/math.sqrt(in_features)
    self.weight=Tensor.uniform(out_features,in_features,low=-bound,high=bound).cast(storage_dtype)
    self.bias=Tensor.uniform(out_features,low=-bound,high=bound).cast(storage_dtype) if bias else None
  def __call__(self,x:Tensor)->Tensor:
    bias=None if self.bias is None else self.bias.cast(dtypes.df16)
    return x.linear(self.weight.cast(dtypes.df16).T,bias)

class DF16Embedding:
  def __init__(self,vocab_size:int,embed_size:int,storage_dtype=dtypes.float16):
    self.weight=Tensor.glorot_uniform(vocab_size,embed_size).cast(storage_dtype)
  def __call__(self,idx:Tensor)->Tensor:
    if not dtypes.is_int(idx.dtype): raise TypeError(f"Expected integer dtype for index in embedding, got {idx.dtype}")
    # Inference embedding is a direct deterministic gather; a one-hot reduction
    # over the full vocabulary is unnecessary and obscures reduction semantics.
    return self.weight[idx].cast(dtypes.df16)
