"""Correctness-first host conversion utilities for deterministic tinygrad tensors."""
from __future__ import annotations
import hashlib
from tinygrad import Tensor, dtypes

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
  ret = {}
  for name,t in state.items():
    ret[name] = tensor_to_df16(t, device=device) if t.dtype in SUPPORTED_WEIGHT_DTYPES else t.to(device) if device is not None else t
  return ret

def safe_load_df16(path:str, device:str|None=None) -> dict[str,Tensor]:
  """Load ordinary Safetensors, then explicitly convert supported IEEE weights."""
  from tinygrad.nn.state import safe_load
  return convert_state_dict_df16(safe_load(path), device=device)
