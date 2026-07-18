"""Hardware-independent SHA-256 commitments for deterministic DF execution.

This module deliberately contains no model hooks.  It defines the canonical
wire format and transcript primitives which CPU and accelerator hashers must
match.  See docs/dfloat-attestation-v1.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
import hashlib
import json
import struct
import unicodedata
from typing import Iterable, Mapping, Sequence

SHA256_SIZE, TENSOR_CHUNK_SIZE = 32, 4096
ZERO_SHA256 = bytes(SHA256_SIZE)


class TensorRole(IntEnum):
  INPUT = 0
  OUTPUT = 1
  STATE = 2
  WEIGHT = 3


class TensorDType(IntEnum):
  DF16 = 1       # signed Q15.16, little-endian int32 storage
  DF32 = 2       # signed Q31.32, little-endian int64 storage
  FLOAT16 = 3    # raw IEEE binary16 bits
  INT32 = 4
  UINT8 = 5


DTYPE_ITEMSIZE = {TensorDType.DF16: 4, TensorDType.DF32: 8, TensorDType.FLOAT16: 2,
                  TensorDType.INT32: 4, TensorDType.UINT8: 1}


def _u8(value:int) -> bytes:
  if not 0 <= value <= 0xff: raise ValueError(f"uint8 out of range: {value}")
  return struct.pack("<B", value)


def _u32(value:int) -> bytes:
  if not 0 <= value <= 0xffffffff: raise ValueError(f"uint32 out of range: {value}")
  return struct.pack("<I", value)


def _u64(value:int) -> bytes:
  if not 0 <= value <= 0xffffffffffffffff: raise ValueError(f"uint64 out of range: {value}")
  return struct.pack("<Q", value)


def _text(value:str) -> bytes:
  normalized = unicodedata.normalize("NFC", value)
  encoded = normalized.encode("utf-8", errors="strict")
  if b"\0" in encoded: raise ValueError("attestation text fields may not contain NUL")
  return encoded


def require_digest(value:bytes, name:str="digest") -> bytes:
  value = bytes(value)
  if len(value) != SHA256_SIZE: raise ValueError(f"{name} must contain exactly {SHA256_SIZE} bytes")
  return value


def _validate_tensor_size(dtype:TensorDType, shape:Sequence[int], byte_length:int):
  elements = 1
  for dim in shape: elements *= dim
  expected = elements * DTYPE_ITEMSIZE[dtype]
  if byte_length != expected: raise ValueError(f"{dtype.name} tensor shape {tuple(shape)} requires {expected} bytes, got {byte_length}")


def sha256_frame(domain:str, fields:Iterable[bytes]) -> bytes:
  """Hash an unambiguous domain-separated sequence of byte fields.

  Wire form: ASCII-domain, NUL, uint32 field count, then for each field a
  little-endian uint64 byte length and the bytes.  This prevents variable-field
  concatenation ambiguity while retaining ordinary FIPS 180-4 SHA-256.
  """
  domain_bytes = domain.encode("ascii", errors="strict")
  if not domain_bytes or b"\0" in domain_bytes: raise ValueError("domain must be non-empty ASCII without NUL")
  materialized = tuple(bytes(x) for x in fields)
  h = hashlib.sha256(domain_bytes + b"\0" + _u32(len(materialized)))
  for value in materialized:
    h.update(_u64(len(value)))
    h.update(value)
  return h.digest()


def canonical_tensor_header(*, step:int, boundary:int, role:TensorRole, name:str,
                            dtype:TensorDType, shape:Sequence[int], byte_length:int) -> bytes:
  """Encode a non-weight tensor header without host-dependent data."""
  if role == TensorRole.WEIGHT: raise ValueError("use canonical_weight_header for persistent weights")
  name_bytes = _text(name)
  dims = tuple(int(x) for x in shape)
  if any(x < 0 for x in dims): raise ValueError(f"negative tensor dimension in {dims}")
  _validate_tensor_size(dtype, dims, byte_length)
  return b"".join((b"DFAT-TENSOR-V1\0", _u64(step), _u32(boundary), _u8(int(role)),
                   _u32(len(name_bytes)), name_bytes, _u8(int(dtype)), _u32(len(dims)),
                   *(_u64(x) for x in dims), _u64(byte_length)))


def canonical_weight_header(*, name:str, dtype:TensorDType, shape:Sequence[int], byte_length:int) -> bytes:
  """Encode a persistent weight header suitable for a cacheable static root."""
  name_bytes = _text(name)
  dims = tuple(int(x) for x in shape)
  if any(x < 0 for x in dims): raise ValueError(f"negative tensor dimension in {dims}")
  _validate_tensor_size(dtype, dims, byte_length)
  return b"".join((b"DFAT-WEIGHT-V1\0", _u8(int(TensorRole.WEIGHT)), _u32(len(name_bytes)), name_bytes,
                   _u8(int(dtype)), _u32(len(dims)), *(_u64(x) for x in dims), _u64(byte_length)))


def tensor_merkle_root(header:bytes, data:bytes|bytearray|memoryview, chunk_size:int=TENSOR_CHUNK_SIZE) -> bytes:
  """Commit canonical tensor bytes using the fixed adjacent-pair Merkle tree."""
  if chunk_size != TENSOR_CHUNK_SIZE: raise ValueError(f"v1 chunk size must be {TENSOR_CHUNK_SIZE}")
  raw = memoryview(data).cast("B")
  chunk_count = max(1, (len(raw) + chunk_size - 1) // chunk_size)
  level = [sha256_frame("DFAT-CHUNK-V1", (header, _u64(i), _u32(len(chunk)), chunk))
           for i in range(chunk_count) for chunk in (raw[i*chunk_size:min(len(raw), (i+1)*chunk_size)],)]
  tree_level = 0
  while len(level) > 1:
    level = [sha256_frame("DFAT-MERKLE-V1", (_u32(tree_level), level[i], level[i+1] if i+1 < len(level) else ZERO_SHA256))
             for i in range(0, len(level), 2)]
    tree_level += 1
  return level[0]


def tensor_commitment(*, step:int, boundary:int, role:TensorRole, name:str,
                      dtype:TensorDType, shape:Sequence[int], data:bytes|bytearray|memoryview) -> bytes:
  raw = memoryview(data).cast("B")
  header = canonical_tensor_header(step=step, boundary=boundary, role=role, name=name,
                                   dtype=dtype, shape=shape, byte_length=len(raw))
  return tensor_merkle_root(header, raw)


def weight_commitment(*, name:str, dtype:TensorDType, shape:Sequence[int],
                      data:bytes|bytearray|memoryview) -> bytes:
  raw = memoryview(data).cast("B")
  header = canonical_weight_header(name=name, dtype=dtype, shape=shape, byte_length=len(raw))
  return tensor_merkle_root(header, raw)


def ordered_root(domain:str, roots:Sequence[bytes]) -> bytes:
  """Commit a small fixed-width root list with one framed SHA-256 invocation."""
  packed = _u32(len(roots)) + b"".join(require_digest(x, f"root[{i}]") for i,x in enumerate(roots))
  return sha256_frame(domain, (packed,))


def xor_roots(roots:Sequence[bytes]) -> bytes:
  out = bytearray(SHA256_SIZE)
  for i,root in enumerate(roots):
    for j,value in enumerate(require_digest(root, f"root[{i}]")): out[j] ^= value
  return bytes(out)


@dataclass(frozen=True)
class WitnessRecord:
  index: int
  name: str
  root: bytes
  chain: bytes

  def json(self) -> dict[str, object]:
    return {"index": self.index, "name": self.name, "root": self.root.hex(), "chain": self.chain.hex()}


@dataclass
class ModuleChain:
  step: int
  name: str
  operation_spec_root: bytes
  input_roots: tuple[bytes, ...]
  _chain: bytes = field(init=False, repr=False)
  witnesses: list[WitnessRecord] = field(default_factory=list, init=False)

  def __post_init__(self):
    self.operation_spec_root = require_digest(self.operation_spec_root, "operation_spec_root")
    self.input_roots = tuple(require_digest(x, f"input_root[{i}]") for i,x in enumerate(self.input_roots))
    self._chain = sha256_frame("DFAT-MODULE-START-V1", (_u64(self.step), _text(self.name),
                               self.operation_spec_root, ordered_root("DFAT-INPUT-ROOTS-V1", self.input_roots)))

  def add(self, name:str, root:bytes) -> WitnessRecord:
    index, root = len(self.witnesses), require_digest(root, "witness_root")
    self._chain = sha256_frame("DFAT-MODULE-WITNESS-V1",
                               (_u64(self.step), _text(self.name), _u32(index), _text(name), root, self._chain))
    record = WitnessRecord(index, unicodedata.normalize("NFC", name), root, self._chain)
    self.witnesses.append(record)
    return record

  def finish(self, output_roots:Sequence[bytes]) -> bytes:
    if not self.witnesses: raise ValueError("a module must contain at least one witness")
    outputs = tuple(require_digest(x, f"output_root[{i}]") for i,x in enumerate(output_roots))
    return sha256_frame("DFAT-MODULE-END-V1", (_u64(self.step), _text(self.name), _u32(len(self.witnesses)),
                         ordered_root("DFAT-OUTPUT-ROOTS-V1", outputs), self._chain))


def boundary_root(*, index:int, name:str, input_roots:Sequence[bytes], witness_roots:Sequence[bytes], output_root:bytes) -> bytes:
  return sha256_frame("DFAT-BOUNDARY-V1", (_u32(index), _text(name),
    ordered_root("DFAT-BOUNDARY-INPUTS-V1", input_roots), ordered_root("DFAT-BOUNDARY-WITNESSES-V1", witness_roots),
    require_digest(output_root, "output_root")))


def token_root(*, previous:bytes, position:int, input_token:int, selected_token:int,
               boundary_roots:Sequence[bytes]) -> tuple[bytes, bytes, bytes]:
  previous = require_digest(previous, "previous_token_root")
  ordered = ordered_root("DFAT-ORDERED-BOUNDARIES-V1", boundary_roots)
  diagnostic_xor = xor_roots(boundary_roots)
  combined = sha256_frame("DFAT-TOKEN-V1", (previous, _u64(position), struct.pack("<i", input_token),
                          struct.pack("<i", selected_token), ordered, diagnostic_xor))
  return ordered, diagnostic_xor, combined


def canonical_json_bytes(value:Mapping[str, object]) -> bytes:
  """Encode the deliberately small deterministic JSON subset used by v1."""
  def validate(x:object):
    if x is None or isinstance(x, (str, bool, int)): return
    if isinstance(x, float): raise TypeError("floating JSON protocol fields are forbidden; encode an integer or string")
    if isinstance(x, list):
      for v in x: validate(v)
      return
    if isinstance(x, dict):
      if not all(isinstance(k, str) for k in x): raise TypeError("JSON object keys must be strings")
      for v in x.values(): validate(v)
      return
    raise TypeError(f"unsupported canonical JSON value {type(x).__name__}")
  validate(value)
  return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def document_root(document:Mapping[str, object]) -> bytes:
  if "document_sha256" in document: raise ValueError("document_sha256 must be added only after hashing")
  return sha256_frame("DFAT-DOCUMENT-V1", (canonical_json_bytes(document),))


def tensor_dtype_id(tensor) -> TensorDType:
  """Map a tinygrad tensor dtype without importing tinygrad at module import time."""
  from tinygrad import dtypes
  mapping={dtypes.df16:TensorDType.DF16,dtypes.df32:TensorDType.DF32,dtypes.float16:TensorDType.FLOAT16,
           dtypes.int32:TensorDType.INT32,dtypes.uint8:TensorDType.UINT8}
  if tensor.dtype not in mapping: raise TypeError(f"attestation does not define canonical bytes for {tensor.dtype}")
  return mapping[tensor.dtype]


def tensor_raw_bytes(tensor) -> bytes:
  """Copy a realized tinygrad tensor to canonical little-endian storage bytes."""
  import sys
  from tinygrad import dtypes
  storage={dtypes.df16:dtypes.int32,dtypes.df32:dtypes.int64,dtypes.float16:dtypes.uint16,
           dtypes.int32:dtypes.int32,dtypes.uint8:dtypes.uint8}
  if tensor.dtype not in storage: raise TypeError(f"attestation does not define canonical bytes for {tensor.dtype}")
  raw=tensor.bitcast(storage[tensor.dtype]).contiguous().realize().to("CPU").contiguous().realize().data().cast("B").tobytes()
  itemsize=storage[tensor.dtype].itemsize
  if sys.byteorder == "big" and itemsize > 1:
    raw=b"".join(raw[i:i+itemsize][::-1] for i in range(0,len(raw),itemsize))
  return raw


@dataclass(frozen=True)
class RootReference:
  root: bytes

  def __post_init__(self): require_digest(self.root,"referenced_root")


@dataclass(frozen=True)
class TensorRecord:
  step: int
  boundary: int
  name: str
  role: TensorRole
  dtype: TensorDType|None
  shape: tuple[int, ...]
  root: bytes
  referenced: bool = False

  def json(self) -> dict[str, object]:
    return {"step":self.step,"boundary":self.boundary,"name":self.name,"role":int(self.role),
            "dtype":None if self.dtype is None else int(self.dtype),"shape":list(self.shape),
            "root":self.root.hex(),"referenced":self.referenced}


@dataclass(frozen=True)
class ModuleRecord:
  step: int
  boundary: int
  name: str
  kind: str
  witnesses: tuple[WitnessRecord, ...]
  input_roots: tuple[bytes, ...]
  output_roots: tuple[bytes, ...]
  module_root: bytes
  boundary_root: bytes

  def json(self) -> dict[str, object]:
    return {"step":self.step,"boundary":self.boundary,"name":self.name,"kind":self.kind,
            "input_roots":[x.hex() for x in self.input_roots],"output_roots":[x.hex() for x in self.output_roots],
            "witnesses":[x.json() for x in self.witnesses],"module_root":self.module_root.hex(),
            "boundary_root":self.boundary_root.hex()}


class AttestationRecorder:
  """Correctness-first recorder; accelerator hashers can replace only commit_tensor."""
  def __init__(self, step:int):
    self.step, self._tensor_index = step, 0
    self.tensors:list[TensorRecord]=[]
    self.modules:list[ModuleRecord]=[]
    self.weights:dict[str,bytes]={}

  def weight_reference(self, name:str, tensor) -> RootReference:
    if name not in self.weights:
      dtype,raw=tensor_dtype_id(tensor),tensor_raw_bytes(tensor)
      self.weights[name]=weight_commitment(name=name,dtype=dtype,shape=tuple(int(x) for x in tensor.shape),data=raw)
    return RootReference(self.weights[name])

  def commit_tensor(self, name:str, value, role:TensorRole=TensorRole.STATE) -> bytes:
    index=self._tensor_index
    self._tensor_index += 1
    if isinstance(value,RootReference):
      record=TensorRecord(self.step,index,unicodedata.normalize("NFC",name),role,None,(),value.root,True)
    elif isinstance(value,(bytes,bytearray,memoryview)):
      raw=bytes(value)
      root=tensor_commitment(step=self.step,boundary=index,role=role,name=name,dtype=TensorDType.UINT8,shape=(len(raw),),data=raw)
      record=TensorRecord(self.step,index,unicodedata.normalize("NFC",name),role,TensorDType.UINT8,(len(raw),),root)
    else:
      dtype,raw=tensor_dtype_id(value),tensor_raw_bytes(value)
      shape=tuple(int(x) for x in value.shape)
      root=tensor_commitment(step=self.step,boundary=index,role=role,name=name,dtype=dtype,shape=shape,data=raw)
      record=TensorRecord(self.step,index,unicodedata.normalize("NFC",name),role,dtype,shape,root)
    self.tensors.append(record)
    return record.root

  def record_module(self, plan, values:Mapping[str, object], *, input_names:Sequence[str], output_names:Sequence[str]) -> ModuleRecord:
    expected=plan.witnesses
    if tuple(values.keys()) != expected:
      raise ValueError(f"{plan.name} witness order mismatch: expected {expected}, got {tuple(values.keys())}")
    if any(x not in values for x in (*input_names,*output_names)): raise ValueError("module input/output name is not a witness")
    roots={name:self.commit_tensor(f"{plan.name}.{name}",value) for name,value in values.items()}
    inputs=tuple(roots[x] for x in input_names)
    outputs=tuple(roots[x] for x in output_names)
    chain=ModuleChain(self.step,plan.name,plan.spec_root,inputs)
    for name in expected: chain.add(name,roots[name])
    module_digest=chain.finish(outputs)
    bindex=len(self.modules)
    bdigest=boundary_root(index=bindex,name=plan.name,input_roots=inputs,witness_roots=tuple(roots.values()),output_root=module_digest)
    record=ModuleRecord(self.step,bindex,plan.name,plan.kind,tuple(chain.witnesses),inputs,outputs,module_digest,bdigest)
    self.modules.append(record)
    return record

  def json(self) -> dict[str, object]:
    boundaries=tuple(x.boundary_root for x in self.modules)
    return {"step":self.step,"tensors":[x.json() for x in self.tensors],"modules":[x.json() for x in self.modules],
            "weights":{k:v.hex() for k,v in sorted(self.weights.items())},
            "ordered_boundary_root":ordered_root("DFAT-ORDERED-BOUNDARIES-V1",boundaries).hex(),
            "boundary_xor":xor_roots(boundaries).hex()}
