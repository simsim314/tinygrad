"""Witness-producing DF operations which preserve the specified arithmetic DAG."""
from __future__ import annotations

from collections import OrderedDict
from tinygrad import Tensor, dtypes
from tinygrad.uop import Ops

from extra.dfloat_attestation_schema import ModulePlan, reduction_plan


def _identity(dtype, op:Ops):
  return {Ops.ADD:0,Ops.MUL:1,Ops.MAX:dtype.min}[op]


def _pad_last(x:Tensor, amount:int, identity) -> Tensor:
  if amount == 0: return x
  pads=((0,0),)*(x.ndim-1)+((0,amount),)
  base=x.pad(pads,value=0)
  if identity == 0: return base
  valid=x.const_like(1).cast(dtypes.bool).pad(pads,value=0)
  return valid.where(base,base.const_like(identity))


def reduction_frontiers(x:Tensor, op:Ops, axis:int=-1) -> tuple[Tensor, tuple[Tensor, ...]]:
  """Execute and expose every v1 coarse frontier of the canonical pair tree."""
  if op not in (Ops.ADD,Ops.MUL,Ops.MAX): raise ValueError(f"unsupported witnessed reduction {op}")
  axis=x._resolve_dim(axis)
  if axis != x.ndim-1: x=x.permute(tuple(i for i in range(x.ndim) if i != axis)+(axis,))
  plan=reduction_plan(int(x.shape[-1]))
  identity=_identity(x.dtype,op)
  padded_length=plan.chunk_width*plan.chunk_count
  x=_pad_last(x,padded_length-x.shape[-1],identity)
  chunks=x.reshape(*x.shape[:-1],plan.chunk_count,plan.chunk_width)
  local={Ops.ADD:chunks.sum,Ops.MUL:chunks.prod,Ops.MAX:chunks.max}[op](-1).contiguous().realize()
  frontiers=[local]
  current=local
  if plan.padded_chunk_count != plan.chunk_count:
    current=_pad_last(current,plan.padded_chunk_count-plan.chunk_count,identity)
  width=plan.padded_chunk_count
  while width > 1:
    pair=current.reshape(*current.shape[:-1],width//2,2)
    left,right=pair[...,0],pair[...,1]
    current={Ops.ADD:left+right,Ops.MUL:left*right,Ops.MAX:left.maximum(right)}[op].contiguous().realize()
    frontiers.append(current)
    width//=2
  return current[...,0] if current.shape[-1] == 1 else current, tuple(frontiers)


def attested_matmul(x:Tensor, weight:Tensor, plan:ModulePlan, bias:Tensor|None=None) -> tuple[Tensor, OrderedDict[str,Tensor]]:
  if plan.kind != "matmul": raise ValueError(f"expected matmul plan, got {plan.kind}")
  dx,dw=x.ndim,weight.ndim
  if not (dx > 0 and dw > 0): raise ValueError("matmul inputs must have rank")
  axis_w=-min(dw,2)
  if x.shape[-1] != weight.shape[axis_w]: raise ValueError(f"matmul K mismatch: {x.shape} and {weight.shape}")
  xr=x.reshape(*x.shape[:-1],*[1]*min(dx-1,dw-1,1),x.shape[-1])
  wr=weight.reshape(*weight.shape[:-2],*[1]*min(dx-1,dw-1,1),*weight.shape[axis_w:]).transpose(-1,axis_w)
  xr32,wr32=xr.cast(dtypes.df32),wr.cast(dtypes.df16).cast(dtypes.df32)
  products=(xr32*wr32)
  accumulator,frontiers=reduction_frontiers(products,Ops.ADD,-1)
  post=accumulator if bias is None else accumulator+bias.cast(dtypes.df32)
  output=post.cast(dtypes.df16).contiguous().realize()
  values=OrderedDict()
  values["activation_input"]=x
  right_role=dict(plan.parameters).get("right_role","weight")
  values[f"{right_role}_df16"]=weight.cast(dtypes.df16)
  red=reduction_plan(int(x.shape[-1]))
  if red.chunk_count == 1: values["products_df32"]=products.contiguous().realize()
  for name,value in zip(red.frontiers,frontiers): values[f"{name}_df32"]=value
  if bias is not None: values["bias_result_df32"]=post.contiguous().realize()
  values["output_df16"]=output
  if tuple(values) != plan.witnesses: raise AssertionError(f"matmul witness implementation differs from schema: {tuple(values)} != {plan.witnesses}")
  return output,values


def attested_rmsnorm(x:Tensor, weight:Tensor, epsilon:float, plan:ModulePlan) -> tuple[Tensor, OrderedDict[str,Tensor]]:
  if plan.kind != "rmsnorm": raise ValueError(f"expected rmsnorm plan, got {plan.kind}")
  work=x.cast(dtypes.df32)
  squares=(work*work).contiguous().realize()
  total,frontiers=reduction_frontiers(squares,Ops.ADD,-1)
  total=total.unsqueeze(-1)
  mean_epsilon=(total/total.const_like(x.shape[-1])+total.const_like(epsilon)).contiguous().realize()
  reciprocal_root=mean_epsilon.rsqrt().contiguous().realize()
  normalized=(work*reciprocal_root).cast(dtypes.df16).contiguous().realize()
  output=(normalized*weight.cast(dtypes.df16)).contiguous().realize()
  values=OrderedDict((("input_df16",x),("squares_df32",squares)))
  for name,value in zip(reduction_plan(int(x.shape[-1])).frontiers,frontiers): values[f"square_{name}_df32"]=value
  values["mean_plus_epsilon_df32"]=mean_epsilon
  values["reciprocal_root_df32"]=reciprocal_root
  values["normalized_pre_weight_df16"]=normalized
  values["norm_output_df16"]=output
  if tuple(values) != plan.witnesses: raise AssertionError(f"RMSNorm witness implementation differs from schema: {tuple(values)} != {plan.witnesses}")
  return output,values


def attested_softmax(scores:Tensor, plan:ModulePlan) -> tuple[Tensor, OrderedDict[str,Tensor]]:
  if plan.kind != "softmax": raise ValueError(f"expected softmax plan, got {plan.kind}")
  maximum,max_frontiers=reduction_frontiers(scores,Ops.MAX,-1)
  shifted=(scores-maximum.unsqueeze(-1)).contiguous().realize()
  shifted32=shifted.cast(dtypes.df32).contiguous().realize()
  exponent=shifted32.exp().contiguous().realize()
  total,sum_frontiers=reduction_frontiers(exponent,Ops.ADD,-1)
  reciprocal=total.reciprocal().contiguous().realize()
  output=(exponent*reciprocal.unsqueeze(-1)).cast(dtypes.df16).contiguous().realize()
  values=OrderedDict((("scores_df16",scores),))
  red=reduction_plan(int(scores.shape[-1]))
  for name,value in zip(red.frontiers,max_frontiers): values[f"row_max_{name}_df16"]=value
  values["shifted_df16"]=shifted
  values["shifted_df32"]=shifted32
  values["exp_df32"]=exponent
  for name,value in zip(red.frontiers,sum_frontiers): values[f"exp_sum_{name}_df32"]=value
  values["reciprocal_sum_df32"]=reciprocal
  values["probabilities_df16"]=output
  if tuple(values) != plan.witnesses: raise AssertionError(f"softmax witness implementation differs from schema: {tuple(values)} != {plan.witnesses}")
  return output,values
