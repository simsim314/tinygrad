"""Witness-producing DF operations which preserve the specified arithmetic DAG."""
from __future__ import annotations

from collections import OrderedDict
import math
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


def attested_rope(q:Tensor, k:Tensor, frequencies:Tensor, plan:ModulePlan) -> tuple[tuple[Tensor,Tensor], OrderedDict[str,Tensor]]:
  if plan.kind != "rope": raise ValueError(f"expected RoPE plan, got {plan.kind}")
  qr,kr=q.reshape(*q.shape[:-1],-1,2),k.reshape(*k.shape[:-1],-1,2)
  c,s=frequencies[...,0:1],frequencies[...,1:2]
  qa,qb=qr[...,0:1],qr[...,1:2]
  ka,kb=kr[...,0:1],kr[...,1:2]
  qac,qbs,qas,qbc=qa*c,qb*s,qa*s,qb*c
  kac,kbs,kas,kbc=ka*c,kb*s,ka*s,kb*c
  qo=Tensor.cat(qac-qbs,qas+qbc,dim=-1).flatten(3).contiguous().realize()
  ko=Tensor.cat(kac-kbs,kas+kbc,dim=-1).flatten(3).contiguous().realize()
  values=OrderedDict((
    ("q_input",q),("k_input",k),("frequency_slice",frequencies),
    ("q_ac_bd",Tensor.stack(qac,qbs,dim=-1).contiguous().realize()),
    ("q_ad_bc",Tensor.stack(qas,qbc,dim=-1).contiguous().realize()),("q_rotated",qo),
    ("k_ac_bd",Tensor.stack(kac,kbs,dim=-1).contiguous().realize()),
    ("k_ad_bc",Tensor.stack(kas,kbc,dim=-1).contiguous().realize()),("k_rotated",ko),
    ("rope_output",qo.flatten().cat(ko.flatten()).contiguous().realize())))
  if tuple(values) != plan.witnesses: raise AssertionError(f"RoPE witness implementation differs from schema: {tuple(values)} != {plan.witnesses}")
  return (qo,ko),values


def attested_attention_scores(q:Tensor, k:Tensor, mask:Tensor|None, plan:ModulePlan) -> tuple[Tensor, OrderedDict[str,Tensor]]:
  if plan.kind != "attention_scores": raise ValueError(f"expected attention score plan, got {plan.kind}")
  kt=k.transpose(-2,-1)
  dx,dw=q.ndim,kt.ndim
  xr=q.reshape(*q.shape[:-1],*[1]*min(dx-1,dw-1,1),q.shape[-1])
  wr=kt.reshape(*kt.shape[:-2],*[1]*min(dx-1,dw-1,1),*kt.shape[-2:]).transpose(-1,-2)
  products=xr.cast(dtypes.df32)*wr.cast(dtypes.df32)
  qk,frontiers=reduction_frontiers(products,Ops.ADD,-1)
  head_root=qk.const_like(q.shape[-1]).sqrt().contiguous().realize()
  scaled=(qk/head_root).contiguous().realize()
  if mask is None: mask=scaled.const_like(1).cast(dtypes.bool).tril()
  additive=mask.where(scaled.const_like(0),scaled.const_like(-float("inf"))) if mask.dtype == dtypes.bool else mask
  masked=(scaled+additive).contiguous().realize()
  output=masked.cast(dtypes.df16).contiguous().realize()
  values=OrderedDict((('q_input',q),('repeated_k_input',k)))
  red=reduction_plan(int(q.shape[-1]))
  if red.chunk_count == 1: values['qk_products_df32']=products.contiguous().realize()
  for name,value in zip(red.frontiers,frontiers): values[f'qk_{name}_df32']=value
  values['qk_df32']=qk.contiguous().realize()
  values['head_dim_root_df32']=head_root
  values['scaled_qk_df32']=scaled
  values['mask_and_masked_qk_df32']=masked
  values['softmax_input_df16']=output
  if tuple(values) != plan.witnesses:
    raise AssertionError(f"attention witness implementation differs from schema: {tuple(values)} != {plan.witnesses}")
  return output,values


def attested_residual(left:Tensor, right:Tensor, plan:ModulePlan) -> tuple[Tensor, OrderedDict[str,Tensor]]:
  if plan.kind != "residual_add": raise ValueError(f"expected residual plan, got {plan.kind}")
  wide=(left.cast(dtypes.df32)+right.cast(dtypes.df32)).contiguous().realize()
  upper=wide.const_like(2147483647<<16)
  lower=wide.const_like(-2147483648<<16)
  saturation=((wide > upper) | (wide < lower)).cast(dtypes.uint8).contiguous().realize()
  output=(left+right).contiguous().realize()
  values=OrderedDict((("left_input",left),("right_input",right),("wide_sum",wide),
                      ("saturation_mask",saturation),("df16_output",output)))
  if tuple(values) != plan.witnesses:
    raise AssertionError(f"residual witness implementation differs from schema: {tuple(values)} != {plan.witnesses}")
  return output,values


def attested_silu(x:Tensor, plan:ModulePlan) -> tuple[Tensor, OrderedDict[str,Tensor]]:
  if plan.kind != "silu": raise ValueError(f"expected SiLU plan, got {plan.kind}")
  argument=(x*(-1/math.log(2))).contiguous().realize()
  exponent=argument.exp2().contiguous().realize()
  denominator=(1+exponent).contiguous().realize()
  sigmoid=denominator.reciprocal().contiguous().realize()
  wide=(x.cast(dtypes.df32)*sigmoid.cast(dtypes.df32)).contiguous().realize()
  output=(x*sigmoid).contiguous().realize()
  values=OrderedDict((("input_df16",x),("exp_argument_df16",argument),("exp2_df16",exponent),
                      ("denominator_df16",denominator),("sigmoid_df16",sigmoid),
                      ("wide_silu_product",wide),("silu_output_df16",output)))
  if tuple(values) != plan.witnesses: raise AssertionError(f"SiLU witness implementation differs from schema: {tuple(values)} != {plan.witnesses}")
  return output,values


def attested_gate_product(silu:Tensor, w3:Tensor, plan:ModulePlan) -> tuple[Tensor, OrderedDict[str,Tensor]]:
  if plan.kind != "gate_product": raise ValueError(f"expected gate plan, got {plan.kind}")
  wide=(silu.cast(dtypes.df32)*w3.cast(dtypes.df32)).contiguous().realize()
  narrowed=wide.cast(dtypes.df16).contiguous().realize()
  output=(silu*w3).contiguous().realize()
  values=OrderedDict((("silu_input",silu),("w3_input",w3),("wide_product",wide),
                      ("round_shift_and_saturation",narrowed),("df16_output",output)))
  if tuple(values) != plan.witnesses: raise AssertionError(f"gate witness implementation differs from schema: {tuple(values)} != {plan.witnesses}")
  return output,values
