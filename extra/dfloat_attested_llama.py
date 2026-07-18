"""Correctness-first dense Llama execution with deterministic DF witnesses."""
from __future__ import annotations

from collections import OrderedDict
import time
from tinygrad import Device, Tensor, TinyJit, Variable, dtypes

from extra.dfloat_attestation import AttestationRecorder, AttestationSession, RootReference, ordered_root, sha256_frame, tensor_raw_bytes
from extra.dfloat_attestation_ops import (attested_attention_scores, attested_gate_product, attested_matmul,
  attested_residual, attested_rmsnorm, attested_rope, attested_silu, attested_softmax)
from extra.dfloat_attestation_schema import (attention_scores_plan, embedding_plan, gate_product_plan, matmul_plan,
  residual_plan, rmsnorm_plan, rope_plan, silu_plan, softmax_plan, kv_update_plan)
from extra.dfloat_attestation_schema import token_selection_plan
from tinygrad.uop import Ops
from extra.dfloat_attestation_ops import pack_frontiers, reduction_frontiers
from extra.models.llama import apply_rotary_emb, repeat_kv
from tinygrad.helpers import getenv


def _record(recorder:AttestationRecorder, plan, values, inputs, outputs):
  return recorder.record_module(plan,values,input_names=inputs,output_names=outputs)


def _record_layer(recorder:AttestationRecorder,name:str,tensor:Tensor):
  if (record_layer:=getattr(recorder,"record_layer",None)) is not None: record_layer(name,tensor)


def _linear(x:Tensor, layer, name:str, recorder:AttestationRecorder, *, right_role="weight") -> Tensor:
  plan=matmul_plan(name,int(x.shape[-1]),bias=layer.bias is not None,right_role=right_role)
  output,values=attested_matmul(x,layer.weight.cast(dtypes.df16).T,plan,
                                None if layer.bias is None else layer.bias.cast(dtypes.df16))
  if right_role == "weight": values["weight_df16"]=recorder.weight_reference(f"{name}.weight.df16",layer.weight.cast(dtypes.df16))
  _record(recorder,plan,values,("activation_input",f"{right_role}_df16"),("output_df16",))
  return output


def _rmsnorm(x:Tensor, layer, name:str, recorder:AttestationRecorder) -> Tensor:
  plan=rmsnorm_plan(name,int(x.shape[-1]))
  output,values=attested_rmsnorm(x,layer.weight,layer.eps,plan)
  _record(recorder,plan,values,("input_df16",),("norm_output_df16",))
  return output


def attest_dense_llama_forward(model, tokens:Tensor, *, step:int=0, start_pos:int=0,
                               recorder:AttestationRecorder|None=None, session:AttestationSession|None=None) -> tuple[Tensor,AttestationRecorder]:
  """Run the dense Llama graph and record witnesses, including KV state transitions."""
  if tokens.dtype != dtypes.int32: tokens=tokens.cast(dtypes.int32)
  if recorder is None: recorder=session.recorder(step) if session is not None else AttestationRecorder(step)
  if session is None:
    session=AttestationSession()
    session.weights=recorder.weights
  if tokens.shape[0] != 1: raise ValueError("v1 attestation supports batch size 1")
  recorder.set_token_io(input_token=int(tokens[0,-1].item()))

  eplan=embedding_plan("tok_embeddings")
  selected=model.tok_embeddings.weight.cast(dtypes.df16)[tokens].contiguous().realize()
  embedding=selected.contiguous().realize()
  evalues=OrderedDict((
    ("token_ids",tokens),("embedding_weight_storage",recorder.weight_reference("tok_embeddings.weight",model.tok_embeddings.weight)),
    ("selected_weight_df16",selected),("gathered_df16",selected),("embedding_output",embedding)))
  _record(recorder,eplan,evalues,("token_ids","embedding_weight_storage"),("embedding_output",))
  h=embedding
  _record_layer(recorder,"embedding",h)
  frequencies=model.freqs_cis.cast(h.dtype)[:,start_pos:start_pos+tokens.shape[1],:,:,:]

  for layer_index,layer in enumerate(model.layers):
    prefix=f"layers.{layer_index}"
    normalized=_rmsnorm(h,layer.attention_norm,f"{prefix}.attention_norm",recorder)
    q=_linear(normalized,layer.attention.wq,f"{prefix}.attention.wq",recorder)
    k=_linear(normalized,layer.attention.wk,f"{prefix}.attention.wk",recorder)
    value=_linear(normalized,layer.attention.wv,f"{prefix}.attention.wv",recorder)
    attention=layer.attention
    q=q.reshape(q.shape[0],q.shape[1],attention.n_heads,attention.head_dim)
    k=k.reshape(k.shape[0],k.shape[1],attention.n_kv_heads,attention.head_dim)
    value=value.reshape(value.shape[0],value.shape[1],attention.n_kv_heads,attention.head_dim)
    rplan=rope_plan(f"{prefix}.attention.rope")
    (q,k),rvalues=attested_rope(q,k,frequencies,rplan)
    _record(recorder,rplan,rvalues,("q_input","k_input","frequency_slice"),("rope_output",))
    causal=not bool(attention.max_context)
    attention_mask=None
    if attention.max_context:
      if not hasattr(attention,"cache_kv"):
        attention.cache_kv=Tensor.zeros(2,q.shape[0],attention.max_context,attention.n_kv_heads,attention.head_dim,
                                        dtype=q.dtype,device=q.device).contiguous().realize()
      prior_k=session.kv_roots.get((layer_index,"k"),sha256_frame("DFAT-KV-EMPTY-V1",(layer_index.to_bytes(4,"little"),b"k")))
      prior_v=session.kv_roots.get((layer_index,"v"),sha256_frame("DFAT-KV-EMPTY-V1",(layer_index.to_bytes(4,"little"),b"v")))
      attention.cache_kv[:,:,start_pos:start_pos+tokens.shape[1],:,:].assign(Tensor.stack(k,value)).realize()
      keys=attention.cache_kv[0,:,0:start_pos+tokens.shape[1],:,:]
      values=attention.cache_kv[1,:,0:start_pos+tokens.shape[1],:,:]
      written_k=attention.cache_kv[0,:,start_pos:start_pos+tokens.shape[1],:,:].contiguous().realize()
      written_v=attention.cache_kv[1,:,start_pos:start_pos+tokens.shape[1],:,:].contiguous().realize()
      new_k=recorder.commit_tensor(f"{prefix}.attention.k_state",keys)
      new_v=recorder.commit_tensor(f"{prefix}.attention.v_state",values)
      session.kv_roots[(layer_index,"k")],session.kv_roots[(layer_index,"v")]=new_k,new_v
      root_reference=getattr(recorder,"root_reference",RootReference)
      combine_roots=getattr(recorder,"combine_roots",lambda name,roots: ordered_root("DFAT-KV-PAIR-V1",roots))
      kvplan=kv_update_plan(f"{prefix}.attention.kv_update")
      kvvalues=OrderedDict((
        ("prior_k_state",root_reference(prior_k)),("prior_v_state",root_reference(prior_v)),
        ("k_update_slice",k),("v_update_slice",value),("written_k_range",written_k),("written_v_range",written_v),
        ("new_kv_state",root_reference(combine_roots(f"{prefix}.attention.kv_pair",(new_k,new_v))))))
      _record(recorder,kvplan,kvvalues,("prior_k_state","prior_v_state","k_update_slice","v_update_slice"),("new_kv_state",))
      k,value=keys,values
      if tokens.shape[1] > 1:
        attention_mask=Tensor.full((1,1,tokens.shape[1],start_pos+tokens.shape[1]),float("-inf"),
                                   dtype=h.dtype,device=h.device).triu(start_pos+1)
    if attention.n_rep != 1:
      k=k.repeat((1,1,1,attention.n_rep)).reshape(k.shape[0],k.shape[1],attention.n_kv_heads*attention.n_rep,attention.head_dim)
      value=value.repeat((1,1,1,attention.n_rep)).reshape(value.shape[0],value.shape[1],attention.n_kv_heads*attention.n_rep,attention.head_dim)
    q,k,value=q.transpose(1,2),k.transpose(1,2),value.transpose(1,2)
    score_plan=attention_scores_plan(f"{prefix}.attention.scores",attention.head_dim)
    score,score_values=attested_attention_scores(q,k,attention_mask,score_plan,causal=causal)
    _record(recorder,score_plan,score_values,("q_input","repeated_k_input"),("softmax_input_df16",))
    probability_plan=softmax_plan(f"{prefix}.attention.softmax",int(k.shape[-2]))
    probability,probability_values=attested_softmax(score,probability_plan)
    _record(recorder,probability_plan,probability_values,("scores_df16",),("probabilities_df16",))
    mix_plan=matmul_plan(f"{prefix}.attention.value_mix",int(value.shape[-2]),right_role="value")
    mixed,mix_values=attested_matmul(probability,value,mix_plan)
    _record(recorder,mix_plan,mix_values,("activation_input","value_df16"),("output_df16",))
    mixed=mixed.transpose(1,2).reshape(h.shape[0],h.shape[1],-1)
    projected=_linear(mixed,attention.wo,f"{prefix}.attention.wo",recorder)
    residual_plan_=residual_plan(f"{prefix}.attention_residual")
    h,residual_values=attested_residual(h,projected,residual_plan_)
    _record(recorder,residual_plan_,residual_values,("left_input","right_input"),("df16_output",))

    ffn_input=_rmsnorm(h,layer.ffn_norm,f"{prefix}.ffn_norm",recorder)
    w1=_linear(ffn_input,layer.feed_forward.w1,f"{prefix}.feed_forward.w1",recorder)
    silu_plan_=silu_plan(f"{prefix}.feed_forward.silu")
    activated,silu_values=attested_silu(w1,silu_plan_)
    _record(recorder,silu_plan_,silu_values,("input_df16",),("silu_output_df16",))
    w3=_linear(ffn_input,layer.feed_forward.w3,f"{prefix}.feed_forward.w3",recorder)
    gate_plan=gate_product_plan(f"{prefix}.feed_forward.gate")
    gated,gate_values=attested_gate_product(activated,w3,gate_plan)
    _record(recorder,gate_plan,gate_values,("silu_input","w3_input"),("df16_output",))
    w2=_linear(gated,layer.feed_forward.w2,f"{prefix}.feed_forward.w2",recorder)
    output_plan=residual_plan(f"{prefix}.output_residual")
    h,output_values=attested_residual(h,w2,output_plan)
    _record(recorder,output_plan,output_values,("left_input","right_input"),("df16_output",))
    _record_layer(recorder,prefix,h)

  normalized=_rmsnorm(h,model.norm,"norm",recorder)
  logits=_linear(normalized,model.output,"output",recorder)
  _record_layer(recorder,"logits",logits)
  return logits,recorder


def attest_greedy_selection(logits:Tensor, recorder:AttestationRecorder, *, emitted_token_bytes:bytes=b"", text_stop_state:bytes=b"") -> int:
  values_1d=logits[:,-1,:].flatten().contiguous().realize()
  maximum,frontiers=reduction_frontiers(values_1d,Ops.MAX,-1)
  selected=values_1d.argmax().cast(dtypes.int32).reshape(1).contiguous().realize()
  token=int(selected.item())
  recorder.set_token_io(selected_token=token)
  tie_state=tensor_raw_bytes(maximum)+token.to_bytes(4,"little",signed=True)
  plan=token_selection_plan(int(values_1d.shape[0]),sampling=False)
  values=OrderedDict((
    ("logits",values_1d),("max_frontiers",pack_frontiers(frontiers)),
    ("maximum_and_tie_state",tie_state),("selected_token",selected),
    ("text_stop_state",emitted_token_bytes+b"\0"+text_stop_state)))
  _record(recorder,plan,values,("logits",),("selected_token","text_stop_state"))
  _record_layer(recorder,"token_selection",selected)
  return token


def _token_trace_forward(model,tokens:Tensor,start_pos):
  """Pure normal model graph which additionally returns one packed token-level layer trace."""
  _bsz,seqlen=tokens.shape
  h=model.tok_embeddings(tokens).contiguous()
  frequencies=model.freqs_cis.cast(h.dtype)[:,start_pos:start_pos+seqlen,:,:,:]
  mask=(Tensor.full((1,1,seqlen,start_pos+seqlen),float("-inf"),dtype=h.dtype,device=h.device).triu(start_pos+1)
        if model.max_context != 0 and seqlen > 1 else None)
  layer_outputs=[]
  for layer in model.layers:
    h=layer(h,start_pos,frequencies,mask)
    layer_outputs.append(h)
  logits=model.output(model.norm(h).contiguous().contiguous_backward()).contiguous_backward()
  return logits,Tensor.stack(*layer_outputs).contiguous()


def fast_attest_dense_llama_forward(model,tokens:Tensor,*,step:int=0,start_pos:int=0,recorder=None,session=None,snapshot_copy:bool=True):
  """JIT-replayed normal DF inference with one packed immutable trace output; no per-layer realization."""
  if tokens.dtype != dtypes.int32: tokens=tokens.cast(dtypes.int32)
  if recorder is None: recorder=session.recorder(step)
  recorder.set_token_io(input_token=int(tokens[0,-1].item()))
  started=time.monotonic()
  if tokens.shape[0:2] == (1,1) and start_pos != 0:
    if not hasattr(model,"_dfloat_token_trace_jit"):
      model._dfloat_token_trace_jit=TinyJit(lambda token,position:_token_trace_forward(model,token,position))
    position=Variable("dfloat_attest_start_pos",1,model.max_context-1).bind(start_pos)
    logits,trace=model._dfloat_token_trace_jit(tokens,position)
  else: logits,trace=_token_trace_forward(model,tokens,start_pos)
  logits.realize(trace)
  Device[str(logits.device or Device.DEFAULT)].synchronize()
  recorder.accept_token_trace([f"layers.{index}" for index in range(len(model.layers))],trace,time.monotonic()-started)
  return logits,recorder


def nonjit_token_trace_dense_llama_forward(model,tokens:Tensor,*,step:int=0,start_pos:int=0,recorder=None,session=None,snapshot_copy:bool=True):
  """Ordinary deterministic DF graph plus one separate XOR/SHA attestation node per transformer-layer output."""
  if tokens.dtype != dtypes.int32: tokens=tokens.cast(dtypes.int32)
  if recorder is None: recorder=session.recorder(step)
  recorder.set_token_io(input_token=int(tokens[0,-1].item()))
  h=model.tok_embeddings(tokens)
  frequencies=model.freqs_cis.cast(h.dtype)[:,start_pos:start_pos+tokens.shape[1],:,:,:]
  if model.max_context != 0 and tokens.shape[1] > 1:
    mask=Tensor.full((1,1,tokens.shape[1],start_pos+tokens.shape[1]),float("-inf"),dtype=h.dtype,device=h.device).triu(start_pos+1)
  else: mask=None
  layer_outputs=[]

  for layer_index,layer in enumerate(model.layers):
    normalized=layer.attention_norm(h)
    attention=layer.attention
    if getenv("WQKV"):
      xqkv=attention.wqkv(normalized)
      xqkv=xqkv.reshape(xqkv.shape[0],xqkv.shape[1],attention.n_kv_heads,attention.n_rep+2,attention.head_dim)
      xq=xqkv[:,:,:,:attention.n_rep].reshape(xqkv.shape[0],xqkv.shape[1],-1)
      xk=xqkv[:,:,:,attention.n_rep:attention.n_rep+1].reshape(xqkv.shape[0],xqkv.shape[1],-1)
      xv=xqkv[:,:,:,attention.n_rep+1:attention.n_rep+2].reshape(xqkv.shape[0],xqkv.shape[1],-1)
    else:
      xq,xk,xv=attention.wq(normalized),attention.wk(normalized.contiguous_backward()),attention.wv(normalized)
    if attention.q_norm is not None and attention.k_norm is not None: xq,xk=attention.q_norm(xq),attention.k_norm(xk)
    xq=xq.reshape(xq.shape[0],xq.shape[1],attention.n_heads,attention.head_dim)
    xk=xk.reshape(xk.shape[0],xk.shape[1],attention.n_kv_heads,attention.head_dim)
    xv=xv.reshape(xv.shape[0],xv.shape[1],attention.n_kv_heads,attention.head_dim)
    xq,xk=apply_rotary_emb(xq,xk,frequencies)
    bsz,seqlen=xq.shape[:2]
    if attention.max_context:
      if not hasattr(attention,"cache_kv"):
        attention.cache_kv=Tensor.zeros(2,bsz,attention.max_context,attention.n_kv_heads,attention.head_dim,
                                        dtype=h.dtype,device=h.device).contiguous().realize()
      attention.cache_kv[:,:,start_pos:start_pos+seqlen,:,:].assign(Tensor.stack(xk,xv)).realize()
      keys=attention.cache_kv[0,:,0:start_pos+seqlen,:,:]
      values=attention.cache_kv[1,:,0:start_pos+seqlen,:,:]
    else: keys,values=xk,xv
    if attention.max_context:
      keys,values=repeat_kv(keys,attention.n_rep),repeat_kv(values,attention.n_rep)
      query,keys,values=xq.transpose(1,2),keys.transpose(1,2),values.transpose(1,2)
      mixed=query.scaled_dot_product_attention(keys,values,mask).transpose(1,2)
    else:
      query,keys,values=xq.transpose(1,2),keys.transpose(1,2),values.transpose(1,2)
      mixed=query.scaled_dot_product_attention(keys,values,is_causal=True,enable_gqa=True).transpose(1,2)
    mixed=mixed.reshape(bsz,seqlen,-1)
    h=(h+attention.wo(mixed)).contiguous().contiguous_backward()
    ffn_input=layer.ffn_norm(h)
    activated=layer.feed_forward.w1(ffn_input).silu()
    expanded=layer.feed_forward.w3(ffn_input.contiguous_backward())
    gated=activated*expanded
    h=(h+layer.feed_forward.w2(gated)).contiguous().contiguous_backward()
    # Retain only the logical expression here. In token-trace mode all layer values are copied into one
    # immutable buffer by the final graph realization; no layer is materialized or preserved individually.
    layer_outputs.append((f"layers.{layer_index}",h))

  logits=model.output(model.norm(h).contiguous().contiguous_backward()).contiguous_backward()
  logits=(recorder.attest_token_trace(layer_outputs,logits) if snapshot_copy else
          recorder.attest_layer_outputs(layer_outputs,logits))
  return logits,recorder


def kernel_tap_attest_dense_llama_forward(model,tokens:Tensor,*,step:int=0,start_pos:int=0,recorder=None,session=None):
  """Normal deterministic DF forward with eight producer-kernel taps at meaningful internal states."""
  if tokens.dtype != dtypes.int32: tokens=tokens.cast(dtypes.int32)
  if recorder is None: recorder=session.recorder(step)
  recorder.set_token_io(input_token=int(tokens[0,-1].item()))
  h=recorder.capture_layer("embedding",model.tok_embeddings(tokens).contiguous())
  frequencies=model.freqs_cis.cast(h.dtype)[:,start_pos:start_pos+tokens.shape[1],:,:,:]
  if model.max_context != 0 and tokens.shape[1] > 1:
    mask=Tensor.full((1,1,tokens.shape[1],start_pos+tokens.shape[1]),float("-inf"),dtype=h.dtype,device=h.device).triu(start_pos+1)
  else: mask=None

  for layer_index,layer in enumerate(model.layers):
    prefix=f"layers.{layer_index}"
    normalized=recorder.capture_tensor(f"{prefix}.attention_norm",layer.attention_norm(h).contiguous())
    attention=layer.attention
    if getenv("WQKV"):
      xqkv=attention.wqkv(normalized)
      xqkv=xqkv.reshape(xqkv.shape[0],xqkv.shape[1],attention.n_kv_heads,attention.n_rep+2,attention.head_dim)
      xq=xqkv[:,:,:,:attention.n_rep].reshape(xqkv.shape[0],xqkv.shape[1],-1)
      xk=xqkv[:,:,:,attention.n_rep:attention.n_rep+1].reshape(xqkv.shape[0],xqkv.shape[1],-1)
      xv=xqkv[:,:,:,attention.n_rep+1:attention.n_rep+2].reshape(xqkv.shape[0],xqkv.shape[1],-1)
    else:
      xq,xk,xv=attention.wq(normalized),attention.wk(normalized.contiguous_backward()),attention.wv(normalized)
    if attention.q_norm is not None and attention.k_norm is not None: xq,xk=attention.q_norm(xq),attention.k_norm(xk)
    xq=xq.reshape(xq.shape[0],xq.shape[1],attention.n_heads,attention.head_dim)
    xk=xk.reshape(xk.shape[0],xk.shape[1],attention.n_kv_heads,attention.head_dim)
    xv=xv.reshape(xv.shape[0],xv.shape[1],attention.n_kv_heads,attention.head_dim)
    xq,xk=apply_rotary_emb(xq,xk,frequencies)
    xq=recorder.capture_tensor(f"{prefix}.attention.q_rotated",xq.contiguous())
    xk=recorder.capture_tensor(f"{prefix}.attention.k_rotated",xk.contiguous())
    bsz,seqlen=xq.shape[:2]
    if attention.max_context:
      # The KV assignment is an unavoidable realization boundary. Instrument all producers reached so far
      # in one schedule before that assignment can realize xk through an ordinary, unattested path.
      recorder.materialize_pending()
      if not hasattr(attention,"cache_kv"):
        attention.cache_kv=Tensor.zeros(2,bsz,attention.max_context,attention.n_kv_heads,attention.head_dim,
                                        dtype=h.dtype,device=h.device).contiguous().realize()
      attention.cache_kv[:,:,start_pos:start_pos+seqlen,:,:].assign(Tensor.stack(xk,xv)).realize()
      keys=attention.cache_kv[0,:,0:start_pos+seqlen,:,:]
      values=attention.cache_kv[1,:,0:start_pos+seqlen,:,:]
    else: keys,values=xk,xv
    if attention.max_context:
      keys,values=repeat_kv(keys,attention.n_rep),repeat_kv(values,attention.n_rep)
      query,keys,values=xq.transpose(1,2),keys.transpose(1,2),values.transpose(1,2)
      mixed=query.scaled_dot_product_attention(keys,values,mask).transpose(1,2)
    else:
      query,keys,values=xq.transpose(1,2),keys.transpose(1,2),values.transpose(1,2)
      mixed=query.scaled_dot_product_attention(keys,values,is_causal=True,enable_gqa=True).transpose(1,2)
    mixed=recorder.capture_tensor(f"{prefix}.attention.mixed",mixed.reshape(bsz,seqlen,-1).contiguous())
    h=recorder.capture_tensor(f"{prefix}.attention_residual",(h+attention.wo(mixed)).contiguous().contiguous_backward())
    ffn_input=recorder.capture_tensor(f"{prefix}.ffn_norm",layer.ffn_norm(h).contiguous())
    activated=layer.feed_forward.w1(ffn_input).silu()
    expanded=layer.feed_forward.w3(ffn_input.contiguous_backward())
    gated=recorder.capture_tensor(f"{prefix}.feed_forward.gated",(activated*expanded).contiguous())
    h=recorder.capture_layer(prefix,(h+layer.feed_forward.w2(gated)).contiguous().contiguous_backward())

  normalized=recorder.capture_tensor("norm.output",model.norm(h).contiguous().contiguous_backward())
  logits=recorder.capture_layer("logits",model.output(normalized).contiguous_backward())
  return logits,recorder


def fast_attest_greedy_selection(logits:Tensor,recorder,*,emitted_token_bytes:bytes=b"",text_stop_state:bytes=b"") -> int:
  selected=logits[:,-1,:].argmax().cast(dtypes.int32).reshape(1).contiguous().realize()
  token=int(selected.item())
  recorder.set_token_io(selected_token=token)
  if getattr(recorder,"_pending_token_trace",None) is not None: recorder.finalize_token_trace()
  return token


def kernel_tap_attest_greedy_selection(logits:Tensor,recorder,*,emitted_token_bytes:bytes=b"",text_stop_state:bytes=b"") -> int:
  selected=recorder.capture_layer("token_selection",logits[:,-1,:].argmax().cast(dtypes.int32).reshape(1).contiguous())
  token=int(selected.item())
  recorder.set_token_io(selected_token=token)
  recorder.commit_tensor("token_selection.text_stop_state",emitted_token_bytes+b"\0"+text_stop_state)
  return token
