"""Correctness-first dense Llama execution with deterministic DF witnesses."""
from __future__ import annotations

from collections import OrderedDict
from tinygrad import Tensor, dtypes

from extra.dfloat_attestation import AttestationRecorder, AttestationSession, RootReference, ordered_root, sha256_frame, tensor_raw_bytes
from extra.dfloat_attestation_ops import (attested_attention_scores, attested_gate_product, attested_matmul,
  attested_residual, attested_rmsnorm, attested_rope, attested_silu, attested_softmax)
from extra.dfloat_attestation_schema import (attention_scores_plan, embedding_plan, gate_product_plan, matmul_plan,
  residual_plan, rmsnorm_plan, rope_plan, silu_plan, softmax_plan, kv_update_plan)
from extra.dfloat_attestation_schema import token_selection_plan
from tinygrad.uop import Ops
from extra.dfloat_attestation_ops import pack_frontiers, reduction_frontiers


def _record(recorder:AttestationRecorder, plan, values, inputs, outputs):
  return recorder.record_module(plan,values,input_names=inputs,output_names=outputs)


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
      kvplan=kv_update_plan(f"{prefix}.attention.kv_update")
      kvvalues=OrderedDict((
        ("prior_k_state",RootReference(prior_k)),("prior_v_state",RootReference(prior_v)),
        ("k_update_slice",k),("v_update_slice",value),("written_k_range",written_k),("written_v_range",written_v),
        ("new_kv_state",RootReference(ordered_root("DFAT-KV-PAIR-V1",(new_k,new_v))))))
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

  normalized=_rmsnorm(h,model.norm,"norm",recorder)
  logits=_linear(normalized,model.output,"output",recorder)
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
  return token
