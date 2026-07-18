"""Correctness-first dense Llama execution with deterministic DF witnesses."""
from __future__ import annotations

from collections import OrderedDict
from tinygrad import Tensor, dtypes

from extra.dfloat_attestation import AttestationRecorder
from extra.dfloat_attestation_ops import (attested_attention_scores, attested_gate_product, attested_matmul,
  attested_residual, attested_rmsnorm, attested_rope, attested_silu, attested_softmax)
from extra.dfloat_attestation_schema import (attention_scores_plan, embedding_plan, gate_product_plan, matmul_plan,
  residual_plan, rmsnorm_plan, rope_plan, silu_plan, softmax_plan)


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
                               recorder:AttestationRecorder|None=None) -> tuple[Tensor,AttestationRecorder]:
  """Run the dense Llama graph and record witnesses.  V1 initially supports no KV cache."""
  if tokens.dtype != dtypes.int32: tokens=tokens.cast(dtypes.int32)
  recorder=recorder or AttestationRecorder(step)
  if any(hasattr(layer.attention,"cache_kv") or layer.attention.max_context for layer in model.layers):
    raise NotImplementedError("attested KV-cache execution is implemented in the next checkpoint")

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
    if attention.n_rep != 1:
      k=k.repeat((1,1,1,attention.n_rep)).reshape(k.shape[0],k.shape[1],attention.n_kv_heads*attention.n_rep,attention.head_dim)
      value=value.repeat((1,1,1,attention.n_rep)).reshape(value.shape[0],value.shape[1],attention.n_kv_heads*attention.n_rep,attention.head_dim)
    q,k,value=q.transpose(1,2),k.transpose(1,2),value.transpose(1,2)
    score_plan=attention_scores_plan(f"{prefix}.attention.scores",attention.head_dim)
    score,score_values=attested_attention_scores(q,k,None,score_plan)
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
