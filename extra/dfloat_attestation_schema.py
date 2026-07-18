"""Deterministic v1 witness planning for dense DF Llama inference."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from extra.dfloat_attestation import canonical_json_bytes, sha256_frame

def next_power_of_two(value:int) -> int:
  if value <= 0: raise ValueError(f"reduction length must be positive, got {value}")
  return 1 << (value-1).bit_length()


@dataclass(frozen=True)
class ReductionPlan:
  length: int
  padded_length: int
  tree_depth: int
  witness_level: int
  frontiers: tuple[str, ...]

  def json(self) -> dict[str, object]:
    return {"length":self.length, "padded_length":self.padded_length, "tree_depth":self.tree_depth,
            "witness_level":self.witness_level, "frontiers":list(self.frontiers)}


def reduction_plan(length:int) -> ReductionPlan:
  if length <= 0: raise ValueError(f"reduction length must be positive, got {length}")
  # Select the lower median level of the complete logical tree.  Both the
  # arithmetic DAG and witness location are functions of input length alone.
  tree_depth=(length-1).bit_length()
  witness_level=tree_depth//2
  return ReductionPlan(length,next_power_of_two(length),tree_depth,witness_level,(f"level_{witness_level}",))


@dataclass(frozen=True)
class ModulePlan:
  name: str
  kind: str
  witnesses: tuple[str, ...]
  parameters: tuple[tuple[str, int|str|bool], ...] = ()

  def __post_init__(self):
    if not 5 <= len(self.witnesses) <= 10:
      raise ValueError(f"{self.name} must have 5-10 witnesses, got {len(self.witnesses)}")
    if len(set(self.witnesses)) != len(self.witnesses): raise ValueError(f"duplicate witnesses in {self.name}")

  def json(self) -> dict[str, object]:
    return {"name":self.name, "kind":self.kind, "witnesses":list(self.witnesses),
            "parameters":dict(self.parameters)}

  @property
  def spec_root(self) -> bytes:
    return sha256_frame("DFAT-OPERATION-SPEC-V1", (canonical_json_bytes(self.json()),))


def embedding_plan(name:str="embedding") -> ModulePlan:
  return ModulePlan(name,"embedding",("token_ids","embedding_weight_storage","selected_weight_df16","gathered_df16","embedding_output"))


def rmsnorm_plan(name:str, width:int) -> ModulePlan:
  red = reduction_plan(width)
  witnesses = ("input_df16","squares_df32","square_reduction_frontiers_df32","square_sum_df32",
               "mean_plus_epsilon_df32","reciprocal_root_df32","normalized_pre_weight_df16","norm_output_df16")
  return ModulePlan(name,"rmsnorm",witnesses,(('width',width),('padded_width',red.padded_length),
    ('tree_depth',red.tree_depth),('witness_level',red.witness_level)))


def matmul_plan(name:str, k:int, *, bias:bool=False, right_role:str="weight") -> ModulePlan:
  red = reduction_plan(k)
  witnesses = ["activation_input",f"{right_role}_df16","reduction_frontiers_df32","accumulator_df32"]
  if bias: witnesses.append("bias_result_df32")
  witnesses.append("output_df16")
  return ModulePlan(name,"matmul",tuple(witnesses),(('k',k),('padded_k',red.padded_length),
    ('tree_depth',red.tree_depth),('witness_level',red.witness_level),('bias',bias),('right_role',right_role)))


def rope_plan(name:str="rope") -> ModulePlan:
  return ModulePlan(name,"rope",("q_input","k_input","frequency_slice","q_ac_bd","q_ad_bc","q_rotated",
                                  "k_ac_bd","k_ad_bc","k_rotated","rope_output"))


def kv_update_plan(name:str="kv_update") -> ModulePlan:
  return ModulePlan(name,"kv_update",("prior_k_state","prior_v_state","k_update_slice","v_update_slice",
                                       "written_k_range","written_v_range","new_kv_state"))


def attention_scores_plan(name:str, head_dim:int) -> ModulePlan:
  red = reduction_plan(head_dim)
  witnesses = ("q_input","repeated_k_input","qk_reduction_frontiers_df32","qk_df32","head_dim_root_df32","scaled_qk_df32",
               "mask_and_masked_qk_df32","softmax_input_df16")
  return ModulePlan(name,"attention_scores",witnesses,(('head_dim',head_dim),('padded_head_dim',red.padded_length),
    ('tree_depth',red.tree_depth),('witness_level',red.witness_level)))


def softmax_plan(name:str, width:int) -> ModulePlan:
  red = reduction_plan(width)
  witnesses = ("scores_df16","row_max_frontiers_df16","row_max_df16","shifted_df16","shifted_df32","exp_df32",
               "exp_sum_frontiers_df32","exp_sum_df32","reciprocal_sum_df32","probabilities_df16")
  return ModulePlan(name,"softmax",witnesses,(('width',width),('padded_width',red.padded_length),
    ('tree_depth',red.tree_depth),('witness_level',red.witness_level)))


def residual_plan(name:str) -> ModulePlan:
  return ModulePlan(name,"residual_add",("left_input","right_input","wide_sum","saturation_mask","df16_output"))


def gate_product_plan(name:str) -> ModulePlan:
  return ModulePlan(name,"gate_product",("silu_input","w3_input","wide_product","round_shift_and_saturation","df16_output"))


def silu_plan(name:str) -> ModulePlan:
  return ModulePlan(name,"silu",("input_df16","exp_argument_df16","exp2_df16","denominator_df16",
                                  "sigmoid_df16","wide_silu_product","silu_output_df16"))


def token_selection_plan(width:int, name:str="token_selection", sampling:bool=False) -> ModulePlan:
  red=reduction_plan(width)
  if sampling:
    witnesses=("logits","temperature_scaled_logits","candidate_reduction","top_p_filter","rng_before",
               "selected_candidate","selected_token","rng_after","emitted_token_bytes","text_stop_state")
  else:
    witnesses=("logits","max_frontiers","maximum_and_tie_state","selected_token","text_stop_state")
  return ModulePlan(name,"token_selection",witnesses,(('sampling',sampling),('width',width),
    ('padded_width',red.padded_length),('tree_depth',red.tree_depth),('witness_level',red.witness_level)))


@dataclass(frozen=True)
class LlamaV1Config:
  dim: int
  hidden_dim: int
  head_dim: int
  context_length: int
  qk_norm: bool = False
  linear_bias: bool = False


def llama_block_plan(layer:int, config:LlamaV1Config) -> tuple[ModulePlan, ...]:
  p=f"layers.{layer}"
  modules=[rmsnorm_plan(f"{p}.attention_norm",config.dim),
           matmul_plan(f"{p}.attention.wq",config.dim,bias=config.linear_bias),
           matmul_plan(f"{p}.attention.wk",config.dim,bias=config.linear_bias),
           matmul_plan(f"{p}.attention.wv",config.dim,bias=config.linear_bias)]
  if config.qk_norm:
    modules.extend((rmsnorm_plan(f"{p}.attention.q_norm",config.dim),rmsnorm_plan(f"{p}.attention.k_norm",config.dim)))
  modules.extend((rope_plan(f"{p}.attention.rope"),kv_update_plan(f"{p}.attention.kv_update"),
                  attention_scores_plan(f"{p}.attention.scores",config.head_dim),
                  softmax_plan(f"{p}.attention.softmax",config.context_length),
                  matmul_plan(f"{p}.attention.value_mix",config.context_length,right_role="value"),
                  matmul_plan(f"{p}.attention.wo",config.dim,bias=config.linear_bias),residual_plan(f"{p}.attention_residual"),
                  rmsnorm_plan(f"{p}.ffn_norm",config.dim),matmul_plan(f"{p}.feed_forward.w1",config.dim,bias=config.linear_bias),
                  silu_plan(f"{p}.feed_forward.silu"),matmul_plan(f"{p}.feed_forward.w3",config.dim,bias=config.linear_bias),
                  gate_product_plan(f"{p}.feed_forward.gate"),
                  matmul_plan(f"{p}.feed_forward.w2",config.hidden_dim,bias=config.linear_bias),residual_plan(f"{p}.output_residual")))
  return tuple(modules)


def schema_root(modules:Sequence[ModulePlan], metadata:Mapping[str, object]) -> bytes:
  document={"version":1,"metadata":dict(metadata),"modules":[x.json() for x in modules]}
  return sha256_frame("DFAT-SCHEMA-V1",(canonical_json_bytes(document),))
