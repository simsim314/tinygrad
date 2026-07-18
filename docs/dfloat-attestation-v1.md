# DF execution attestation v1 specification (draft)

## 1. Purpose and security claim

An attestation is a reproducible commitment to canonical model inputs, semantic
intermediate tensors, state updates, selected tokens, and emitted text.  Two
conforming implementations that execute the same specified DF computation must
produce the same attestation, even if they split, fuse, schedule, or parallelize
CUDA kernels differently.

This is a deterministic execution transcript and proof of possession of the
committed intermediate values.  Its purpose is to distinguish the specified DF
execution from a native-float execution that merely produces similar text.  A
native-float run will not reproduce the raw per-layer DF tensors and therefore
will not reproduce the transcript.  An implementation that deliberately
reconstructs every required DF intermediate in order to obtain the transcript
has performed the computation this specification intends to attest.

This is not proof-of-work, remote hardware identity, or a zero-knowledge proof.
There is no challenge, verifier nonce, trusted clock, TEE, or TPM requirement.

## 2. Attestation boundary

Version 1 begins from a canonical persistent FP16 weight manifest.  Matching a
GGUF file is not sufficient until integer-only GGUF dequantization is complete.
The manifest commits to every tensor name, shape, dtype, and raw FP16 bytes.
Its roots are static and may be cached after verification.  A run commits to
the static manifest root, so weights do not need to be rehashed for every run.

The execution transcript is defined at semantic tensor boundaries, never at
backend kernel boundaries.  Kernel fusion/splitting and launch geometry are not
part of the digest if the same semantic tensors are produced.

The run also commits an `execution_spec_root`: a canonical hash of the DF
arithmetic version, model architecture/configuration, ordered boundary schema,
tensor dtype rules, tokenizer, and generation algorithm/settings.  It excludes
hardware names, addresses, launch geometry, compiler temporaries, and timing.
This binds the values to the requested computation while remaining hardware
agnostic.

### 2.1 Architecture decision

Attestation is a sidecar over realized semantic tensors:

- do not add SHA state to DF16/DF32 scalar values;
- do not insert SHA updates into `+`, `-`, `*`, `/`, or every CUDA kernel;
- do hash every schema-required layer/state output in canonical raw form;
- report each boundary's input roots and output root;
- reference an already committed input root instead of hashing its bytes again;
- hash the small ordered list of roots to bind graph order and dependencies.

Hashing every scalar operation would cost vastly more than inference and would
make the transcript depend on lowering details.  Hashing only final text would
not distinguish a similar native-float run.  Semantic tensor boundaries give
the intended evidence at a tractable cost: a producer must possess the exact DF
state after every required layer for every token.

Layer boundaries alone are not sufficient for v1.  Numerically significant
operations also expose fixed internal witness points.  These witnesses make an
accidental match from a native-float execution negligible without attaching a
SHA state to every scalar arithmetic operation.

### 2.2 Internal operation witnesses

Witness locations are part of the ordered schema and may not be selected or
autotuned at runtime.  Each witness is encoded and committed like any other
semantic tensor, with its operation name, position, dtype, shape, and dependency
roots.  Version 1 requires:

- matrix multiplication and linear layers: the complete deterministic midpoint
  state selected from the specified adjacent-pair reduction DAG; witnesses must
  follow the actual DAG and are not arbitrary prefix percentages;
- general reductions: the derived midpoint and final level from section 2.3;
- RMSNorm: squared-value tensor, reduction result, deterministic reciprocal-root
  result, scale product, and normalized output;
- softmax: maximum, shifted values, deterministic exponent values, sum, and
  normalized probabilities;
- RoPE: static deterministic sine/cosine table roots, Q/K product terms, and
  rotated Q/K outputs;
- storage/conversion boundaries: source representation and deterministically
  converted DF representation;
- KV state: every K/V update slice and the resulting state root referenced by
  subsequent attention.

Elementwise operations whose complete inputs and outputs are already committed
do not need per-element witnesses unless their implementation contains a
multi-stage approximation.  Such approximations expose their specified stage
outputs.  A witness may be generated inside a computation kernel or by a
semantically equivalent sidecar, but its canonical value and schema position
must not depend on CUDA scheduling, fusion, tiling, or hardware.

For large accumulator states, the implementation hashes fixed output tiles as
soon as each required reduction position is complete.  It need not retain all
witness tensors simultaneously.  All tile leaves are covered by the normative
tensor Merkle root, so deterministic streaming reduces memory but not coverage.

These records demonstrate possession of exact internal DF states.  A
native-float run cannot obtain matching witnesses by numerical luck at a
meaningful probability.  A producer that intentionally recreates every required
DF witness has performed the relevant DF computation for this specification.
Hashes still do not prove which physical instructions or device produced those
values; that stronger claim requires trusted hardware or a verifiable-compute
proof system.

### 2.3 Deterministic witness-selection rule

There is no random, value-dependent, timing-dependent, or hardware-dependent
witness selection.  Given `execution_spec_root`, tensor shapes, and token
position, the complete ordered witness list is known before execution.

For every reduction over a positive logical axis K:

1. enumerate K in increasing logical index order;
2. let `D = ceil(log2(K))` and pad K to `2^D` with the operation identity:
   zero for add, one for multiply, and the dtype minimum for max;
3. at every level, combine adjacent pairs `(0,1), (2,3), ...`; an implementation
   may fuse independent lower subtrees only if it preserves this exact operation
   DAG rather than substituting a grouped reduction;
4. retain the complete level `floor(D/2)` as the internal reduction witness;
5. make the midpoint a full execution barrier, then materialize each dependent
   upper level before starting the next, through level D, whose single value is
   the separately witnessed final accumulator;
6. order non-reduction dimensions lexicographically in contiguous row-major
   logical coordinate order before Merkle chunking.

For matmul reductions, the retained midpoint commits every balanced subtree at
that level across every output coordinate. It is a deterministic intermediate
between individual products and final accumulators and is never selected from a
model dimension, runtime value, hardware schedule, or safety threshold. The
operation specification commits K, the padded length, D, and the witness level.

Thus witness coverage is a property of the mathematical dependency DAG, not
CUDA thread blocks, completion order, a model dimension, or an input-size cap.

For non-reduction modules, the witness list is exactly the numbered semantic
state list in section 5.  Conditional states are controlled only by committed
model configuration and generation settings.  Inactive states are omitted
according to the schema; implementations may not replace them with convenient
internal temporaries.

## 3. Canonical tensor encoding

Each non-weight execution tensor uses this canonical header followed by data:

1. domain string `DFAT-TENSOR-V1` followed by one zero byte;
2. semantic step index as little-endian `uint64`;
3. semantic boundary index as little-endian `uint32`;
4. role byte (`input` = 0, `output` = 1, `state` = 2);
5. UTF-8 name length as little-endian `uint32`, then name bytes;
6. dtype identifier;
7. rank as little-endian `uint32` and each dimension as little-endian `uint64`;
8. byte length as little-endian `uint64`;
9. contiguous row-major raw bytes in little-endian element order.

DF16 is encoded as signed Q15.16 `int32`; DF32 as signed Q31.32 `int64`;
token IDs as signed `int32`.
There are no textual numbers, native-endian values, padding bytes, addresses,
timestamps, device names, or compiler-generated identifiers in tensor hashes.

The static weight-tensor header uses domain `DFAT-WEIGHT-V1`, omits the
step/boundary indices, uses role byte 3, and encodes persistent FP16 as raw
IEEE binary16 `uint16`.  Tensor names are Unicode NFC and valid UTF-8.  Boolean
and enum protocol fields use their assigned integer encodings, not host values.

The normative tensor commitment is the Merkle root defined below.  A flat
`SHA-256(header || data)` may be reported only as a diagnostic and is never
substituted for the normative root.

V1 dtype identifiers are DF16 = 1, DF32 = 2, IEEE binary16 = 3, signed
little-endian int32 = 4, and raw uint8 = 5.  Byte length must equal shape element
count multiplied by the assigned dtype width.

## 4. Fast parallel tensor hashing

Hashing is a sidecar computation and must not modify DF arithmetic kernels.
Scalar arithmetic does not carry a hash field.

Every variable-field SHA input uses the same unambiguous framing function:

`FRAME(domain, fields) = domain_ASCII || NUL || field_count_u32_le ||
                         (field_length_u64_le || field_bytes)*`

`H(domain, fields) = FIPS-180-4-SHA-256(FRAME(domain, fields))`

This is ordinary SHA-256 with domain separation and explicit length prefixes.
It avoids ambiguity such as `(ab,c)` versus `(a,bc)`.  Small ordered root lists
of fixed 32-byte SHA values are encoded as `count_u32_le || root_0 || root_1 ||
...`, passed as one `FRAME` field, and hashed once.  Likewise four unsigned
64-bit values are the 32 bytes `LE64(a) || LE64(b) || LE64(c) || LE64(d)`.
Large tensors use the Merkle construction below so their data can be hashed in
parallel.

1. Split raw tensor data into fixed 4096-byte chunks.  An empty tensor has one
   zero-length chunk.
2. Hash chunks independently as `H("DFAT-CHUNK-V1", [header, chunk_index_u64_le,
   actual_chunk_length_u32_le, chunk_bytes])`.
3. Combine chunk hashes with a fixed adjacent-pair binary Merkle tree using
   `H("DFAT-MERKLE-V1", [level_uint32_le, left, right])`.
4. For an odd node, pair it with the 32-byte all-zero value; never duplicate it.
   A one-chunk tensor's leaf digest is its root and does not gain a parent.
5. SHA-256 is exactly FIPS 180-4 SHA-256 with fixed 32-bit modular operations;
   CUDA and CPU implementations must match the same pinned test vectors.

CUDA work within a level may run concurrently.  Level N+1 may start only after
all required level N digests are complete.  Chunk size and tree topology are
fixed by this specification and may not be autotuned per hardware.

For small activations, one multi-tensor sidecar launch should hash many semantic
boundaries.  Boundary buffers remain alive until hashing completes.  A separate
CUDA stream may overlap hashing with later computation only when producer events
and buffer-lifetime events make the dependency explicit.

## 5. Llama semantic boundaries

Every boundary has a stable index and name.  The 1B dense Llama v1 schema covers:

- input token IDs and embedding output;
- input and output of every RMSNorm and linear projection;
- post-RoPE Q and K;
- K and V cache update slices for every transformer block;
- deterministic attention probabilities and attention output;
- each transformer-block residual output;
- final norm and logits;
- selected token ID and emitted UTF-8 text.

Inputs already committed by a preceding boundary are reported by digest but
referenced rather than rehashed.  Thus the artifact contains the requested
per-layer input and output hashes without duplicate tensor reads.  This avoids
duplicate work and prevents XOR cancellation from being mistaken for coverage.
A schema manifest lists the exact ordered names, roles, expected dtypes, shape
rules, and dependency edges.

### 5.1 Chaining rule inside a computational module

Each substantial computational module has five to ten meaningful chain entries.
An entry commits a full canonical witness tensor root, not a single sampled
scalar.  Pure reshapes, transposes, and views do not receive invented witnesses;
their shape/stride interpretation is committed by the consuming operation.

For module `m` in token step `t`:

`C_0 = SHA-256("DFAT-MODULE-START-V1\0" || t || m || operation_spec || input_roots)`

`C_j = SHA-256("DFAT-MODULE-WITNESS-V1\0" || t || m || j || witness_name || witness_root || C_(j-1))`

`module_root = SHA-256("DFAT-MODULE-END-V1\0" || t || m || witness_count || output_roots || C_last)`

The index, name, dtype, shape, and dependency roots are already present in each
witness header.  They are repeated in the chain domain where necessary to make
parsing unambiguous.  Chaining these small roots is cheap.  Computing a witness
root requires reading or streaming the covered internal values and is the main
attestation cost.

### 5.2 Embedding lookup: 5 entries

1. `token_ids`: canonical input token IDs and positions.
2. `embedding_weight_storage`: cached full persistent storage root; it is
   referenced rather than rehashed on later token steps.
3. `selected_weight_df16`: integer-only FP16-to-DF16 conversion result.
4. `gathered_df16`: gathered rows before output layout materialization.
5. `embedding_output`: contiguous DF16 tensor consumed by block zero.

Repeated token IDs remain repeated in the selected-row tensor.  This commits the
lookup behavior and not merely the set of rows.

### 5.3 RMSNorm: 8 entries

1. `input_df16`: input root referenced from the producer.
2. `squares_df32`: elementwise deterministic widening and squares.
3. `square_reduction_frontiers_df32`: the complete derived square-sum midpoint
   according to section 2.3.
4. `square_sum_df32`: final deterministic sum.
5. `mean_plus_epsilon`: final sum, deterministic division by width, and pinned
   DF epsilon addition.
6. `reciprocal_root_df32`: deterministic integer square-root/reciprocal result.
7. `normalized_pre_weight_df16`: normalization product and narrowing.
8. `norm_output_df16`: deterministic product with the norm weight.

The schema records the reduction length, padded length, tree depth, and derived
witness level. It never substitutes a hardware-selected grouped reduction. The
midpoint witness changes length, but its name and index do not change with
reduction depth.

### 5.4 Linear projection and matmul: 5–10 entries

This schema is used by Q, K, V, O, W1, W2, W3, the output projection, QK score
matmul, and probability-by-V matmul.  Inputs and outputs differ, but the
reduction evidence is shared.

1. `activation_input`: canonical activation root.
2. `weight_df16`: deterministically converted weight root, cached and referenced
   on later tokens; attention-value matmul references the V-state root instead.
3. `reduction_frontiers_df32`: the complete derived midpoint level in canonical
   logical order.
4. `accumulator_df32`: final deterministic accumulator per output.
5. `bias_result`: emitted only when the configured projection has a bias.
6. `output_df16`: deterministic narrowing/saturation and final layout.

The exact count follows only from the semantic presence of bias, giving five or
six entries for every input size.
Every final result is produced by the fully specified pair loop, and the complete
derived midpoint is committed—there is no random or hand-picked sampling.
Implementations may stream midpoint tiles into the hasher instead of storing the
complete frontier.

### 5.5 RoPE for Q and K: 10 entries

1. `q_input` and 2. `k_input`: projection roots after head reshape.
3. `frequency_slice`: static integer-generated sine/cosine slice for positions.
4. `q_ac_bd`: Q real×cos and imaginary×sin product pair.
5. `q_ad_bc`: Q real×sin and imaginary×cos product pair.
6. `q_rotated`: Q `(ac-bd, ad+bc)` result.
7. `k_ac_bd`: K real×cos and imaginary×sin product pair.
8. `k_ad_bc`: K real×sin and imaginary×cos product pair.
9. `k_rotated`: K `(ac-bd, ad+bc)` result.
10. `rope_output`: ordered pair of final contiguous Q and K roots.

The full static frequency-table root is part of `execution_spec_root`; each run
hashes only the position slice it actually consumes.

### 5.6 KV-cache update: 7 entries

1. `prior_k_state` and 2. `prior_v_state`: roots from the preceding token step.
3. `k_update_slice`: post-RoPE K values and destination position range.
4. `v_update_slice`: projected V values and destination position range.
5. `written_k_range` and 6. `written_v_range`: values read from the updated
   cache range, detecting incorrect placement or aliasing.
7. `new_kv_state`: ordered commitment to the new K and V state roots.

The initial all-zero cache root and maximum-context shape are committed by the
execution specification.  A later attention module references `new_kv_state`.

### 5.7 Attention score, scaling, and mask: 8 entries

1. `q_input` and 2. `repeated_k_input`: Q and GQA-expanded K roots.
3. `qk_reduction_frontiers_df32`: the complete derived QK midpoint in
   deterministic logical order.
4. `qk_df32`: final score accumulators.
5. `head_dim_root_df32`: deterministic square root of pinned head dimension.
6. `scaled_qk_df32`: deterministic division result.
7. `mask_and_masked_qk_df32`: exact mask/position interpretation and score after
   deterministic mask addition.
8. `softmax_input_df16`: deterministic narrowing consumed by softmax.

The derived midpoint remains covered for arbitrary K without changing the
witness count.

### 5.8 Softmax: 10 entries

1. `scores_df16`: exact input score root.
2. `row_max_frontiers_df16`: the complete derived max midpoint.
3. `row_max_df16`: final detached row maximum.
4. `shifted_df16`: scores minus detached row maximum.
5. `shifted_df32`: exact widening before exponentiation.
6. `exp_df32`: deterministic integer `exp2(x/log(2))` output.
7. `exp_sum_frontiers_df32`: the complete derived sum midpoint.
8. `exp_sum_df32`: final row sum.
9. `reciprocal_sum_df32`: deterministic reciprocal result.
10. `probabilities_df16`: product, narrowing, and final probabilities.

The packed tensor lengths follow the context shape, while witness names and
indices remain fixed and are determined before execution.

The deterministic exp implementation may additionally expose a versioned
internal polynomial/table witness in a future schema.  V1 commits its complete
input and output and binds its exact arithmetic implementation version through
`execution_spec_root`.

### 5.9 Attention probability-by-V and output projection

Probability-by-V uses the linear/matmul schema in section 5.4, replacing the
weight reference with the committed V-cache state.  The attention output
projection uses it again with the static O-projection weight.  Each therefore
has six to ten entries and its own module root; they are never collapsed into a
single attestation record even if CUDA fuses surrounding work.

### 5.10 Saturating residual or gate multiplication: 5 entries

For residual addition:

1. `left_input`, 2. `right_input`, 3. `wide_sum`, 4. `saturation_mask`, and
5. `df16_output`.

For the gated MLP product:

1. `silu_input`, 2. `w3_input`, 3. `wide_product`, 4. `round_shift_and_saturation`,
and 5. `df16_output`.

The wide intermediate and saturation decision are meaningful DF witnesses: a
native floating operation does not naturally produce their exact integer state.

### 5.11 SiLU: 7 entries

1. `input_df16`.
2. `exp_argument_df16`: direct multiplication by pinned `-1/log(2)`, matching
   the actual tinygrad sigmoid graph without inventing a separate negation.
3. `exp2_df16`: deterministic table/polynomial exponential result.
4. `denominator_df16`: `1 + exp(-x)`.
5. `sigmoid_df16`: deterministic reciprocal.
6. `wide_silu_product`: widened `x * sigmoid(x)` before rounding/saturation.
7. `silu_output_df16`.

### 5.12 One transformer block

The block chain references module roots in this exact graph order:

1. block input;
2. attention RMSNorm;
3. Q, K, and V projections;
4. optional Q/K norms when configured;
5. RoPE;
6. KV-cache update;
7. attention score/scale/mask;
8. softmax;
9. probability-by-V;
10. attention O projection;
11. attention residual addition;
12. FFN RMSNorm;
13. W1 projection and SiLU;
14. W3 projection;
15. gated product;
16. W2 projection;
17. final residual addition and block output.

Each item references its own five-to-ten-entry module root as specified above.
The block chain is allowed to contain more than ten module roots: the five-to-ten
target applies inside each substantial computational module, not to the entire
transformer block, whose complete graph must remain visible.

### 5.13 Final norm, logits, and token selection

Final RMSNorm uses section 5.3 and the vocabulary projection uses section 5.4.
Token selection then records five to ten entries according to sampling mode:

1. final logits root;
2. temperature-scaled logits or a labelled identity for greedy decoding;
3. max/top-k reduction frontier and final candidates;
4. top-p filter state when enabled;
5. canonical RNG state before selection when sampling is enabled;
6. selected candidate index;
7. selected token ID;
8. RNG state after selection when applicable;
9. emitted token bytes;
10. tokenizer-updated text/stop state.

Greedy decoding omits RNG and inactive top-k/top-p entries but retains at least
five meaningful entries: logits, max-reduction frontier, final maximum/tie state,
selected token ID, and emitted token/text state.

## 6. Layer, token, and run commitments

For boundary `i` (all integer encodings are fixed-width little-endian):

`B_i = SHA-256("DFAT-BOUNDARY-V1" || index || name || input roots ||
witness roots || output root)`

The ordered token commitment is a SHA-256 chain over `B_0 ... B_n` in schema
order.  It commits to order regardless of kernel completion order.

An XOR of all `B_i` values is also reported for parallel diagnostics, as
requested.  XOR is never accepted alone: it is commutative, permits duplicate
cancellation, and is not collision resistant as an aggregate.

`token_combined = SHA-256("DFAT-TOKEN-V1" || previous_token_combined ||
position || input_token || selected_token || ordered_boundary_hash || boundary_xor)`

The run commitment chains token commitments in position order and includes the
weight-manifest root, tokenizer hash, prompt token IDs, DF arithmetic/version
identifier, model/schema identifier, sampling parameters, and stop reason.
These fields are also covered by `execution_spec_root`; their explicit presence
keeps the artifact independently inspectable.

## 7. Interpretation and limitations

The model root, tokenizer root, prompt tokens, generation settings, complete
ordered per-token boundary transcript, selected tokens, and output text jointly
identify the computation.  A previously recorded artifact for exactly the same
computation is expected to verify again; replay prevention is not a v1 goal.

The artifact proves that its producer possessed every committed intermediate.
It does not identify the physical GPU or process that produced them, and it does
not prevent a producer from performing the same specified DF computation in a
different conforming implementation.

## 8. Output artifacts

The JSON artifact contains schema/version, canonical input manifest, ordered
boundary records and their internal witness records per token,
XOR/ordered/combined commitments, generated text, and final run commitment.  The
text artifact contains generated text and a compact per-token table of combined
commitments plus the final run commitment.

Canonical JSON hashing uses UTF-8, lexicographically sorted keys, no insignificant
whitespace, and integers only for numeric protocol fields.  The document hash is
computed before adding its own hash field.

## 9. Verification requirements

- CPU reference and CUDA Merkle implementations must match pinned vectors.
- CPU and CUDA schema planners must produce the same complete witness names,
  indices, shapes, and dependency order before executing tensor arithmetic.
- Replanning with different CUDA launch sizes, available concurrency, or device
  identity must produce an identical witness plan.
- Reordering two boundaries must preserve XOR but change ordered and combined
  commitments.
- Changing one tensor or witness bit, name, role, shape, witness position, token
  position, or state update must change the combined run commitment.
- Replacing required DF intermediate states with native-float intermediates must
  fail pinned witness-vector verification, even when the selected token matches.
- Different CUDA launch geometry and safe kernel fusion/splitting must preserve
  commitments.
- Attestation disabled/enabled must produce identical model tensor values and
  selected tokens.
- Performance and memory overhead are reported separately from correctness.
