# Deterministic DF16/DF32 inference and attestation

## Main idea

Ordinary floating-point inference can change slightly across GPUs, compilers,
kernel schedules, and reduction orders. This project adds deterministic fixed-
point arithmetic to tinygrad so the same model, prompt, and execution schema
produce exactly the same integer values on every conforming backend.

DF16 stores signed Q15.16 values in 32 bits. DF32 stores signed Q31.32 values in
64 bits. Arithmetic, rounding, saturation, transcendental approximations, and
reduction graphs are explicitly defined. CUDA may execute independent work in
parallel, but operations whose order affects a result use a fixed dependency
graph.

The attestation system commits selected intermediate tensors to SHA-256. Each
layer has 5–10 meaningful witnesses, each token chains the ordered layer roots,
and the final artifact binds the model weights, tokens, generated text, and
complete execution transcript. This is a reproducibility proof, not proof of
work or proof that a particular physical GPU was used.

## What is implemented

- DF16 and DF32 arithmetic on CPU and CUDA, including explicit rounding and
  saturation.
- Fixed reduction DAGs for sums, maxima, matrix products, RMSNorm, attention,
  softmax, residuals, SiLU, and token selection.
- Deterministic integer-generated RoPE sine/cosine tables.
- Dense Llama execution with KV-cache state witnesses.
- Canonical SHA-256 framing, tensor Merkle roots, ordered roots, diagnostic XOR,
  per-module witness chains, per-token chains, and JSON/text run artifacts.
- Integer-only CPU conversion of Q6_K matrices directly to canonical FP16.
- Integer-only CPU conversion of native F32 norm weights to canonical FP16.
- Byte-preserving upload of the resulting pure-FP16 model storage to CUDA;
  weights are converted to DF16 when consumed, while wider intermediates and
  accumulators use DF32.
- CPU/CUDA equality tests for individual operations, a complete small Llama
  block, greedy selection, KV updates, and the resulting attestation artifact.

The initial proof model is Llama-3.2-1B-Instruct Q6_K. Large model files should
be stored under `/mnt/pacer` (or another dedicated model disk), not in this
repository or a home-directory download cache.

## Run the CUDA model

Use Python 3.11 or newer and place `tokenizer.model` beside the GGUF file. Set
`DEV=CUDA`; the old `CUDA=1` selector is not supported by this tinygrad version.

```sh
DEV=CUDA python examples/llama3.py \
  --model /mnt/pacer/ai-models/tinygrad/llama3-1b-instruct/Llama-3.2-1B-Instruct-Q6_K.gguf \
  --size 1B --dfloat --low_memory --no_api --temperature 0 --seed 42
```

The command first decodes Q6_K and F32 source tensors on CPU with integer-only
conversion, uploads 147 canonical FP16 tensors, pre-fills the system prompt,
and opens an interactive `Q:` prompt. For the current smoke test enter:

```text
Tell me a story about a llama
```

The first token is slower because CUDA kernels compile. Later tokens reuse the
compiled programs.

## Run the tests

Focused deterministic conversion and arithmetic tests:

```sh
python -m pytest \
  test/dfloat/test_df_q6_load.py \
  test/dfloat/test_df_convert.py \
  test/dfloat/test_df16_cuda.py -q
```

CPU/CUDA model and attestation equality tests:

```sh
DEV=CUDA python -m pytest test/dfloat/test_df_cpu_cuda.py -q
```

Canonical SHA framing, pinned digests, schema, and artifact tests:

```sh
python -m pytest test/dfloat/test_df_attestation.py -q
```

The most important current checks are:

- `test_one_block_full_attestation`
- `test_one_block_selection_artifact`
- `test_two_token_kv_attestation`
- `test_q6_cpu_decode_then_upload_preserves_canonical_fp16`
- `test_integer_float32_to_fp16_matches_reference`

## Getting an attestation

Attestation is currently exposed as a Python API. Create one
`AttestationSession`, use it for every ordered token step, record greedy token
selection, then serialize the final document:

```python
import json
from extra.dfloat_attestation import AttestationSession
from extra.dfloat_attested_llama import attest_dense_llama_forward, attest_greedy_selection

session = AttestationSession()
logits, recorder = attest_dense_llama_forward(
  model, token_tensor, step=position, start_pos=position, session=session)
selected_token = attest_greedy_selection(logits, recorder)

artifact = session.artifact({
  "model": "Llama-3.2-1B-Instruct-Q6_K",
  "prompt_tokens": prompt_tokens,
  "sampling": "greedy",
}, generated_text)

with open("attestation.json", "w", encoding="utf-8") as f:
  json.dump(artifact, f, sort_keys=True, separators=(",", ":"))

with open("attestation.txt", "w", encoding="utf-8") as f:
  f.write(session.text_artifact(artifact))
```

For multiple generated tokens, repeat the forward and selection calls with
strictly increasing `step`/`position` before calling `session.artifact`.

Useful JSON fields are:

- `run_root`: final run commitment.
- `document_sha256`: commitment to the canonical artifact document.
- `weight_manifest_root`: commitment to persistent weights.
- `steps[].token_combined_root`: ordered per-token commitment.
- `steps[].ordered_boundary_root` and `steps[].boundary_xor`: ordered proof and
  parallel diagnostic aggregate.
- `steps[].modules[].witnesses`: intermediate layer witness names, tensor roots,
  and chained roots.

The full framing and witness specification is in
[`docs/dfloat-attestation-v1.md`](docs/dfloat-attestation-v1.md).

## Remaining work

- Run the complete 1B attested graph on both CPU and CUDA and compare complete
  artifacts. Small-model equality is already validated; a full independent-
  machine proof is not yet complete.
- Add a CLI exporter/verifier for `attestation.json`; the current interface is
  Python-only.
- Move tensor SHA-256/Merkle hashing to an optional accelerator sidecar. The
  current correctness-first recorder copies canonical tensor bytes to CPU and
  uses standard `hashlib` SHA-256.
- Produce and compare a complete persistent-weight manifest on another machine.
- Add integer-only canonical decoders for Q4/Q5/Q8 before claiming the same
  GGUF-level guarantee for models using those formats.
- Continue auditing compiler-defined shifts/conversions and inspect generated
  kernels for every fixed reduction/materialization boundary.
- Optimize witnessed full-model execution; the current implementation favors
  clarity and proof coverage over speed.

Detailed limitations and future work are tracked in
[`docs/dfloat-hard-determinism-todo.md`](docs/dfloat-hard-determinism-todo.md).
