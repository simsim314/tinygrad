# DF16/DF32 hard-determinism boundary and remaining work

## Current proof boundary

The first Llama attestation proof treats the fully realized persistent FP16
weight tensors as canonical inputs.  Their names, shapes, raw little-endian
bytes, and SHA-256 digests must match before execution attestations can be
compared.  All FP16-to-DF conversion and subsequent model arithmetic must use
the deterministic integer implementation.

## Required before GGUF-level cross-machine claims

- Replace native-FP Q4/Q5/Q6/Q8 GGUF dequantization with integer-only decoding
  and explicitly specified rounding to canonical FP16, DF16, or consuming
  kernel operands.
- Pin block traversal, scale application order, subnormal behavior, NaN/Inf
  policy, overflow, and ties-to-even/ties-away rounding for every supported
  GGUF quantization type.
- Test each decoder against raw-bit reference vectors including subnormal
  scales, signed extrema, half-way rounding cases, and saturation.
- Compare the complete persistent-weight manifest on CUDA and at least one
  independent CPU implementation before removing the FP16-manifest precondition.
- Eventually keep Q4/Q5/Q6/Q8 blocks compact and decode them with integer-only
  operations inside the consuming deterministic matmul kernel.

Until these items pass, matching GGUF file hashes alone do not prove matching
deterministic model inputs; matching canonical persistent-weight hashes do.

## Additional backend conformance

- Implement and test the same DF arithmetic semantics on an independent CPU
  backend, not merely repeated CUDA runs.
- Audit compiler-defined signed conversions and shifts; use unsigned arithmetic
  plus explicit bitcasts wherever the language standard leaves behavior open.
- Prohibit graph rewrites and kernel optimizations that change a specified DF
  dependency graph.  Optimization may parallelize independent nodes only.
- Validate fixed reduction DAGs with saturation-sensitive counterexamples and
  inspect generated kernels for required materialization/synchronization edges.
