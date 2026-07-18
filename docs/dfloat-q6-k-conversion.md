# Deterministic Q6_K to FP16 conversion

The DF Llama load path decodes each 210-byte Q6_K block on CPU and uploads the
resulting FP16 storage bytes unchanged. CUDA does not dequantize Q6_K.

For each of the 256 weights, the decoder reconstructs signed `q` in `[-32,31]`,
reads its signed 8-bit subscale `s`, and reads the block scale `d` as raw IEEE
binary16 bits. It computes the exact value `d * (q*s)` using integer sign,
significand, and exponent arithmetic. It then rounds exactly once to binary16
using round-to-nearest, ties-to-even. No native FP16 or FP32 arithmetic occurs.

The conversion has these canonical edge rules:

- exact zero weights are encoded as positive zero (`0x0000`);
- finite overflow is encoded as the appropriately signed infinity;
- a NaN block scale, or infinity multiplied by zero, is canonical quiet NaN
  (`0x7e00`);
- subnormals are retained and rounded with the same ties-to-even rule;
- output order is the fixed GGML Q6_K block and lane order.

The ordinary GGUF loader retains its existing dequantization behavior. The
integer decoder is selected explicitly by the DF Llama loader, which also
forces GGUF decoding onto CPU before the byte-preserving device upload.

The same DF loader converts native F32 tensors directly from their IEEE-754 raw
bits to canonical FP16 with integer-only round-to-nearest, ties-to-even. Exact
zero is canonicalized to `0x0000`, overflow becomes signed infinity, and NaNs
become `0x7e00`. This covers the model's norm vectors without a native float
cast; all persistent tensors uploaded by this model path therefore use FP16.
