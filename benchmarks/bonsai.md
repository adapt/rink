# Bonsai 2.5 performance

Measured on November 24, 2026, on an Apple M4 Max with 48 GiB of unified memory.
The checkpoint is PrismML's MLX pack at revision
`aa432fd0e17612f6684e728d4d3c42b6656bcabb`. Its SHA-256 was checked before use.
See the [original baseline](m4-max-bonsai.json) and
[projection optimization measurements](m4-max-bonsai-projections.json).

| Context capacity | Prompt tokens | Before prefill | After prefill | Before decode | After decode | Metal allocation |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4,096 | 32 | 12.1 tok/s | 26.8 tok/s | 11.2 tok/s | 23.5 tok/s | 7.49 GiB |
| 4,096 | 128 | 11.2 tok/s | 24.3 tok/s | 10.2 tok/s | 19.2 tok/s | 7.49 GiB |
| 262,144 | 32 | — | 26.8 tok/s | — | 23.6 tok/s | 23.26 GiB |

All figures are medians of two trials. The before binary is commit `3cda61c`.
Each trial uses 16 fixed decode steps after a short warmup. Kernel profiling is
off. Load and reset time are excluded. The 4K comparison uses the same inputs
and settings; all generated token IDs match between the two binaries.
Metal allocation includes model weights, caches, and workspace. It is not process
RSS or total system memory. The 256K row reserves a large cache but uses a short
prompt. It does **not** measure throughput with 256K tokens of active history.
The allocations are unchanged, with no new device allocations during inference.

The packed projection kernel now loads each 32-bit word and its scale and bias
once for sixteen weights. It loads input vectors in groups of four.
Weights stay packed; no expanded weight buffer is used.
FP16 dequantization and FP32 accumulation are retained, with a different reduction
order in the packed projections.

Prefill still runs one token at a time. Batched projections and attention are
the next major prefill changes; these results do not establish parity with an
optimized MLX or GGUF runtime.

## Correctness

The independent oracle uses the release's `Packed` module, MLX 0.32.0, MLX-LM
0.31.3, and MLX-VLM 0.6.3. It loads the same checkpoint, casts auxiliary weights
to FP16 as the published loader does, and runs one token at a time. Two inputs
cover arithmetic and a conversation with a remembered codeword plus distractors.

The comparison checks 9,436,160 logits at 38 positions across 25- and 474-token
inputs. All 38 greedy predictions match. The maximum relative RMS logit error
is 0.6353%; the maximum absolute error is 0.4140625. The comparison gate is 1%
relative RMS error with exact greedy agreement. Results are not bitwise equal;
the FP16 paths use different reduction orders. These checks do not establish
equivalence for every prompt or for a full 256K conversation.

The initial native CLI checks passed for normal and raw generation, the default `xhigh`
reasoning mode, cached conversation recall, reset, and continuation at a one-token
reply limit. A chat also ran with the default 262,144-token capacity. Unsupported
image, tool, cache-precision, and prefill-batch options return errors. Native
arithmetic generation also passed after the projection change.

The synthetic GPU fixture covers both attention types, recurrent head grouping,
partition boundaries, split prefill, and cache reset. Separate scalar checks cover
forward and inverse Hadamard transforms and packed projections at native widths,
including partial output groups, short widths of 128 and 384 coefficients,
all four codes, independent affine biases, and output buffer guards. The optimized
kernels pass the existing reset and split-prefill checks without changing their
tolerances. macOS workspace tests, GPU tests, formatting, and strict Clippy pass.
The initial shared-code change also passed the existing Inkling GPU suite and
Linux CI checks. This optimization changes only the Bonsai model crate.
