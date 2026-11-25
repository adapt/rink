# Bonsai 2.5 baseline

Measured on November 24, 2026, on an Apple M4 Max with 48 GiB of unified memory.
The checkpoint is PrismML's MLX pack at revision
`aa432fd0e17612f6684e728d4d3c42b6656bcabb`. Its SHA-256 was checked before use.
See [raw measurements and native comparisons](m4-max-bonsai.json).

| Context capacity | Prompt tokens | Prefill tok/s | Decode tok/s | Metal allocation |
| ---: | ---: | ---: | ---: | ---: |
| 4,096 | 32 | 12.4 | 11.4 | 7.49 GiB |
| 4,096 | 128 | 11.6 | 10.4 | 7.49 GiB |
| 262,144 | 32 | 12.3 | 11.3 | 23.26 GiB |

The 4K figures are medians of two trials. The 256K row is one trial. Each trial
uses 16 fixed decode steps after a short warmup. Load and reset time are excluded.
Metal allocation includes model weights, caches, and workspace. It is not process
RSS or total system memory. The 256K row reserves a large cache but uses a short
prompt. It does **not** measure throughput with 256K tokens of active history.

This first implementation uses sequential prefill and basic packed matrix-vector
kernels. It establishes a tested second model path; it does not match the speed
of optimized MLX or GGUF runtimes. Batched prefill and packed projection tuning
are separate work.

## Correctness

The independent oracle uses the release's `Packed` module, MLX 0.32.0, MLX-LM
0.31.3, and MLX-VLM 0.6.3. It loads the same checkpoint, casts auxiliary weights
to FP16 as the published loader does, and runs one token at a time. Two inputs
cover arithmetic and a conversation with a remembered codeword plus distractors.

The comparison checks 9,436,160 logits at 38 positions across 25- and 474-token
inputs. All 38 greedy predictions match. The maximum relative RMS logit error
is 0.7185%; the maximum absolute error is 0.40625. The comparison gate is 1%
relative RMS error with exact greedy agreement. Results are not bitwise equal;
the FP16 paths use different reduction orders. These checks do not establish
equivalence for every prompt or for a full 256K conversation.

Native CLI checks passed for normal and raw generation, the default `xhigh`
reasoning mode, cached conversation recall, reset, and continuation at a one-token
reply limit. A chat also ran with the default 262,144-token capacity. Unsupported
image, tool, cache-precision, and prefill-batch options return errors.

The synthetic GPU fixture covers both attention types, recurrent head grouping,
partition boundaries, split prefill, and cache reset. Separate scalar checks cover
forward and inverse Hadamard transforms and packed projections at native widths,
including partial output groups. The shared-code change also passed the existing
Inkling GPU suite. macOS and Linux workspace tests, formatting, and strict Clippy
checks passed.
