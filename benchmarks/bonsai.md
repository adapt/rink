# Bonsai 2.5 performance

Measured on November 24, 2026, on an Apple M4 Max with 48 GiB of unified memory.
The checkpoint is PrismML's MLX pack at revision
`aa432fd0e17612f6684e728d4d3c42b6656bcabb`. Its SHA-256 was checked before use.
[Raw measurements and reference checks](m4-max-bonsai-batched.json) accompany this
report. The targets of 1,000 tok/s prefill and 60 tok/s decode are **not met**.

| Context capacity | Prompt tokens | Before prefill | After prefill | Before decode | After decode | Metal allocation after |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4,096 | 128 | 24.6 tok/s | 198.9 tok/s | 19.8 tok/s | 36.8 tok/s | 7.64 GiB |
| 4,096 | 512 | 20.0 tok/s | 222.4 tok/s | 17.3 tok/s | 35.9 tok/s | 7.64 GiB |
| 4,096 | 1,024 | — | 219.0 tok/s | — | 35.5 tok/s | 7.64 GiB |
| 262,144 | 128 | — | 194.8 tok/s | — | 35.4 tok/s | 23.56 GiB |
| 262,144 | 512 | — | 224.5 tok/s | — | 35.8 tok/s | 23.56 GiB |

Figures are medians of two wall-clock trials. The before binary is commit
`a0718d7`. Each trial starts with an empty conversation and uses 32 fixed decode
steps, including steps after EOS. Four warmup steps precede the trials. Kernel
profiling is off. Load and reset time are excluded. All 128 generated token IDs
in the four matched before/after trials are identical.

Context capacity is reserved cache space. The 256K rows use short prompts;
they do **not** measure throughput with 256K tokens of active history.
Metal allocation includes weights, caches, and workspace. It is not process RSS
or total system memory. The 4K allocation increased from 7.49 to 7.64 GiB for the
batched workspace. The 256K cache policy is unchanged. Neither prefill nor decode
makes device allocations after loading. Decode reads back four bytes per step.

## Changes

Prefill now processes up to 512 tokens per batch. Packed matrix tiles are
unpacked into on-chip FP16 storage and multiplied with FP32 SIMD matrix
accumulators. Full attention processes causal query tiles in bounded workspace.
Recurrent attention updates its FP32 state in token order. Only the last prompt
token runs the vocabulary projection.

Decode uses a specialized packed projection when the stored FP16 affine bias is
exactly the negative scale and all four weight values are representable. That
path factors the scale after a partial dot product. Other affine matrices retain
explicit FP16 dequantization. Exact matching rotation signs allow projections to
share a Hadamard transform. Small-head normalization uses SIMD reductions. The
GPU selects the next token with the same tie and non-finite rules as the CPU.

Weights remain packed. The implementation retains FP16 activation boundaries,
FP32 accumulators and recurrent state, and disabled Metal fast math. Different
reduction orders produce small numerical differences from the reference runtime.

The large projections remain the main bottleneck. Lookup-table products,
lossless ternary repacking, larger tiles, alternate threadgroup sizes, and the
tested Metal TensorOps path did not beat the retained kernels. These experiments
are not part of the inference implementation.

## Reproduce

```sh
cargo build --release -p rink-cli
target/release/rink bench --model checkpoints/bonsai-2.5-27b \
  --prompt-tokens 128,512,1024 --decode-tokens 32 --repetitions 2 \
  --context 4096 --prefill-batch 512
target/release/rink bench --model checkpoints/bonsai-2.5-27b \
  --prompt-tokens 128,512 --decode-tokens 32 --repetitions 2 --context 262144
```

For the before result, build `a0718d7` and run the first benchmark with
`--prompt-tokens 128,512` and without `--prefill-batch`.
The [initial implementation](m4-max-bonsai.json) and
[first projection changes](m4-max-bonsai-projections.json) have separate reports.

## Correctness

The independent oracle uses the release's `Packed` module, MLX 0.32.0, MLX-LM
0.31.3, and MLX-VLM 0.6.3. It loads the same checkpoint, casts auxiliary weights
to FP16 as the published loader does, and runs one token at a time. Two inputs
cover arithmetic and a conversation with a remembered codeword plus distractors.

Batched comparisons check 9,932,800 logits at 40 positions across 25-, 474-,
and 1,025-token inputs. The longest check includes two full 512-token batches
followed by a cached decode step. All 40 greedy predictions match. The maximum relative RMS
logit error is 0.6494%. A separate token-at-a-time run over the 474-token input
also matches all 21 saved greedy predictions, with maximum relative RMS error
0.8735%. The existing gate remains 1% relative RMS error with exact greedy
agreement. These checks do not establish equivalence for every prompt or for a
full 256K conversation.

The six synthetic GPU tests cover both attention types, grouped recurrent heads,
batches from 2 through 512, cached continuation, split prefill, and reset. Scalar
checks cover Hadamard transforms, packed projections at native widths, all four
weight codes, independent affine biases, symmetric negative scales, partial tiles,
and output guards. Attention checks cover the 256-, 4,096-, and 8,192-token
boundaries. GPU sampling checks ties, NaN, infinities, and vocabulary tails.
Existing tolerances are unchanged. Workspace tests, formatting, and strict
Clippy also pass. Native CLI generation with batched prefill and GPU sampling
returns `437` for `19 times 23` with reasoning disabled.

Run the GPU suite with:

```sh
cargo test --release -p rink-bonsai --test metal -- --ignored --test-threads=1
```
