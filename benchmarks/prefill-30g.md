# Prefill with a 30 GiB memory target

Measured on November 22, 2026, on an Apple M4 Max with 40 GPU cores and
48 GiB of unified memory. The checkpoint is the original Inkling 2 Mini release,
revision `6e7c8248f75265bdd4fbab5273304243b5d8c2cb`.

The default prefill batch is now 1,024 tokens. Text prefill at 262,144-token
capacity peaks at 28.83 GiB of Metal allocation. The native weight formats,
FP32 activations and accumulation, FP16 KV default, and all three model passes
stay the same.

## Kernels

Local attention shares each key/value load across the three query heads that
use that KV head. Each head keeps its original dot products, relative-position
bias, softmax reductions, 128-key partitions and ordered value sums. The ring
cache and query-group offsets retain their existing meaning.

Expert multiplication shares decoded weights across larger groups of routed
tokens. Batches of 8 through 256 tokens keep eight-slot tiles. Larger batches
use 32-slot tiles with 32 input columns. Smaller batches keep their vector
kernels. Each 8-by-8 FP32 accumulator retains the original operation order and
zero padding. NVFP4 uses the existing exact decode
table. The new expert kernel uses 12 KiB of threadgroup memory, with no new
large buffer.

At a batch of 1,024, isolated local attention falls from 25.99 to 12.13 ms.
The native NVFP4 up projection falls from 18.61 to 11.43 ms; the MXFP8 up
projection falls from 25.72 to 13.14 ms. These timings use synthetic inputs.
Expert routes are balanced, and repeated weights can stay cached.
[Kernel measurements](m4-max-prefill-30g-kernels.json)

## Full-model text benchmark

Each build loads the same checkpoint and reserves a 262,144-token context.
Each of three trials starts with an empty conversation, prefills an 8,192-token
repeated text prompt, then runs 32 fixed decode steps. The maximum batch is
1,024 in every run. Loading and four warmup tokens are excluded; the first
trial includes scratch allocation. GPU jobs run one at a time. Per-kernel
profiling is off.

| Build | Prefill trials, tokens/s | Median | Peak Metal allocation |
| --- | --- | ---: | ---: |
| Baseline `0b6d93d` | 289.1, 259.0, 254.9 | 259.0 | 28.83 GiB |
| Selected: shared-head attention, 32-slot experts | 419.3, 376.7, 398.7 | 398.7 | 28.83 GiB |
| Experiment: shared-head attention, 64-slot experts | 413.3, 415.8, 410.6 | 413.3 | 28.83 GiB |

The selected version improves median throughput by 53.9% at the same memory
budget. The 64-slot experiment has a slightly higher text median but no
consistent gain across text and image runs. Its implementation is not retained.
All generated tokens match across all nine trials.
[Full measurements](m4-max-prefill-30g.json)

The throughput table covers an 8K prompt. At long histories, the existing
full-attention kernel takes a larger share of the total time. See the
[earlier long-context measurements](long-context.md) for that workload.

Run the text benchmark with:

```sh
cargo run --release -- bench --context 262144 --prompt-tokens 8192 \
  --decode-tokens 32 --repetitions 3 --prefill-batch 1024
```

Run the isolated benchmarks with:

```sh
cargo run --release -p rink-inkling --example bench_expert_tiles
cargo run --release -p rink-inkling --example bench_attention_heads
```

## Image benchmark

The original 5,344 by 3,044 image produces 10,318 image tokens and a
10,342-token prompt. Each variant runs once with a fresh model load,
262,144-token capacity and a 1,024-token batch limit.

| Build | Image encoding | Prompt prefill | Prefill tokens/s | Total preparation |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 44.39 s | 44.60 s | 231.9 | 90.19 s |
| Selected, 32-slot experts | 43.97 s | 28.63 s | 361.3 | 73.88 s |
| Experiment, 64-slot experts | 47.23 s | 30.67 s | 337.2 | 79.15 s |

The selected version reduces image prefill time by 35.8%. Peak Metal allocation
is 29.51 GiB in all three runs. Image encoder code is unchanged; the encoding
times show run-to-run variation. The complete prompt, 32 generated tokens and
all 6,401,856 continuation logits match the private baseline in every run.
[Image measurements](m4-max-prefill-30g-image.json) exclude private content.

The memory target describes Metal allocation for this configuration and tested
image. CPU image buffers, file mappings and other applications are outside
that total. Larger images or FP32 KV storage can use more memory.

## Allocation audit

The allocation simulator runs the real decoder schedule through each full
context. It tracks buffer sizes and lifetimes, including pending GPU resources,
without executing tensor operations. The prefill allocation peak stays constant
from the first full batch to the end of each context. Batches stay at 1,024 tokens.

| Context capacity | Peak tracked allocation |
| ---: | ---: |
| 4,096 | 20.21 GiB |
| 65,536 | 22.26 GiB |
| 262,144 | 28.83 GiB |
| 1,048,576 | 55.12 GiB |

The audit excludes GPU-driver overhead, allocation page rounding, image
processing and process RSS. The 1M result is a size/lifetime audit; that
configuration still exceeds this Mac's memory. It does not establish 1M
full-model throughput or quality. [Allocation records](m4-max-prefill-30g-memory.json)

```sh
cargo run --release -p rink-inkling --example prefill_memory -- \
  checkpoints/inkling-2-mini 262144
```

## Correctness checks

All 51 workspace tests pass, including the ignored Metal tests, run with one
GPU test at a time. The expert tests compare every output bit for native
quantized and F32 weights, shared and per-expert inputs, sparse routes, empty
experts, odd dimensions and partial tiles. Guard regions check output and tile
list bounds. Attention tests compare partials and final outputs across ring
wraps, fresh/cached KV boundaries and query groups, in FP16 and FP32 KV modes.
They include a local ring at positions near 256K and 1M. Existing full-attention
and independent analytic-reference tests also pass.

The full-model regression compares 49,214,268 logits with the established
FP16-KV and FP32-KV references. Every bit matches. Cases include an 8K prompt,
code, conversation, prefix boundaries and 32 continuation steps per case.
These runs use 16,384-token capacity. The separate image test compares another
6,401,856 continuation logits at 262,144-token capacity, also bit for bit.
[Text comparison records](m4-max-prefill-30g-validation.json)

The release build, formatting check and Clippy with warnings denied also pass.

```sh
cargo test --release --workspace -- --include-ignored --test-threads=1
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
```
