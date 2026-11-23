# Smaller workgroups for prefill projections

This pass starts from `a0e92fb` on the 48 GiB Apple M4 Max, with checkpoint
revision `6e7c8248f75265bdd4fbab5273304243b5d8c2cb`. It keeps native quantized
weights, FP32 activations and accumulation, the model schedule, KV precision,
batch limits and device tensor buffers.

## Full-model text results

The alternating comparison gives these median prefill rates:

| Prompt | Reference | Updated | Gain | Identical peak Metal allocation |
| --- | ---: | ---: | ---: | ---: |
| 512 tokens | 687.6 tok/s | 751.7 tok/s | 9.3% | 26.99 GiB |
| 8,192 tokens | 551.2 tok/s | 564.6 tok/s | 2.4% | 27.46 GiB |

There are six trials per version per prompt length. All 24 generated sequences
match. The 8K timings vary substantially, and four of its six paired rounds
favor the update. Its improvement is modest; the 512-token gain is more
consistent. These measurements do not establish a decode speedup.
[Alternating trial records](m4-max-prefill-compact-alternating.json)

## Full-image result

The same 5,344-by-3,044 image produces 10,318 image tokens and a 10,342-token
prompt. One sequential pair, reference first, gives:

| Phase | Reference | Updated |
| --- | ---: | ---: |
| Image encoding | 35.59 s | 34.70 s |
| Prompt prefill | 20.45 s | 19.53 s |
| Total preparation, including encoder loading | 57.13 s | 55.40 s |
| Peak Metal allocation | 28.79 GiB | 28.79 GiB |

Each process starts with an 8K text warmup. Image resolution and token count
are unchanged. Both builds match the private continuation snapshot exactly.
This is one paired observation, not a repeated-trial median.
[Image records](m4-max-prefill-compact-image.json)

## Selected changes

Native expert projections use 128 threads per workgroup instead of 256. Two
SIMD groups load activations, and two load weights. Each loader reads 16 adjacent
values. NVFP4 weights share a scale lookup across those values; MXFP8 weights
share their scale conversion. The existing 16-, 32- and 64-slot tail choices
remain in use. The 8-by-8 FP32 matrix operations retain their original order.

Two staging banks share the existing 16 KiB scratch allocation. While one bank
is read, the next tile is written to the other bank. A barrier protects both
the next read and the next reuse. Weight fragments load transposed from
contiguous staged rows. No full-precision weight copy is stored in device memory.

A second kernel applies this scheduling to large BF16 dense projections. It
uses the existing 64-token by 64-output geometry and the same 16 KiB scratch.
Runtime selection requires a 64-token tile, columns divisible by 32, and either
at least 2,048 output rows or a batch of at least 4,096 tokens. Smaller shapes,
other weight formats and column tails retain their previous kernel. The image
encoder and decoder share this kernel.

Single-token decode kernels, attention, image preprocessing and image resolution
are unchanged. The Metal command submission implementation is unchanged.

A diagnostic 8K profile put about 37% of kernel time in expert projections,
34% in attention and 17% in large dense projections. Profiling changes dispatch
scheduling, so these shares guide target selection, not throughput claims.
[Initial profile](m4-max-prefill-compact-profile-before.json)

## Measurement method

Full-model text trials use 256K reserved capacity, FP16 KV, a 1,024-token batch
limit and 32 fixed decode steps. The primary comparison alternates the reference
and updated projections in one loaded checkpoint, with six trials per version
at each prompt length. Each round reverses the order. The reference reproduces
the native expert entry points from `a0e92fb` and uses the unchanged dense
kernel. All other kernels, tensors and model operations are shared.

The model runs four decode warmup tokens before measurements. Each trial starts
with empty history. GPU jobs run sequentially, with kernel profiling disabled.
Loading is excluded; first-use scratch allocations are included. No trials are
removed. The benchmark checks all 32 generated tokens on every trial.
The 512-token trials precede the 8K trials, so the latter retain earlier scratch
buffers. Their peak can differ from a process that runs only 8K prompts.

An initial pair in separate processes gave a 5.7% increase at 512 tokens and
a 1.4% decrease at 8K. These mixed results prompted the alternating comparison;
the initial pair is retained separately and is not pooled with it.
[Separate-process records](m4-max-prefill-compact-separate.json)

These are short-prompt measurements at 256K reserved capacity, not measurements
with 256K of active history. Decode timing is recorded as a regression check.

## Validation and memory

The projection tests compare every output bit, with independently poisoned
outputs and guards. Dense cases include FP32, BF16 and FP16 weights, wide
exponents, cancellation, token and row tails, and unaligned input offsets.
Native expert cases include both quantized formats and projection directions,
sparse routing, empty experts, and partial tiles.

All 59 workspace tests pass. The full-model FP16- and FP32-KV comparisons match
49,214,268 logits bit for bit against the established reference already matched
by the baseline build. The image comparison adds 6,401,856 exact continuation
logits, for 55,616,124 total. The independent CPU vision reference checks 1,495,296
features with zero tolerance violations and a maximum absolute error of 1.08e-5.
No numerical tolerance is increased.
[Validation records](m4-max-prefill-compact-validation.json)

The allocation schedules through 256K and 1M exactly match the previous build.
Warmed decode, repeated 8K prefill and repeated images retain zero Rust heap
allocations and zero device allocations in the allocation audit. These are
allocation simulations; they do not measure GPU throughput or process RSS.
The 1M reservation still exceeds this Mac's physical memory.
[Allocation records](m4-max-prefill-compact-memory.json)

## Rejected experiments

Wider local-attention query tiles, direct activation-fragment reads, and early
command submission did not establish a full-model improvement. Dense workgroup
reordering and 256-thread staging changes showed only small isolated gains.
These experiments are not part of the production changes.

## Reproduce

Run GPU jobs one at a time:

```sh
cargo test --release --workspace -- --include-ignored --test-threads=1
cargo run --release -p rink-inkling --example compare_prefill -- 6 both
cargo run --release -- bench --context 262144 --prompt-tokens 512,8192 \
  --decode-tokens 32 --repetitions 3 --prefill-batch 1024
cargo run --release -p rink-inkling --example bench_dense_compact -- 1024
cargo run --release -p rink-inkling --example bench_dense_compact -- 41273
cargo run --release -p rink-inkling --example bench_expert_specialized
cargo run --release -p rink-inkling --example prefill_memory -- \
  checkpoints/inkling-2-mini 262144
cargo run --release -p rink-inkling --example allocation_audit -- \
  checkpoints/inkling-2-mini 262144
```

Kernel benchmarks alternate timing order, exclude one warmup, check exact
outputs and guard bytes, and allocate no device buffers during dispatch.
Repeated synthetic weights and routes do not predict full-model speedups.
[Selected kernel probes](m4-max-prefill-compact-kernels.json)
