# Metal inference benchmarks

See [TensorOps prefill projections](tensor-prefill.md) for the latest kernel
improvements, paired measurements and correctness checks.

See [prefill speed and memory budgets](long-context.md) for the latest long-context
measurements and batch settings.

Initial optimization pass, measured November 20, 2026 on an Apple M4 Max with 48 GiB
unified memory, using the original Inkling 2 Mini checkpoint at revision
`6e7c8248f75265bdd4fbab5273304243b5d8c2cb` and a 4,096-token context limit.

| Prompt tokens | Original prompt tok/s | Optimized prompt tok/s | Original decode tok/s | Optimized decode tok/s |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 12.0 | 72.4 | 11.7 | 18.0 |
| 128 | 11.5 | 71.6 | 10.9 | 16.7 |
| 1,024 | — | 67.9 | — | 16.4 |

These are medians of two original trials and three optimized trials, each with a
fixed 16-token continuation. Prompt processing is about 6× faster, and decoding
about 1.5× faster, for the two comparable prompt lengths. No original measurement
was taken at 1,024 tokens. Every generated token ID in the optimized 32- and
128-token trials matches the corresponding original trial.

## Parallel residual reads

The next pass distributes residual-block reductions across eight SIMD groups and
the output across 256 threads, replacing a single SIMD group per token. Each
block's reduction and the weighted sum retain their original arithmetic order.

Fresh measurements with kernel profiling disabled, on the same machine and
checkpoint with three trials and 16 decode tokens per trial:

| Prompt tokens | Before prompt tok/s | After prompt tok/s | Before decode tok/s | After decode tok/s |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 72.3 | 73.9 | 17.9 | 20.3 |
| 128 | 71.2 | 73.1 | 16.8 | 19.1 |
| 1,024 | — | 69.2 | — | 19.0 |

This improves decoding by about 14% and prompt processing by 2–3% for the two
prompt lengths measured immediately before and after the change. The 1,024-token
comparison from the earlier pass is historical rather than a fresh paired run.
All 144 generated token IDs in the nine new trials match the prior recorded
outputs. Steady-state decode still allocates no GPU buffers and reads eight bytes
per token. The first prefill trial includes scratch allocation, as in the initial
measurements.

- [Before residual parallelism](m4-max-residual-before.json): revision `d4615c8`.
- [After residual parallelism](m4-max-residual-after.json): revision `a6ce376`.

Each JSON includes the exact benchmark command. Loading took 12.5 and 20.2 seconds
in these two runs; the earlier recorded runs took 33–34 seconds. File-cache and
shader-cache state were not controlled, so these do not establish a loading-time
improvement. Loading is excluded from the throughput figures.

## Routing, controllers, and grouped experts

The follow-up pass makes three changes while retaining native FP4/FP8 checkpoint
storage and float32 activations:

- Routing uses a SIMD group per token, with stable lowest-index ties and unbiased
  mixture weights. Wide RMS normalization and controller-input normalization use
  256 threads; small rows retain the single-SIMD path.
- Controller forget/gain projections and gating execute in one kernel, using
  vector products for decoding and matrix tiles for prefill. This removes 192
  dispatches per decode token (2,113 to 1,921). Its isolated throughput measurements
  were roughly unchanged; the dispatch reduction alone is not a measured speedup.
- For prompt chunks of at least eight tokens, GPU kernels compact routing slots
  by expert, build nonempty eight-token tiles, gather inputs, and scatter the
  matrix results back into their original slots. Up/down projections reuse that
  grouping. Smaller chunks and single-token decoding keep vector products. There
  is no CPU routing readback or persistent dequantized copy of the expert weights.

Fresh before/after medians on the same M4 Max and checkpoint:

| Prompt tokens | Before prompt tok/s | After prompt tok/s | Before decode tok/s | After decode tok/s |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 73.4 | 100.8 | 20.4 | 24.7 |
| 128 | 72.9 | 101.1 | 19.4 | 22.9 |
| 1,024 | — | 92.6 | — | 23.1 |

Prompt processing improves by 37–39%, and decoding by 18–21%, at the two freshly
paired lengths. No fresh before measurement was taken at 1,024 tokens. Routing
and normalization account for the measured decode improvement; grouped expert
projections account for the prefill improvement.

The short-chunk comparison isolates grouping against the already-fused controller
version. Median prefill throughput is effectively unchanged at eight tokens
(33.8 to 33.7 tok/s), improves from 54.6 to 60.8 tok/s at 16 tokens, and from 68.5
to 85.3 tok/s at 24 tokens. These results support the eight-token threshold on this
checkpoint and machine; other workloads and GPUs may favor a different threshold.

Raw measurements and source revisions:

- [Before this pass](m4-max-followup-before.json): `020acbe`, three trials.
- [Routing and normalization](m4-max-followup-reductions.json): `c6b847f`, two trials.
- [Controller fusion](m4-max-followup-controllers.json): `cc786d8`, two trials.
- [Short-chunk comparison before grouping](m4-max-followup-short-before.json):
  `cc786d8`, three trials.
- [Final measurements](m4-max-followup-final.json): `fb2eed6`, three trials at each
  of six prompt lengths, including a guard for nonfinite routing inputs.

All runs use 16 decode tokens and context 4,096 with profiling disabled. The first
trial at a larger chunk size includes scratch allocation; no trials are removed.
Only two trials were taken for the intermediate stages, so their 32-token prefill
medians mix allocation and reuse. The before/after comparison uses three trials.

All 288 generated token IDs in the final 18 trials match the previous recorded
outputs at their respective prompt lengths. Decode reads back eight bytes and
dispatches 1,921 kernels per token. The first decode trial grows one 2,352-byte
scratch buffer; every subsequent decode trial and every reset allocates no GPU
buffers. These counters and cold allocation costs are retained in the raw results.

The 27-test suite covers finite and nonfinite routing, stable ties, normalization
tails and zero rows, fused controller weight formats and tile boundaries, grouped
FP4/FP8/float32 experts with sparse routes and empty groups, and decoder cache/reset
behavior. Native block widths, non-aligned float32 rows, and grouping across a
32-token boundary are exercised explicitly. Workspace formatting and Clippy checks
pass. A full-checkpoint chat smoke test also passes multi-turn recall, reset, and
clean answer display on the final executable.

## Residual and SwiGLU fusion

The next pass fuses the two smaller groups of adjacent operations:

- During decoding, expert summation, static convolution, and residual addition
  run in one kernel.
  Attention-output convolution also performs its residual addition except at a
  residual-block boundary. Each channel retains ownership of its convolution
  history across a prompt chunk, storing the pre-convolution input in the ring.
  This removes 136 dispatches per token, from 1,921 to 1,785.
- Each expert gate/up vector projection applies SwiGLU before writing its result.
  A SIMD group computes one gate/up pair with shared input loads. This removes 48
  further dispatches and the gate/up intermediate buffers, reaching 1,737 per token.

The first version also fused these operations during large prompt chunks, including
the grouped expert matrix projection. Prefill measured about 1–2% slower in repeat
comparisons. A second experiment shared inputs across four gate/up pairs per SIMD
group but slowed decoding by about 6%; that experiment was reverted. The retained
version applies fusion to decode and chunks shorter than eight tokens, preserving
the original parallel kernels for larger prefill chunks. Decode removes 184
dispatches (9.6%). All math and caches remain float32, with native packed FP4/FP8
expert weights.

Raw measurements, in execution order:

- [Initial baseline](m4-max-fusion-before.json): `a5085cb`.
- [Residual/convolution fusion](m4-max-fusion-residual.json): `6f66315`.
- [Initial SwiGLU fusion](m4-max-fusion-initial.json): `78f2c5e`.
- [Repeated baseline](m4-max-fusion-baseline-repeat.json): `a5085cb`.
- [Four-pair experiment](m4-max-fusion-row-reuse.json): `27aeb5a`, reverted.
- [Retained decode fusions](m4-max-fusion-final.json): `eedfdce`.

All runs use three trials, 32 decode tokens, context 4,096, and profiling disabled.
Cold scratch allocations are included; load time is reported separately. The
repeated baseline and final run use the same prompt lengths and order.

Median throughput on the same M4 Max and checkpoint:

| Prompt tokens | Baseline prompt tok/s | Final prompt tok/s | Baseline decode tok/s | Final decode tok/s |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 33.8 | 33.9 | 25.4 | 25.4 |
| 32 | 101.1 | 100.3 | 25.1 | 24.9 |
| 128 | 101.5 | 101.3 | 23.5 | 23.3 |
| 1,024 | 92.8 | 92.7 | 23.0 | 23.1 |

The final medians differ from the repeated baseline by at most 1%. This pass
reduces dispatch count and intermediate writes during decoding, but does not
establish a throughput improvement. The larger prefill chunks retain their
original dispatch counts. All 384 generated token IDs match the repeated baseline.
Every final decode trial allocates no GPU buffers, uses one command buffer and
encoder per token, and reads eight bytes per token. Every reset also allocates no
GPU buffers.

The 28-test suite compares fused operations with the separate kernels, including
FP4/FP8/float32 weights, sparse expert groups, gate/up pair ordering, partial matrix
tiles, convolution history wraparound, chunks longer than the history ring, and
resets without clearing state. Tiny-model scalar and batched comparisons also pass,
along with workspace formatting and Clippy checks. A full-checkpoint chat smoke
test passes multi-turn recall, reset, and clean answer display on the final build.

## FP16 caches and 256K chat

Revision `4adbfa0` defaults to FP16 KV-cache storage and shares a single attention
partition workspace across all layer visits. Attention arithmetic, softmax,
convolution history, and activations remain FP32. Fresh keys/values are rounded
before attention as well as cache writes, so prompt chunking does not change the
storage precision. `--kv-cache f32` retains the FP32 comparison path.

At a capacity of 262,144 tokens, calculated from the released configuration:

| Allocation | Previous implementation | FP16 and shared workspace |
| --- | ---: | ---: |
| KV caches, all 48 visits | 16.31 GiB | 8.16 GiB |
| Attention partitions, batch 32 | 6.21 GiB | 780 MiB |

The eight full-attention visits keep independent histories; the other 40 visits
keep 1,024-token rings. Workspace reuse relies on the accelerator's ordered command
stream: each attention result consumes its partials before the next visit overwrites
them. Workspace is sized for the configured capacity and grows only when a larger
batch requires it. Cache capacity is allocated up front. No extra dispatches or CPU
synchronizations are needed for either change.

The 256K default applies to `chat`. `bench` remains at 4K and `generate` at 2K unless
given an explicit `--context`. Benchmark JSON now includes `kv_cache`, the Metal
recommended working-set limit, and `device_allocated_bytes` at phase boundaries;
the existing `allocated_bytes` counter is cumulative allocation traffic per phase.

The [full-checkpoint 256K capacity run](m4-max-fp16-256k.json) on the M4 Max loaded
in 34.3 seconds, reporting 25.76 GiB of Metal allocations after load and at most
27.19 GiB at the recorded phase boundaries (37.44 GiB recommended limit). After
the first prompt, allocations settled at 27.17 GiB. Decode still used zero GPU
allocations, 1,737 dispatches, one encoder/command buffer, and eight readback bytes
per token. These are phase-boundary allocation readings, not a continuously sampled
process-memory peak.

| Prompt tokens | FP16 prefill tok/s | FP16 decode tok/s |
| ---: | ---: | ---: |
| 32 | 65.75 | 25.23 |
| 128 | 101.24 | 23.26 |
| 1,024 | 91.07 | 23.10 |
| 8,192 | 80.88 | 21.86 |

This is one trial per length, with 32 fixed-work decode tokens and profiling
disabled. The first prompt includes allocation of the larger workspace. It is a
capacity and correctness smoke run, not evidence of a throughput improvement.
The cache reserves the complete 256K capacity, but the longest full-model prompt
in this run contains 8K tokens.

The [FP32 comparison run](m4-max-fp32-cache-comparison.json) uses the same prompt
and decode lengths with 16K reserved capacity to reduce test memory. Its outputs
for 32/128/1,024-token prompts match all 96 corresponding token IDs in the earlier
FP32 fusion baseline. FP16 matches 32/32, 29/32, 8/32, and 32/32 generated IDs at
32/128/1,024/8,192 prompt tokens, respectively. Once greedy sampling chooses a
different token, subsequent histories differ, so these counts are not accuracy
scores or independent next-token comparisons.

For a comparison on identical histories, run:

```sh
cargo run --release -p rink-inkling --example compare_kv_cache -- \
  checkpoints/inkling-2-mini > precision.json
```

This loads one model at a time, compares full-vocabulary logits after 128, 1,024,
and 8,192 prompt tokens, and then feeds both cache formats the same 32 continuation
tokens chosen by FP32. It reports centered logit RMSE, maximum absolute error,
KL divergence from the FP32 distribution, and greedy-token agreement.

The [recorded comparison](m4-max-kv-precision.json) agrees on the greedy token at
all 35 sampled positions. Mean KL divergence is 0.00241 nats (maximum 0.04022).
The full-vocabulary logits do change: mean centered RMSE is 0.481, its maximum
is 1.808, and the largest absolute logit difference is 9.372. These measurements
describe numerical sensitivity on one repeated-text prompt, not general model
quality or proof that quantization never changes answers.

The 31-test suite includes separate FP16/FP32 scalar decoder references, chunked
versus single-token parity, cache rounding and ring guards, attention comparisons
at 8K/64K/256K positions, and memory-growth/reuse checks across chunks and resets.
An additional full-geometry test uses 24 query heads, eight KV heads, dimension
128, a 32-token batch and 262,144 cached positions. It retrieves a marked first
token and checks every output against an analytic softmax, with unwritten cache
rows poisoned to detect incorrect reads. This exposed rounding drift in long
partition reductions. Compensated FP32 sums now merge more than 128 partitions;
the accumulation order at shorter contexts is retained.
The [final release check](m4-max-256k-final.json) at revision `31eac49` reserves a
256K FP16 cache and processes a 1,024-token prompt with 32 decode steps. All 32
generated IDs match the pre-compensation FP16 run, with zero decode allocations.
An interactive chat smoke test with no flags confirmed the 262,144-token context,
FP16 cache, effort 0.9 and 512-token reply defaults. It answered `OK`, recalled `57`
on the next turn, and answered `4` after `/reset`, without exposing protocol markers
or reloading the model.
These kernel checks exercise long addresses and full cache scans; they do not
establish end-to-end model accuracy on a 256K-token prompt.

## Projection scheduling with identical arithmetic

Revision `f090215` changes how Metal schedules projections. It retains FP32
activations and arithmetic, native expert weight formats, FP16/FP32 cache options,
and 32-token prompt chunks. Fast math remains disabled.

- Packed dense vectors give each output row its own SIMD group. Each row retains
  the original four-element dot products and reduction order. Small projections
  keep scalar products and use 32-thread groups. The scalar/packed shape boundary
  is unchanged because those two algorithms have different reduction orders.
- Dense prompt projections share staged inputs and weights across a 16-token by
  32-row tile for batches of at least 16 tokens. Each 8-by-8 matrix accumulator
  performs the same ordered operations as the original kernel.
- Grouped expert projections share gathered inputs across 32 output rows. Weight
  conversion, matrix accumulation, routing order, and output placement are unchanged.

Fresh medians on the same M4 Max, using three trials per prompt length, 32 decode
tokens per trial, FP16 caches, and a 262,144-token capacity:

| Prompt tokens | Before prefill tok/s | After prefill tok/s | Before decode tok/s | After decode tok/s |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 100.73 | 150.99 | 25.62 | 28.08 |
| 128 | 101.34 | 152.57 | 23.91 | 25.76 |
| 1,024 | 92.55 | 133.40 | 23.01 | 25.91 |

Prompt processing improves by 44–51%, and decoding by 8–13%. All 288 generated
token IDs match the baseline. Decode retains zero buffer allocations, 1,737 kernel
dispatches, one encoder/command buffer, and eight readback bytes per token.
Steady-state Metal allocations remain 27.17 GiB. Loading is excluded from throughput;
the first prompt's allocation cost is included. Filesystem and shader-cache state
are not controlled, and the before and after runs execute sequentially.

- [Before](m4-max-projection-before.json): `8c179d0`.
- [After](m4-max-projection-after.json): `f090215`.
- [Baseline kernel profile](m4-max-projection-profile-before.json): diagnostic
  sampled intervals, with profiling overhead.
- [Isolated projection timings](m4-max-projection-kernels.json): repeated buffers
  can remain cached; these measurements do not establish full-model throughput.

All 32 workspace tests pass. New checks require identical output bits for dense
projections across FP32, BF16, and FP16 weights, varied magnitudes, partial tiles,
input offsets, and output guards. Expert checks compare both matrix kernels bit
for bit with NVFP4, MXFP8, and FP32 weights, sparse routes, both input layouts,
empty groups, and partial tiles. Existing scalar decoder, cache, reset, and full
256K attention tests also pass. Formatting and Clippy checks pass.

Full-checkpoint comparisons use three token histories: repeated text through 8,192
tokens, a code prompt, and a conversation-style prompt. Each has 32 continuation
tokens chosen by the baseline. Intermediate checks cross prompt chunk boundaries
and the sliding-window boundary; the model resets between histories. Both
[FP16 caches](m4-max-projection-logits-f16.json) and
[FP32 caches](m4-max-projection-logits-f32.json) match every output bit: 24,607,134
logits at 123 positions per precision, or 49,214,268 logits total. These comparisons
use 16K cache capacity and establish agreement on these histories on the M4 Max.

To reproduce, copy `models/inkling/examples/compare_revisions.rs` into the baseline
checkout if needed. Build the example in each checkout. Record with the baseline
executable, then compare with the candidate executable:

```sh
# Baseline build:
cargo run --release -p rink-inkling --example compare_revisions -- \
  record /absolute/path/to/checkpoint /tmp/rink-reference.json f16 8192

# Candidate build:
cargo run --release -p rink-inkling --example compare_revisions -- \
  compare /absolute/path/to/checkpoint /tmp/rink-reference.json > comparison.json
```

Repeat with `f32` when recording the second reference. The comparison exits with an
error if any logit bit differs. Snapshots store integer float bits to avoid decimal
rounding; each reference is about 258 MiB and should remain outside the repository.

Run isolated projection measurements with:

```sh
cargo run --release -p rink-inkling --example bench_projections
```

## Prefetch and parallel residual reads

Revision `df78ec0` reduces decode latency in two places:

- Scalar projections load four successive lane inputs before consuming them in
  their original order. Weight conversion, multiplication, and summation stay the
  same. The scalar/packed projection boundary is unchanged.
- Residual reads also prefetch inputs. Several threadgroups repeat the same score
  calculation, then each writes a separate set of output channels. No reduction is
  split across groups. Native decode uses eight output shards; prompt batches of
  eight or more tokens use four. Small hidden widths use fewer shards.

Fresh M4 Max medians, with three trials per prompt length, 32 decode tokens per
trial, FP16 caches, and a 262,144-token capacity:

| Prompt tokens | Before prefill tok/s | After prefill tok/s | Before decode tok/s | After decode tok/s |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 151.43 | 152.30 | 27.51 | 34.38 |
| 128 | 152.35 | 154.60 | 25.91 | 31.32 |
| 1,024 | 133.57 | 134.93 | 25.35 | 30.99 |

Decoding improves by 21–25%. Prefill changes by only 0.6–1.5%; treat it as roughly
unchanged. All 288 generated token IDs match. Decode still uses zero buffer
allocations, 1,737 dispatches, one encoder/command buffer, and eight readback bytes
per token. Steady-state Metal allocations remain 27.17 GiB. Loading is excluded;
the first prompt's allocation cost is included. Before and after runs are
sequential, with filesystem and shader-cache state uncontrolled.

- [Before](m4-max-prefetch-before.json): `277cfbe`, with engine code from `f090215`.
- [After](m4-max-prefetch-after.json): `df78ec0`.
- [Baseline kernel profile](m4-max-prefetch-profile-before.json): sampled diagnostic
  intervals, with profiling overhead.
- [Isolated kernel timings](m4-max-prefetch-kernels.json): includes scalar
  projections and residual reads with one, four, and eight output shards. Repeated
  buffers can remain cached, so use the full-model runs to establish throughput.

All 32 workspace tests pass, as do formatting and Clippy checks. Scalar prefetch
must match every output bit across three weight dtypes, offsets, varied magnitudes,
and column tails. Residual tests preserve the original input distribution and add
wider magnitudes, partial prefetch groups, batch sizes 1/3/32, bank/query offsets,
one/four/eight shards, and output guards. No numerical tolerance was loosened.

The [FP16](m4-max-prefetch-logits-f16.json) and
[FP32](m4-max-prefetch-logits-f32.json) full-checkpoint comparisons match all
49,214,268 logits bit for bit at 246 positions. They reuse the reference snapshots from
`8c179d0` that `f090215` already matched above; each result records the snapshot's
SHA-256. The three histories include an 8,192-token prompt, 32 continuation tokens
per history, chunk and sliding-window boundaries, and resets. This establishes
agreement on those histories on the M4 Max, using 16K cache capacity.

Use the comparison and isolated-kernel commands above to reproduce these checks.
For full-model throughput, run each revision with:

```sh
cargo run --release -- bench --prompt-tokens 32,128,1024 --decode-tokens 32 \
  --repetitions 3 --context 262144 --kv-cache f16 > benchmark.json
```

## Four-wide staging for dense prompt projections

Revision `3ad776b` loads four adjacent values at a time when it stages
activations and dense weights for matrix multiplication. It retains the 16-token
by 32-row tile, 32-column staging steps, float32 accumulators, and the original
ordered 8-column matrix operations. Short batches now use this tile too. Partial
vectors use scalar loads, and packed loads support unaligned input offsets.
Prompt chunks remain 32 tokens. Single-token projections and precision are unchanged.

Fresh M4 Max medians, with three trials per prompt length, 32 decode tokens per
trial, FP16 caches, and a 262,144-token capacity:

| Prompt tokens | Before prefill tok/s | After prefill tok/s | Before decode tok/s | After decode tok/s |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 34.93 | 77.51 | 36.60 | 36.80 |
| 16 | 92.61 | 126.12 | 36.58 | 36.51 |
| 32 | 154.11 | 196.20 | 35.75 | 35.80 |
| 128 | 155.57 | 197.76 | 32.48 | 32.63 |
| 1,024 | 135.32 | 165.16 | 32.25 | 32.25 |

Prefill improves by 22–27% at 32–1,024 tokens, 36% at 16 tokens, and 2.22× at eight
tokens. Decode changes by less than 0.6%; treat it as unchanged. All 480 generated
token IDs match. Decode retains zero buffer allocations, 1,737 dispatches, one
encoder/command buffer, and eight readback bytes per token. Allocation counters and
Metal memory at every recorded phase boundary match the baseline. Steady-state
allocations grow from 26.12 GiB at batch eight to 27.17 GiB at batch 32.

Loading is excluded. The first trial at each larger batch includes scratch
allocation; no trials are removed. Runs execute sequentially with profiling
disabled. Filesystem and shader-cache state are uncontrolled.

- [Before](m4-max-matrix-staging-before.json): `be26299`, with engine code from
  `df78ec0`.
- [After](m4-max-matrix-staging-after.json): `3ad776b`.
- [Baseline kernel profile](m4-max-matrix-staging-profile-before.json): sampled
  diagnostic intervals, with profiling overhead.
- [Isolated kernel timings](m4-max-matrix-staging-kernels.json): dense projections
  at batches 1, 2, 8, 16, and 32. Repeated buffers can remain cached; use the
  full-model runs to establish throughput.

All 32 workspace tests pass, as do formatting and Clippy checks. The matrix kernel
must match every output bit across FP32, BF16, and FP16 weights, varied magnitudes,
partial rows/columns/token tiles, unaligned input offsets, and output guards.
No numerical tolerance was loosened.

The [FP16](m4-max-matrix-staging-logits-f16.json) and
[FP32](m4-max-matrix-staging-logits-f32.json) full-checkpoint comparisons match all
49,214,268 logits bit for bit at 246 positions. They reuse the `8c179d0` reference
snapshots already matched by `df78ec0`; each result records the snapshot's SHA-256.
The three histories include an 8,192-token prompt, 32 continuation tokens per
history, chunk and sliding-window boundaries, and resets. These checks establish
agreement on those histories on the M4 Max, using 16K cache capacity.

Use the comparison and isolated-kernel commands above to reproduce these checks.
For full-model throughput, run each revision with:

```sh
cargo run --release -- bench --prompt-tokens 8,16,32,128,1024 --decode-tokens 32 \
  --repetitions 3 --context 262144 --kv-cache f16 > benchmark.json
```

## Decode attention scheduling and cache addressing

Revision `80340a6` speeds up attention during single-token decoding:

- Sixteen SIMD groups compute independent key scores when at most two 128-key
  partitions are visible. Larger grids use eight groups. The first four groups
  retain the original softmax reductions and ordered value sums.
- Each partition computes its first ring-cache slot outside the key loops. Valid
  indices span at most one cache capacity, so a conditional subtraction replaces
  the repeated remainder operations.
- The larger-grid kernel preloads four cache values and consumes them in their
  original order. Cache precision, fresh-token rounding, partition size, and
  compensated long-context reductions are unchanged.

Batched prefill retains its existing attention kernel. Fresh M4 Max medians use
three trials per prompt length, 64 decode tokens per trial, FP16 caches, and a
262,144-token capacity:

| Prompt tokens | Before prefill tok/s | After prefill tok/s | Before decode tok/s | After decode tok/s |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 195.33 | 194.41 | 33.66 | 35.27 |
| 128 | 196.34 | 195.92 | 31.36 | 33.56 |
| 256 | 189.21 | 190.92 | 31.29 | 32.69 |
| 512 | 180.15 | 179.78 | 31.24 | 32.73 |
| 1,024 | 160.76 | 165.73 | 31.02 | 32.41 |

Decode improves by 4.4–7.0% across these prompts. Prefill kernels are unchanged;
the observed prefill differences of −0.5% to +3.1% do not establish a speedup.
All 960 generated token IDs match. Decode retains zero buffer allocations, 1,737
dispatches, one encoder/command buffer, and eight readback bytes per token.
Allocation counters and Metal memory at each recorded phase boundary match the
baseline, with 27.17 GiB of steady-state Metal allocations.

Loading is excluded; the first prompt's scratch allocation cost is included.
Runs execute sequentially with profiling disabled. Filesystem and shader-cache
state are uncontrolled, and the measurements apply to this M4 Max and workload.

- [Before](m4-max-decode-attention-before.json): `29e0b3f`, with engine code from
  `3ad776b`.
- [After](m4-max-decode-attention-after.json): `80340a6`.
- [Baseline kernel profile](m4-max-decode-attention-profile-before.json): sampled
  diagnostic intervals, with profiling overhead.
- [Isolated kernel timings](m4-max-decode-attention-kernels.json): includes native
  attention geometry with FP16 and FP32 caches at histories from 32 to 16,384
  tokens. Repeated buffers can remain cached; use full-model runs for throughput.

All 32 workspace tests pass, as do formatting and Clippy checks. Both new kernels
must match every partition-output bit across FP16/FP32 caches, tiny and partial
rings, wraparound, causal chunks, empty partitions, output guards, and head widths
below and above 128. The full 256K geometry test also compares the native decode
kernel bit for bit, with NaNs in unwritten cache positions. No numerical tolerance
was loosened.

The [FP16](m4-max-decode-attention-logits-f16.json) and
[FP32](m4-max-decode-attention-logits-f32.json) full-checkpoint comparisons match all
49,214,268 logits bit for bit at 246 positions. They reuse the `8c179d0` reference
snapshots already matched by `3ad776b`; each result records the snapshot's SHA-256.
The three histories include an 8,192-token prompt, 32 continuation tokens per
history, chunk and sliding-window boundaries, and resets. These checks establish
agreement on those histories on the M4 Max, using 16K cache capacity; the synthetic
256K kernel test is separate from full-checkpoint validation.

Use the comparison and isolated-kernel commands above to reproduce these checks.
For full-model throughput, run each revision with:

```sh
cargo run --release -- bench --prompt-tokens 32,128,256,512,1024 --decode-tokens 64 \
  --repetitions 3 --context 262144 --kv-cache f16 > benchmark.json
```

## Native decode projections and exact FP4 tables

Revision `5803d6e` speeds up single-token projections without changing their
arithmetic:

- Each NVFP4 matrix gets a table of the exact float32 value for every expert,
  scale byte, and weight code. Metal builds the table once at load time, using
  the original ordered multiplications. Decode retains the original four-wide
  dots, SIMD reductions, activation functions, and native packed weight buffers.
- BF16 matrix-vector kernels specialize the common input widths. Small
  projections retain scalar sums; larger projections retain packed dots.
  Other widths and dtypes retain the generic kernels.
- Native MXFP8 gate/up and down projections use fixed dimensions. Weight
  decoding, scaling, dot products, and reduction order are unchanged.

Fresh M4 Max medians use three trials per prompt length, 64 decode tokens per
trial, FP16 caches, and a 262,144-token capacity. Loading is excluded. The fresh
baseline covers the three lengths below; the candidate also measures 256 and
512 tokens.

| Prompt tokens | Before prefill tok/s | After prefill tok/s | Before decode tok/s | After decode tok/s | Decode gain |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 193.69 | 194.63 | 35.53 | 39.26 | 10.5% |
| 128 | 195.94 | 196.22 | 33.38 | 37.58 | 12.6% |
| 1,024 | 165.56 | 160.38 | 32.46 | 36.19 | 11.5% |

Decode improves by 10.5–12.6% on the fresh comparison lengths. All 576
continuation token IDs in those paired trials match, and all 960 IDs in the
candidate's five-length run match the previously recorded `80340a6` results.
The 1,024-token prefill median is 3.1% lower in this run; the shorter prefill
medians change by less than 0.5%. These are measurements on one device, not a
claim of unchanged prefill performance.

The tables add 80 MiB for this checkpoint: 40 NVFP4 matrices, each with 128
experts and 4,096 float32 entries per expert. Table construction adds 40 load-time
dispatches and synchronizations. Steady-state decode still has zero allocations,
1,737 dispatches, one encoder/command buffer, and eight readback bytes per token.
No checkpoint files are changed. Batched prefill kernels are unchanged; these
measurements do not establish a prefill speedup.

All 34 workspace tests pass, including Metal hardware tests, as do formatting
and Clippy checks. Added checks cover every FP4 code and FP8 scale byte, signed
zero, output guards, shared and per-route inputs, sparse routes, short batches,
native MXFP8 dimensions, dense row tails, and input offsets. No numerical
tolerance was loosened.

Full-checkpoint comparisons match all 49,214,268 logits bit for bit across FP16
and FP32 cache runs. The three histories include an 8,192-token prompt and 32
continuation tokens per history, with chunk/ring boundaries and resets. They
reuse the `8c179d0` snapshots previously matched by `80340a6`; each result records
the snapshot SHA-256. The full-model comparisons use 16K cache capacity; the
existing native-geometry 256K attention test is separate.

- [Before](m4-max-native-decode-before.json): `bac7d73`, engine `80340a6`.
- [After](m4-max-native-decode-after.json): `5803d6e`.
- [FP16 logits](m4-max-native-decode-logits-f16.json).
- [FP32 logits](m4-max-native-decode-logits-f32.json).
- [Isolated kernels](m4-max-native-decode-kernels.json): repeated buffers can
  remain cached; do not treat these timings as full-model throughput.
- [Candidate kernel profile](m4-max-native-decode-profile.json): diagnostic
  stage-boundary intervals include profiling overhead and can overlap.

This pass did not reach the 2x decode target. Experiments with loop unrolling,
smaller shared FP4 tables, paired-code tables, residual-read specialization,
and a combined attention-projection dispatch did not justify inclusion. In
longer trials, combining the six attention projections reduced dispatches by
240 per token but improved median decode by only 0.6%, within measurement noise.
The original execution graph is retained.

Reproduce throughput on each revision with:

```sh
cargo run --release -- bench --prompt-tokens 32,128,1024 --decode-tokens 64 \
  --repetitions 3 --context 262144 --kv-cache f16 > benchmark.json
```

Use the comparison and isolated-kernel commands above for numerical validation
and kernel experiments. Runs execute sequentially. Filesystem and shader-cache
state are uncontrolled, and results apply to this M4 Max and workload.

## Reproduce

Build in release mode and run with the same local checkpoint:

```sh
cargo run --release -- bench --prompt-tokens 32,128,1024 --decode-tokens 16 \
  --repetitions 3 --context 4096 --kv-cache f32 > benchmark.json
```

The benchmark constructs deterministic prompts from repeated text, loads weights
once, warms the runtime with four single-token steps, and resets before each trial.
It decodes through EOS to keep the work fixed. Load and reset time are reported
separately and excluded from token throughput. The first optimized trial includes
allocation of larger scratch buffers for batched prefill (57.9 prompt tok/s); later
trials reuse them. All trials, including that first one, are retained in the JSON.

- [Original measurements](m4-max-baseline.json): source revision `7ea3e33`, prompt
  lengths 32 and 128, two repetitions, 16 decode tokens, context 4,096.
- [Optimized measurements](m4-max-optimized.json): source revision `a43646a`, command
  shown above.

The `--reference` option on the current executable compares single-token prefill
and CPU sampling against the optimized path. It uses the current math kernels and
does not recreate the original revision.

To identify costly GPU operations, run a separate diagnostic:

```sh
cargo run --release -- bench --profile-kernels --prompt-tokens 32,128 \
  --decode-tokens 4 --repetitions 2 --context 4096 > kernel-profile.json
```

This uses Metal stage-boundary timestamp counters, splitting each dispatch into its
own compute pass. It calibrates GPU timestamps against the CPU clock and aggregates
durations and invocation counts by kernel in each phase. The timestamp buffer rolls
over into another command buffer when full. Splitting passes and sampling counters
add overhead and alter scheduling: use these results to locate expensive kernels,
then judge an optimization with profiling disabled. Support for timestamp counters
is checked explicitly; an unsupported GPU reports an error.
Pass intervals can overlap and include stalls, so their durations are not additive
and their sum is not the command buffer's elapsed GPU time.

The implementation follows Apple's [counter sampling documentation](https://developer.apple.com/documentation/metal/sampling-gpu-data-into-counter-sample-buffers)
and [GPU clock calibration procedure](https://developer.apple.com/documentation/metal/converting-gpu-timestamps-into-cpu-time).

## Changes and costs

| Cost | Original | Optimized |
| --- | ---: | ---: |
| GPU allocations per steady-state decode token | 2,505 | 0 |
| Kernel dispatches per decode token | 3,157 | 1,737 |
| Compute encoders per decode token | 3,157 | 1 |
| Readback bytes per decode token | 800,232 | 8 |
| GPU allocations per conversation reset | 288 | 0 |

Scratch buffers and KV/convolution caches persist across tokens and resets. One
ordered compute encoder serves each decode step or prompt chunk. Residual attention
reads stored block states directly, removing repeated packing copies. Greedy GPU
sampling reads back only the selected token and a validity flag.

Dense prompt projections use SIMD matrix multiplication over chunks of up to 32
tokens. Causal attention and convolution preserve token order within each chunk;
only the final prompt token needs vocabulary logits. Dense and expert vector
products use packed weight loads, and attention reduces 128-key partitions before
merging their results with the attention sink and output gate.

The initial runs loaded in approximately 33–34 seconds; subsequent runs varied
with uncontrolled cache state. Removing redundant weight-buffer zeroing did not
produce a meaningful measured startup improvement.
GPU execution and CPU wait times overlap; do not sum those counters. Allocation
timing includes initialization/copy work inside the allocator, so its breakdown
changed when uploads began initializing buffers directly.

## Remaining work

The [earlier kernel diagnostic](m4-max-kernel-profile.json) at revision `a6ce376`
uses a 32-token prompt, four decode tokens and two trials. Approximate per-token
decode pass durations before the routing, normalization, and controller changes
are below. These are sampled intervals, not independent
parts of elapsed time; they overlap and must not be summed into a speedup estimate.

| Kernel | Dispatches per decode token | Sampled milliseconds per token |
| --- | ---: | ---: |
| Vectorized dense projections | 529 | 14.9 |
| Expert projections | 96 | 7.8 |
| Parallel residual reads | 208 | 5.8 |
| Small dense projections | 160 | 5.7 |
| RMS normalization | 202 | 5.0 |
| Expert routing | 48 | 3.5 |

Those three follow-up changes, residual/SwiGLU fusions, projection scheduling,
prefetch, parallel residual outputs, four-wide matrix staging, and decode attention
scheduling are implemented above.
Further experiments include combining independent attention projections, fusing
K/V convolution with normalization and cache updates during decoding, and tuning
matrix tiles and grouping thresholds across more prompt distributions. The earlier
controller fusion did not materially change throughput, so reducing dispatch count
should not be treated as proof of a speedup.

FP16 KV caches reduce long-context memory, but broader accuracy evaluation and
long-prompt attention optimization remain. A 1M context on a 48 GiB Mac would need
further cache compression and workspace reduction. Startup loading also warrants
its own I/O and copy profiling. Batching across requests and speculative decoding
are not implemented.

Validation includes Metal kernel comparisons against scalar results, tiny-model
single-token/batched parity across cache wraparound and resets, full-checkpoint
token comparisons above, and terminal chat smoke tests. These checks do not replace
a broader model accuracy evaluation.
