# Metal inference benchmarks

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

## Reproduce

Build in release mode and run with the same local checkpoint:

```sh
cargo run --release -- bench --prompt-tokens 32,128,1024 --decode-tokens 16 \
  --repetitions 3 --context 4096 > benchmark.json
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
| Kernel dispatches per decode token | 3,157 | 2,113 |
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

Load time remains approximately 33–34 seconds in these runs. Removing redundant
weight-buffer zeroing did not produce a meaningful measured startup improvement.
GPU execution and CPU wait times overlap; do not sum those counters. Allocation
timing includes initialization/copy work inside the allocator, so its breakdown
changed when uploads began initializing buffers directly.

## Remaining work

The [updated kernel diagnostic](m4-max-kernel-profile.json) at revision `a6ce376`
uses a 32-token prompt, four decode tokens and two trials. Approximate per-token
decode pass durations are below. These are sampled intervals, not independent
parts of elapsed time; they overlap and must not be summed into a speedup estimate.

| Kernel | Dispatches per decode token | Sampled milliseconds per token |
| --- | ---: | ---: |
| Vectorized dense projections | 529 | 14.9 |
| Expert projections | 96 | 7.8 |
| Parallel residual reads | 208 | 5.8 |
| Small dense projections | 160 | 5.7 |
| RMS normalization | 202 | 5.0 |
| Expert routing | 48 | 3.5 |

This suggests the next experiments, with ordinary throughput and output parity
checked after each change:

1. Parallelize expert routing (currently one thread per token) and use more threads
   for wide RMS normalization. These are small, contained kernel changes.
2. Fuse the controller's forget/gain projections and elementwise gating. There are
   still 2,113 dispatches per token, including many small matrix-vector products.
3. Group prompt tokens by expert and use matrix multiplication for expert
   projections. Expert vector products and dense matrix multiplication dominate
   the sampled prefill intervals, at about 7.1 and 5.9 ms per prompt token.

Float32 KV caches retain the current numerical behavior and consume substantial
memory at longer contexts; lower-precision caches need separate accuracy and
capacity measurements. Startup loading also warrants its own I/O and copy
profiling. Batching across requests and speculative decoding are not implemented.

Validation includes Metal kernel comparisons against scalar results, tiny-model
single-token/batched parity across cache wraparound and resets, full-checkpoint
token comparisons above, and terminal chat smoke tests. These checks do not replace
a broader model accuracy evaluation.
