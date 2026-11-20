# Metal inference benchmarks

Measured November 20, 2026 on an Apple M4 Max with 48 GiB unified memory, using the
original Inkling 2 Mini checkpoint at revision
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

There are still 2,113 kernel dispatches per decoded token. More operation fusion or
reusable command encoding may help, but should be measured against the GPU timings.
Expert projections during prefill still use routed vector products; grouping tokens
by expert for matrix multiplication is another candidate. Float32 KV caches retain
the current numerical behavior and consume substantial memory at longer contexts;
lower-precision caches need separate accuracy and capacity measurements. Startup
loading also warrants its own I/O and copy profiling. Batching across requests and
speculative decoding are not implemented.

Validation includes Metal kernel comparisons against scalar results, tiny-model
single-token/batched parity across cache wraparound and resets, full-checkpoint
token comparisons above, and terminal chat smoke tests. These checks do not replace
a broader model accuracy evaluation.
