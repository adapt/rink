# Prefill performance investigation

This report records the experiments at `f5d1fd2`. The selected kernels and
further improvements are now described in [Parallel prefill kernels](prefill-parallel.md).
The patch and historical reproduction steps below apply to that earlier revision.

The target is higher prefill throughput with the existing tensor-memory budget,
FP32 activations and accumulation, native quantized weights, FP16 KV default,
1,024-token batch limit and unchanged model schedule. This investigation uses
the 48 GiB M4 Max and baseline `41050f5`. The checkpoint revision is
`6e7c8248f75265bdd4fbab5273304243b5d8c2cb`.

The experimental kernels live in the model's examples. The production model
dispatch remains unchanged. The combined experiment is available as
[a patch](prefill-next-combined.patch) for an isolated checkout.

## Where time goes

A new diagnostic profile uses an 8,192-token prompt at 262,144-token capacity.
Kernel timestamp totals are normalized for the following shares. Instrumentation
splits compute passes, so these shares are approximate and the profile is not a
throughput benchmark.

| Work | Share of kernel time |
| --- | ---: |
| Expert matrix multiplication | 36.1% |
| Dense matrix multiplication | 19.5% |
| Local attention partitions | 16.6% |
| Full attention partitions | 13.4% |
| Convolution | 6.1% |
| Other operations | 8.4% |

CPU dispatch encoding is 0.071 seconds of a 19.21-second profiled prefill.
More CPU allocation or dispatch cleanup cannot yield a large overall speedup
for this workload. Even doubling the speed of all matrix multiplications would
give only about 1.38x overall if everything else stayed constant.

[Profile and counters](m4-max-prefill-next-profile.json)

## Measured kernel candidates

The matrix experiment varies the token, output-channel and input-column tile
sizes. The best general large tile is 64 by 64 by 32. It reuses the input
staging area for the final accumulator store. Smaller output matrices retain
smaller tiles. Increasing input-column tiles to 64 or 128 is generally slower
in this experiment.

| Native matrix at batch 1,024 | Best isolated speedup |
| --- | ---: |
| Dense BF16, 1,024 outputs by 2,304 inputs | 1.25x |
| Dense BF16, 3,072 outputs by 2,304 inputs | 1.38x |
| Dense BF16, 2,304 outputs by 3,072 inputs | 1.37x |
| NVFP4 expert up/down | 1.20x / 1.21x |
| MXFP8 expert up/down | 1.35x / 1.33x |

These kernels preserve every FP32 8-by-8 accumulator's operation order.
All matrix outputs and guard bytes match production. Expert routes are balanced
synthetic data, and repeated weights may stay cached. Real routing can waste
more work on partial tiles, so these ratios must not be treated as model speedups.

The attention experiment reuses each KV load across four queries and their
three shared query heads. It retains 128-key partitions, ordered dot products,
softmax reductions and value sums. It gives 1.32x at 64K history and 1.38x near
256K with FP16 KV. At 8K, the best isolated variant instead uses sixteen queries
for one head and gives 1.29x. The combined experiment uses the four-query,
three-head variant throughout its eligible full-attention layers.

All tested attention partials and guard bytes match production, for both FP16
and FP32 KV. Cases include empty history, causal boundaries, partial partitions,
query offsets, query tails and histories through 262,144. The existing attention
finish kernel is unchanged.

The probes allocate no device tensor buffers during dispatch. They vary bounded
on-chip threadgroup storage, up to 32 KiB; the input, output, KV and attention
workspace sizes stay the same. [Kernel measurements](m4-max-prefill-next-kernels.json)

## Combined full-model result

Each build runs three trials after loading the same checkpoint and four warmup
tokens. Each trial starts with empty history, prefills 8,192 text tokens, then
decodes 32 tokens. Capacity is 262,144; the batch limit is 1,024. Kernel profiling
is off. GPU jobs run sequentially. A second set reverses build order.

| Run order | Build | Prefill trials, tokens/s | Median |
| --- | --- | --- | ---: |
| Prototype first | Baseline | 421.2, 354.9, 393.9 | 393.9 |
| Prototype first | Prototype | 473.2, 469.7, 452.3 | 469.7 |
| Baseline first | Baseline | 422.7, 414.6, 412.6 | 414.6 |
| Baseline first | Prototype | 472.1, 469.8, 463.5 | 469.8 |

Across all six trials per build, median throughput rises from 413.6 to 469.7
tokens/s, a **13.6% increase**. The individual run orders give 19.2% and 13.3%.
The slower first baseline set shows system variability; these measurements
support an incremental improvement, not a twofold speedup.

Every trial peaks at exactly 29,396,271,104 bytes (**27.38 GiB**) of Metal
allocation. This measures device allocations, not CPU buffers, file mappings,
driver overhead or whole-process RSS. All twelve generated sequences match.
[Full-model measurements and validation](m4-max-prefill-next-combined.json)

## Validation scope

The combined prototype matches all 24,607,134 full-vocabulary logits in the
saved FP16-KV reference. This covers an 8K prompt, code, conversation, chunk
boundaries and 32 continuation steps per case, at 16,384-token capacity.
The isolated attention checks also cover FP32 KV and histories through 256K.
Each isolated candidate gets independently poisoned output and guard storage,
so skipped writes cannot inherit a correct result from the reference dispatch.
The combined prototype has not had a full-model FP32-KV or image regression,
or a full-model throughput run near 256K. Those checks remain necessary before
selecting production dispatch rules. Image encoding kernels are unchanged.

Both benchmark examples build in release mode. Formatting and Clippy with
warnings denied pass. Production files are unchanged; the earlier production
regression results are in [the allocation report](allocation-reuse.md).

## Larger steps

The measured tile and reuse changes are an incremental step. A doubling needs
larger gains across both matrix multiplication and attention. The current
profile does not support a claim that one small kernel change can achieve it.

1. Pipeline matrix-tile loading, dequantization and computation. Keep weights
   quantized in device memory and decode only the current tiles. Avoid a full
   dequantized weight copy. Retain the existing input/output buffers and FP32
   operation order, and measure with real expert-route occupancy.
2. Share KV reads across neighboring local-attention queries. The current local
   kernel shares three heads but handles one query at a time. Preserve each
   query's sliding-window and partition boundaries; query-relative positions
   shift at ring wraps. This targets another 16.6% of the short-prompt profile.
3. Parallelize prefill convolution across tokens and channels. Its current
   channel threads walk the whole batch serially. Read prior taps from existing
   history and fresh taps from the current input, then commit the final history
   in a separate dispatch. This needs no extra tensor buffer and targets 6.1%
   of the profile.
4. Investigate matrix-based tiled attention if those changes are insufficient.
   Keep scores and softmax state in bounded tiles, including relative-position
   bias, attention sinks, GQA, causal/local masks and stable long-context sums.
   Matrix reductions can change rounding even in FP32; this needs independent
   numerical validation and full-model checks before production use. It is a
   larger change than the bit-exact experiments above.

Full attention takes a larger share at long histories, so an 8K prompt speedup
does not predict throughput near 256K. The 256K numbers above measure one
attention kernel, not a full model at that history.

## Reproduce

Use a separate checkout of `f5d1fd2` for these historical commands. Run GPU jobs
one at a time:

```sh
cargo run --release -- bench --context 262144 --prompt-tokens 8192 \
  --decode-tokens 1 --repetitions 1 --prefill-batch 1024 --profile-kernels
cargo run --release -p rink-inkling --example bench_prefill_tiling
cargo run --release -p rink-inkling --example bench_prefill_attention
```

Apply the combined patch in a separate checkout containing these examples.
Before applying the patch, build and save the baseline CLI, then record the
full-vocabulary reference with the baseline example:

```sh
cargo run --release -p rink-inkling --example compare_revisions -- \
  record checkpoints/inkling-2-mini /tmp/rink-prefill-reference.json f16 8192
git apply benchmarks/prefill-next-combined.patch
cargo build --release -p rink-cli
cargo run --release -p rink-inkling --example compare_revisions -- \
  compare checkpoints/inkling-2-mini /tmp/rink-prefill-reference.json 1024
```

Run both CLIs without kernel profiling, with 8,192 prompt tokens, 32 decode
tokens, three repetitions, 262,144 capacity and a 1,024-token batch. Reverse
build order for a second set of trials. Keep GPU jobs sequential.
