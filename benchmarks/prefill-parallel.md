# Parallel prefill kernels

The subsequent [image encoding and native expert update](vision-tiles.md)
includes paired measurements against this version.

This change speeds up text-model prefill without increasing the batch limit or
adding device tensor buffers. Native NVFP4/MXFP8 weights, FP32 activations and
accumulation, FP16 KV defaults and all model passes stay the same. Image prompt
prefill uses these kernels too; the image encoder is unchanged.

The baseline is `f5d1fd2`, whose production engine is identical to `41050f5`.
Measurements use the 48 GiB Apple M4 Max and checkpoint revision
`6e7c8248f75265bdd4fbab5273304243b5d8c2cb`.

## Full-model text performance

Each build loads the same checkpoint, warms four tokens, then runs three fresh
8,192-token prompts and 32 decode steps per prompt. Loading and warmup are
excluded. Capacity is 262,144, the batch limit is 1,024, KV is FP16, and kernel
profiling is off. GPU jobs run sequentially. A second set reverses build order.

| Run order | Build | Prefill trials, tokens/s | Median |
| --- | --- | --- | ---: |
| Candidate first | Baseline | 424.9, 421.5, 417.5 | 421.5 |
| Candidate first | Candidate | 564.2, 527.5, 524.5 | 527.5 |
| Baseline first | Baseline | 424.3, 402.4, 399.2 | 402.4 |
| Baseline first | Candidate | 536.4, 538.4, 534.1 | 536.4 |

Across all six trials per build, median prefill throughput rises from **419.5
to 535.2 tokens/s, a 27.6% increase**. Every trial peaks at exactly
29,396,271,104 bytes (**27.38 GiB**) of Metal allocation. All twelve generated
sequences match. The warmed prefill trials make no new device allocations.

These results cover an 8K prompt at 256K reserved capacity. They do not measure
full-model throughput at 256K of active history. The variation between trials
is included rather than selecting the fastest run.
[Text measurements and exact-logit validation](m4-max-prefill-parallel.json)

## Image prompt prefill

The same 5,344 by 3,044 image produces 10,318 image tokens and a 10,342-token
prompt. Each build runs once after a fresh load, 8K text warmup and conversation
reset. Capacity is 262,144 and the batch limit is 1,024.

| Build | Image encoding | Prompt prefill | Prefill tokens/s | Peak Metal allocation |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 43.03 s | 28.22 s | 366.4 | 28.79 GiB |
| Updated | 51.76 s | 20.50 s | 504.5 | 28.79 GiB |

Image prompt prefill throughput rises **37.7%**, reducing prefill time by
**27.4%**. Both runs peak at exactly 30,915,985,408 bytes. The complete prompt,
generated tokens and all 6,401,856 continuation logits match the private baseline
snapshot. Combined with the text regressions, the updated engine matches
**55,616,124 logits bit for bit**.

This is one run per build. The encoder code is unchanged, but its timing varies;
the slower encoder run offsets the prefill saving in total preparation time.
These measurements establish a prefill improvement, not an end-to-end image
preparation improvement. [Image measurements](m4-max-prefill-parallel-image.json)

## Kernel changes

- Convolution runs one thread per token and channel. Each thread reads prior
  taps from the existing history and fresh taps from the current input. A second
  dispatch commits the last taps after all history reads finish. Static and
  dynamic weights retain their original ordered arithmetic. Batches below eight
  keep the original convolution path.
- Large dense projections use 32- or 64-token tiles with 64 output channels.
  Small projections and batches retain their existing tiles.
- Large expert projections use 64-slot, 64-output tiles. The kernel prefetches
  the next 16 input columns into registers while computing the current tile.
  Final tiles with at most 16 or 32 occupied slots use smaller compute tiles.
  This avoids multiplying a full 64 rows for sparse tails. Every FP32 8-by-8
  accumulator retains the original operation order and 32-column zero padding.
- Full attention shares KV loads across four queries and three query heads.
  Local attention also shares four neighboring queries once the window is full
  and the batch has at least 128 tokens. Its workgroup reads the union of their
  key ranges, then places scores back into each query's original partition.
  Window boundaries, causal masks, relative-position bias, sinks, softmax trees
  and ordered value sums are unchanged.

The matrix kernels reuse input staging storage for the output store. They use
at most 16 KiB of threadgroup memory, with no dequantized weight copy. Attention
keeps the existing partition workspace and finish kernels. It does not allocate
a full prompt-by-history score matrix. All changes are confined to prefill;
single-token decode selects the existing kernels.

At batch 1,024, isolated measurements of the selected kernels show:

| Operation | Speedup over the baseline kernel |
| --- | ---: |
| Large dense BF16 projections | 1.30–1.38x |
| NVFP4 expert up / down | 1.37x / 1.31x |
| MXFP8 expert up / down | 1.41x / 1.41x |
| Local attention after the window fills | About 1.17x |
| Native static/dynamic convolution, including history commit | 7.2–17.9x |

All 104 matrix benchmark configurations match the baseline output and guards.
The attention and convolution benchmarks also match exactly. These use
synthetic values, balanced expert routes and repeated weights that may remain
cached. Their speedups are not full-model throughput predictions. Small
projections and short local-attention prefixes retain their earlier paths.
[Selected kernel measurements](m4-max-prefill-parallel-kernels.json)

## Validation

All 57 workspace tests pass, including normally ignored Metal tests, with GPU
tests run sequentially. The new checks compare every output bit and include:

- Convolution output and final history, static/dynamic weights, channel tails,
  short and long chunks, and reset with poisoned history.
- Dense projections in FP32, BF16 and FP16, odd dimensions, partial tiles,
  nonzero input offsets, cancellation and wide exponents.
- NVFP4, MXFP8 and FP32 experts, sparse routing, empty experts, shared/per-slot
  input, partial 16/32/64-slot tiles and output/tile-list guards.
- Full and local attention partials and final outputs, FP16/FP32 KV, query
  offsets, tails, causal boundaries, fresh/cached data and sliding-ring wraps.
  These include full-attention history near 256K and local positions near 1M.

Candidate outputs are independently poisoned before comparisons so skipped
stores cannot inherit a correct reference result.

The native FP16-KV and FP32-KV regressions each compare 24,607,134 logits with
the established baseline. All 49,214,268 logits match bit for bit. The cases
cover an 8K prompt, code, conversation, prefix boundaries and 32 continuation
steps per case, at 16,384-token capacity.

## Memory

The allocation simulator runs the real model schedule through the full context.
Its peaks exactly match the baseline: 27.37 GiB at 256K and 53.66 GiB at 1M.
Prefill batches remain at 1,024 tokens throughout, with a stable prefill peak.
The final logit workspace adds the same small reservation as before.

The CPU allocation audit still reports zero Rust heap allocations for warmed
decode and repeated 8K prefill. Eight repeated image requests also make no new
device allocations after warmup. These audits exclude driver overhead, page
rounding and process RSS. The 1M audit executes no GPU math and exceeds this
Mac's memory; it does not establish 1M throughput or quality.
[Allocation records](m4-max-prefill-parallel-memory.json)

## Reproduce

Run GPU jobs one at a time:

```sh
cargo test --release --workspace -- --include-ignored --test-threads=1
cargo run --release -- bench --context 262144 --prompt-tokens 8192 \
  --decode-tokens 32 --repetitions 3 --prefill-batch 1024
cargo run --release -p rink-inkling --example bench_prefill_tiling
cargo run --release -p rink-inkling --example bench_attention_heads
cargo run --release -p rink-inkling --example bench_prefill_convolution
cargo run --release -p rink-inkling --example prefill_memory -- \
  checkpoints/inkling-2-mini 262144
```

Record baseline full-vocabulary references before switching builds, using
`compare_revisions record MODEL SNAPSHOT f16 8192` and the corresponding `f32`
command. Check the candidate with `compare_revisions compare MODEL SNAPSHOT 1024`.
The `bench_image` example compares the same image prompt and continuation logits
against a private baseline snapshot; do not commit snapshots or private content.
