# Prefill speed and memory budgets

Measured on November 22, 2026, on an Apple M4 Max with 40 GPU cores and 48 GiB
of unified memory. The checkpoint is the original Inkling 2 Mini release,
revision `6e7c8248f75265bdd4fbab5273304243b5d8c2cb`. These runs use FP16 KV storage,
FP32 activations and accumulation, all three model passes, and greedy decoding.
They do not use a draft model. GPU jobs run one at a time.
The implementation is commit `1966113`; the baseline is `3939976`.

## Changes

Full-attention prefill processes four queries together. Each key/value load
serves four sets of registers. Each query keeps its original 128-key partitions,
dot-product order, softmax reductions, ordered value sums and KV rounding.
Local attention and single-token decode retain their existing kernels.

For 129 through 2,048 attention partitions, a separate finish kernel computes
each partition's exponential once and shares it across value channels. It
keeps the original compensated sums and attention sink. Larger histories use
the existing finish kernel.

The default decoder batch increases from 128 to 2,048 tokens. Full decoder
batches keep this size as history grows. Attention can split their queries
into groups that share the existing partition workspace: 780 MiB at 256K.
Short tails use smaller batches. No checkpoint conversion, image resize,
precision reduction or arithmetic approximation is added.

`chat`, `generate` and `bench` accept `--prefill-batch`. Supported values are
powers of two from 32 through 4,096. The chat header and benchmark JSON report
the selected limit.

## Batch selection

Each size uses a fresh model load, 262,144-token capacity, an 8,192-token repeated
text prompt, three trials and 32 fixed decode steps per trial. Loading and
warmup are excluded. The first trial includes scratch allocation. Command-buffer
timing is enabled; per-kernel profiling is disabled.

| Maximum batch | Median prefill tokens/s | Peak Metal allocation |
| ---: | ---: | ---: |
| 1,024 | 264.1 | 28.83 GiB |
| 2,048 | 302.5 | 31.07 GiB |
| 4,096 | 279.8 | 35.54 GiB |

The 2,048-token batch has the best measured median and leaves more room for
image processing and other applications than 4,096. Individual trials vary;
this is a measured default for this Mac, not an optimal size for every device
and prompt. The memory figures are Metal allocations, not process RSS or total
system memory.

A fresh comparison with revision `3939976` gives 217.7 tokens/s for its default
128-token batch and 302.5 tokens/s for the new default, a 38.9% median gain.
All generated token IDs match in both builds. Peak Metal allocation rises from
26.88 to 31.07 GiB. [All trials](m4-max-prefill-budget.json)

At 256K history, the isolated partition kernel takes 94.16 ms instead of
245.28 ms for 32 queries, a 2.61× speedup. The finish kernel takes 2.81 ms instead
of 3.69 ms. These are individual attention-kernel timings, not full-model token
rates. [Kernel timings and exact comparisons](m4-max-prefill-budget-kernels.json)

## Image input

The original 5,344 × 3,044 image produces 10,318 image tokens and a 10,342-token
prompt. One run with the new default gives:

| Stage | Time |
| --- | ---: |
| Image encoding | 53.48 s |
| Prompt prefill | 39.22 s (263.7 tokens/s) |
| Preparation, including encoder load | 93.90 s |

Peak Metal allocation is 31.75 GiB. The complete prompt, 32 generated tokens
and all 6,401,856 continuation logits match the private baseline bit for bit.
The [record](m4-max-prefill-budget-image.json) excludes image-derived text,
token IDs and logits.

The earlier [128-token batch measurement](m4-max-prefill-batches.json) took
52.23 s for prefill, 48.24 s for encoding and 101.76 s for preparation. This is
a historical comparison, not a fresh paired image benchmark. Encoder code is
unchanged in this pass; the shorter prefill does not imply faster image encoding.

## Predictable memory

Context capacity sets the KV allocation. The batch limit bounds decoder
scratch. Attention groups reuse one workspace. The allocation audit follows
the real decoder schedule through every token, but its buffers contain sizes
and lifetimes instead of tensor data. It retains pending buffers through
synchronization. It does not execute GPU math or include page rounding,
driver overhead, process RSS, images or other applications.

With the 2,048-token default, the audited peak is:

| Context capacity | Peak tracked allocation |
| ---: | ---: |
| 4,096 | 22.44 GiB |
| 65,536 | 24.49 GiB |
| 262,144 | 31.06 GiB |
| 1,048,576 | 57.35 GiB |

Each audit reaches its peak in the first full batch and stays within that peak
through the end of the configured context. Smaller prompts can use less
scratch. Image processing adds encoder weights, temporary image buffers and
projected features; its peak depends on image dimensions and retained decoder
scratch. The table is a text-inference budget, not a promise about total RAM use.

At 256K, the measured short-run Metal peak is 31.07 GiB. This Mac reports a
recommended Metal working set of 37.44 GiB. A 1M context exceeds its physical
48 GiB even before image processing and system overhead. The 1M allocation
audit is not a full-model correctness or throughput test.
[Allocation records](m4-max-prefill-budget-memory.json) include six batch limits
at 256K and the four context capacities above.

## Correctness checks

All 49,214,268 saved full-vocabulary logits match bit for bit across FP16 and
FP32 KV storage. Each precision covers an 8,192-token prompt, code and
conversation prompts, intermediate checkpoints and 32 continuation steps per
case. These full-model reference comparisons use a 16K capacity.
[Results and snapshot hashes](m4-max-prefill-budget-logits.json)

Native attention tests compare every partial and output bit at histories up
to 256K. They cover both cache formats, fresh KV, causal masking, relative bias,
unaligned positions, partial query groups and output guards. The full native
256K attention geometry also passes an independent analytic reference.

Text and image fixtures check batch limits from 32 through 4,096, prompt tails,
later turns, image placeholders across batch boundaries and continuation logits.
An additional text check crosses 128 attention partitions, where compensated
summation starts. All 50 workspace tests pass with no tests skipped. Formatting
and Clippy with warnings denied pass.

## Reproduce

Run one GPU job at a time:

```sh
cargo run --release -- bench --context 262144 --prompt-tokens 8192 \
  --decode-tokens 32 --repetitions 3 --prefill-batch 2048 --prefill-window 4096
cargo run --release -- bench --context 262144 --prompt-tokens 262016 \
  --decode-tokens 128 --repetitions 1 --prefill-window 4096
cargo run --release -p rink-inkling --example bench_attention_queries
cargo run --release -p rink-inkling --example prefill_memory -- \
  checkpoints/inkling-2-mini 262144 2048
cargo test --release --workspace -- --include-ignored --test-threads=1
```

Timing windows follow existing chunk synchronization points. Recording a window
does not split a batch or add a vocabulary projection. The final window includes
the final projection. A window's rate describes that part of prefill; it is
separate from average prompt throughput and single-token decode throughput.

## EXO's reported rates

EXO reports 125 decode tokens/s and 1,750 prefill tokens/s on a 48 GB M5 Max,
and higher rates on an M5 Ultra. In a follow-up, EXO attributes its gains to
kernel improvements, fusion and adaptive drafting with DSpark in an upcoming
Apple silicon engine. The post does not provide a complete benchmark recipe
or the prompt/history length. [EXO's explanation](https://x.com/exolabs/status/2123953917302354430)

DSpark proposes multiple continuation tokens for the target model to verify
together. Rink currently generates one target-model token per decode step.
Drafting can help explain a decode gap; it does not by itself explain prefill
speed. [Inkling's DSpark design](https://huggingface.co/blog/inkling2mini)

The hardware also differs. M5 adds GPU matrix accelerators, and Apple has
measured substantial prefill gains with MLX on M5. That makes hardware and
matrix-kernel support plausible contributors to EXO's prefill rate, but it
does not establish EXO's exact implementation or speedup over this M4 Max.
[Apple's M5 measurements](https://machinelearning.apple.com/research/exploring-llms-mlx-m5)

A near-256K rate must be compared with another near-256K run. Full attention
reads much more history there than it does for a short prompt. The screenshot's
headline rates are not evidence of the same rates at 256K.
