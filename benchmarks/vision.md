# Image input validation

Checked on November 22, 2026, on an Apple M4 Max with 48 GiB unified memory.
The checkpoint is `thinkingmachines/Inkling-2-Mini` at revision
`6e7c8248f75265bdd4fbab5273304243b5d8c2cb`.

The image encoder uses the checkpoint's BF16 matrix weights with FP32 activations
and accumulation. Its CPU reference follows [Transformers PR #51050 at
6173a2f](https://github.com/huggingface/transformers/tree/6173a2f679ff4ed393afc499aefc569da7e68bc1/src/transformers/models/inkling2).
The reference uses PyTorch 2.8.0 on CPU, with the original BF16 weights converted
to FP32. This comparison does not claim bitwise parity with BF16 activation math.

Preprocessing keeps the image size, composites alpha onto the reference
checkerboard, pads the right and bottom edges, and packs 20-pixel patches into
2×2 groups. Each 40-pixel tile occupies one decoder context position. Image
features replace placeholders after text embedding normalization.

## Automated checks

The 46 workspace tests include these image checks:

- Preprocessed pixels match the independent reference exactly, including alpha,
  padding and patch order. JPEG EXIF orientation is checked against Pillow.
- The complete encoder matches an independent two-layer CPU reference.
- Noncausal attention matches FP64 results across partial tiles, multiple heads,
  and head widths of 8, 16, 64 and 128.
- Error-function GELU is checked at 100,001 points against the system's FP64 erf.
- Mixed image/text prompts match the scalar decoder with FP16 and FP32 KV caches.
  Cases cross prefill chunks, append another image, continue text, reject invalid
  placeholders, and reset the conversation.

Run all tests on a Metal device, without other GPU workloads:

```sh
cargo test --release --workspace -- --include-ignored --test-threads=1
```

The full 23-layer encoder is also compared on four synthetic images: 83×47 RGB,
41×79 RGBA, 641×41 RGBA, and 400×320 RGB. The wide image tests interpolation beyond
the learned position grid. Every preprocessed float must match exactly. Each
output feature must satisfy `abs(actual - expected) <= 2e-5 + 2e-5 * abs(expected)`.
The test compares 285,696 output features in total.

To reproduce this comparison, install PyTorch, NumPy, Pillow and SafeTensors in a
Python environment, then run:

```sh
python scripts/vision_reference.py native checkpoints/inkling-2-mini /tmp/rink-vision-reference
cargo run --release -p rink-inkling --example compare_vision -- \
  checkpoints/inkling-2-mini /tmp/rink-vision-reference
```

The checked-in small fixtures can be rebuilt with
`python scripts/vision_reference.py fixture`. Python is not needed to run the Rust tests.

## Native checkpoint results

All four images passed. The maximum absolute feature error was `7.35e-6`; the
maximum relative RMS error was `6.52e-6`. [Full results](m4-max-vision.json) include
encoder times. These are single runs that exclude model loading and shader
compilation, not a throughput benchmark.

Text-only comparisons against the existing baseline checked 24,607,134 logits
for each KV precision. Every bit matched in both [FP16](m4-max-vision-logits-f16.json)
and [FP32](m4-max-vision-logits-f32.json). Cases include an 8,192-token prompt,
chunk and cache boundaries, code, conversation, and 32-token continuations.

A native chat smoke test used the default 262,144-token context with a 512-token
reply budget and effort 0.1. It read `RINK 5731` from a JPEG and identified a red
square and blue circle. A text-only follow-up recalled `5731`. A second PNG image,
reset, attachment clearing and a missing-file error also worked. No image was
sent to an external service.

## Image processing times

Chat and `generate` now report image encoder loading, image encoding, prompt
prefill, and generation separately. Encoding updates follow completed vision
layers. Prefill updates follow completed 32-token chunks; the final count includes
the output projection and sampling. These use the existing GPU synchronization
points, with no kernel profiling or extra GPU waits. Status goes to stderr and
updates at most once per second, except when a stage starts or finishes.

On November 22, 2026, one local 5,344 × 3,044 PNG was measured on the same M4 Max.
It produced 10,318 image tokens and 10,342 prompt tokens with the question
`Describe this image briefly.` The image is not included in the repository.
The command was `target/release/rink chat --max-tokens 64`, using the default
262,144-token context, FP16 KV caches, and effort 0.9.

| Stage | Wall time | Throughput |
| --- | ---: | ---: |
| Image encoder load | 1.28 s | — |
| Image encoding | 172.61 s | — |
| Prompt prefill, 10,342 tokens | 88.49 s | 116.9 tokens/s |
| Generation, 64 tokens | 1.89 s | 33.9 tokens/s |

This is a single run with first-use allocations included. Model startup and PNG
file decoding are excluded from the table. Image encoding includes preprocessing,
workspace allocation, all vision layers, and the projector. The encoder stays
loaded for later images. Generation stopped at the 64-token reply limit while
the model was still thinking, so this does not measure time to a complete answer.

The stages before generation took 262.38 seconds in total. Image encoding accounts
for about 66% of that time, making it the largest measured cost for this image.
This change adds measurement and display; batch sizes, kernels, and precision
are unchanged. Progress tests compare token results, continuation logits, GPU
dispatches, encoders, command buffers, and readback bytes with reporting enabled
and disabled, for both KV-cache precisions.

## Faster image encoding and prefill

Implementation `92b3c8e` reduces preparation time for the same 5,344 × 3,044 image
from 272.09 to 132.12 seconds. These are fresh before/after runs on the M4 Max,
with the same prompt, 262,144-token context, FP16 KV caches, effort 0.9, and
32-token prefill batches. The baseline is `931c014` with the benchmark harness
and allocation counter added. The image size and model precision are unchanged.

| Measurement | Before | After |
| --- | ---: | ---: |
| Image encoding | 182.42 s | 58.01 s |
| Prompt prefill, 10,342 tokens | 88.42 s | 72.88 s |
| Prefill throughput | 117.0 tokens/s | 141.9 tokens/s |
| Preparation, including image encoder load | 272.09 s | 132.12 s |
| Peak Metal allocation | 28.75 GiB | 28.19 GiB |

Encoding is 3.14× faster, prefill throughput is 21.3% higher, and total preparation
is 2.06× faster in this pair. Peak Metal allocation falls by 574.38 MiB. These
are single runs, with first-use allocations included and per-kernel timestamp
profiling disabled. Model startup and PNG decoding are excluded. Peak allocation
samples Metal's `current_allocated_size` after buffer allocations and at counter
reset; it includes resources retained by pending commands. Process RSS was not
measured. [Full measurements](m4-max-image-prefill.json) omit image-derived output
tokens; the image and logit snapshot stay outside the repository.

The changes are:

- Native image attention uses eight SIMD groups to process two independent
  eight-query tiles. They share each 32-key tile and its values. Output
  accumulators stay in registers. The kernel retains the original FP32 matrix
  operations, online softmax, and reduction order. Other head widths use the
  existing kernel.
- Image encoding reuses two buffers with nonoverlapping lifetimes. It also
  releases the pixel upload after its projection and layer workspaces before
  the final projector. All GPU dependencies use the existing ordered stream.
- Prefill attention uses the existing ordered cache-prefetch kernel.
- Grouped NVFP4 projections use the exact lookup tables already resident for
  decode. This adds no tables or activation buffers. Matrix accumulation order
  and the MXFP8 path remain unchanged.

Isolated [kernel measurements](m4-max-image-prefill-kernels.json) used the full
41,273-patch sequence, including CLS, and compared every output byte. The retained
vision kernel took 2.40 seconds versus 7.91 seconds for the original. NVFP4 expert
projections improved by about 10% for gate/up and 28% for down projection. These
kernel measurements guide selection; the table above reports full model runs.

Correctness checks passed:

- All 46 workspace tests, including Metal tests, ran serially and passed.
- The large image produced the same first token, 32 continuation tokens, and
  all 6,401,856 continuation logits, bit for bit, as the baseline.
- Vision attention matched the original for all 42,263,552 output floats at
  the image's sequence length. Small tail cases also retain the FP64 oracle check.
- Grouped NVFP4 tests compare exact output bytes across partial rows, short
  input widths, sparse routes, partial token tiles, and shared/per-route inputs.
  The existing exhaustive table test covers FP4 codes, FP8 scales, and signed zero.
- The full 23-layer encoder passed the four independent CPU-reference images:
  285,696 features, maximum absolute error `7.35e-6`, and no tolerance violations.
  [Reference results](m4-max-image-prefill-vision.json) retain the existing bounds.
- Formatting, Clippy with warnings denied, and diff checks passed.

To repeat the full image comparison, build the benchmark with the baseline model
code first and save its executable. Run both builds on the same image and snapshot:

```sh
cargo build --release -p rink-inkling --example bench_image
cp target/release/examples/bench_image /tmp/rink-image-before
/tmp/rink-image-before record checkpoints/inkling-2-mini /path/to/image.png /tmp/image-logits.bin
# Switch to the candidate model code, then rebuild.
cargo run --release -p rink-inkling --example bench_image -- \
  compare checkpoints/inkling-2-mini /path/to/image.png /tmp/image-logits.bin
```

The baseline needs the benchmark harness and allocation counter. Keep snapshots
outside the repository: they contain image-derived data. Run GPU checks one at a
time. The isolated comparisons require no checkpoint:

```sh
cargo run --release -p rink-inkling --example bench_prefill_kernels -- 41273
cargo run --release -p rink-inkling --example bench_projections -- --experts-only
```

## Larger batches with bounded workspace

Implementation `d46ec68` reduces preparation for the same image from 132.12 to 101.76
seconds, a 23.0% reduction. The baseline is `e1a89f8` (the implementation measured
above). Both runs use the same 262,144-token context, FP16 KV cache, effort 0.9,
image and prompt. The same first token, 32 continuation tokens, and all 6,401,856
continuation logits match bit for bit.

| Measurement | Previous version | This version |
| --- | ---: | ---: |
| Image encoding | 58.01 s | 48.24 s |
| Prompt prefill, 10,342 tokens | 72.88 s | 52.23 s |
| Prefill throughput | 141.9 tokens/s | 198.0 tokens/s |
| Preparation, including image encoder load | 132.12 s | 101.76 s |
| Peak Metal allocation | 28.19 GiB | 28.19 GiB |

These are single runs with the same timing method and exclusions as above.
Memory held after the benchmark falls by 422.86 MiB. A 64-token candidate took
60.73 seconds to prefill and also matched every continuation logit. The default
is 128 because it was faster and stayed within the previous memory peak.
[Measurements](m4-max-prefill-batches.json) include both candidates.

Prefill now permits up to 128 tokens per batch. A separate scratch pool reuses
temporary activations between layers. Layer outputs, residual banks, prelude
state, and previous-pass messages remain in the outer pool until they are no
longer needed. The final layer operation writes directly into its outer output;
reuse adds no copy kernels or CPU waits.

Attention retains the previous 32-token, full-capacity workspace budget. For the
native model at a 262,144-token capacity, that budget is 780 MiB. The batch size
falls as history grows:

| History covered by the batch | Maximum batch |
| --- | ---: |
| Through 65,536 tokens | 128 tokens |
| Through 131,072 tokens | 64 tokens |
| Through 262,144 tokens | 32 tokens |

The scheduler checks the actual attention shapes, available input, and workspace
size for each chunk. Short tails keep the original 32-token boundary so they use
the same expert kernels and reductions. Smaller configured contexts use the same
budget rule. KV precision, context capacity, image resolution, and effort are
unchanged.

The vision encoder uses a 32-query attention tile for native 64-channel heads at
sequence lengths of 32,768 or more. Four independent eight-query tiles share each
32-key load. FP32 operations and softmax reduction order are unchanged. Shared
threadgroup storage is 28.4 KiB; device buffer sizes do not change. Shorter
sequences retain the 16-query kernel: the larger tile was slower at 321, 1,025,
4,097, and 16,385 positions, and approximately tied at 8,193.
[Kernel comparisons](m4-max-vision-query-tiles.json) include the rejected load
layouts and the sequence-length sweep. All compared output bytes match.

The `prefill_memory` example runs the decoder's allocation schedule through a
complete context using buffers that track size and lifetime. It retains pending
resources until synchronization and rejects growth of the prefill peak after
the first complete batch. It does not execute GPU math or measure driver memory,
page rounding, or process RSS. Full-model correctness and actual Metal peak
allocation are checked separately with `bench_image`.

The native 256K allocation audit keeps its peak unchanged from the first full
batch to the end of prefill. The full schedule uses 512 batches of 128 tokens,
1,024 batches of 64 tokens, and 4,096 batches of 32 tokens. The final vocabulary
projection adds about 1.53 MiB of live buffers and stays below the initial peak.
This is not a full 256K model execution or a process-RSS measurement.
[Allocation records](m4-max-prefill-memory.json) include the 32-token control.

All 48 workspace tests pass, including the Metal tests. Native attention at
64K/128 tokens, 128K/64 tokens, and 256K/32 tokens passes an independent analytic
reference and an exact comparison with the original kernel. Additional tests
cover image placeholders crossing batch boundaries, conversation history,
resets, both KV formats, sparse expert routes, and partial matrix tiles.
The native text runs match all 49,214,268 logits across
[FP16](m4-max-prefill-batches-logits-f16.json) and
[FP32](m4-max-prefill-batches-logits-f32.json) KV caches, including an 8,192-token
prompt and 32 continuation steps per case. The independent full vision-encoder
reference passes all 285,696 features from four images with no tolerance
violations. [Vision reference results](m4-max-prefill-batches-vision.json) retain
the existing error bounds. Formatting and Clippy with warnings denied pass.

```sh
cargo run --release -p rink-inkling --example prefill_memory -- \
  checkpoints/inkling-2-mini 262144 128
cargo run --release -p rink-inkling --example bench_image -- \
  compare checkpoints/inkling-2-mini /path/to/image.png /tmp/image-logits.bin 128
```

Use a final argument of `32` or `64` to compare smaller batch limits. The library
also exposes `set_prefill_batch_limit`. The next change removes the reduction
of the decoder batch as history grows.

## Fixed decoder batches through 256K

Implementation `c61f973` keeps 128-token batches through the full configured
context. Only the attention queries are split into smaller groups. Projections, convolutions,
expert routing and expert matrix products keep the larger batch. Short prompt
tails retain their previous chunk boundaries and expert kernels.

On the M4 Max, median prefill throughput improves by 17.7% in a 16K-context
benchmark that crosses both former batch-size thresholds. All three candidate
runs improve throughput, and their 32 generated tokens match the baseline.
Peak Metal allocation is identical.

| Measurement | Previous scheduler | Fixed decoder batches |
| --- | ---: | ---: |
| Prefill, 16,352 tokens (median of three) | 103.65 s | 88.10 s |
| Prefill throughput (median) | 157.8 tokens/s | 185.6 tokens/s |
| Prefill dispatches | 706,567 | 263,233 |
| Prefill command buffers | 351 | 129 |
| Peak Metal allocation across trials | 20,041,547,776 bytes | 20,041,547,776 bytes |

[Measurements](m4-max-prefill-query-groups.json) include every trial. Both builds
use the same repeated text prompt, FP16 KV storage, a 16,384-token capacity and
32 fixed decode steps. Command-buffer timing is enabled; per-kernel profiling
is disabled. Prefill excludes loading and warmup, and the first trial includes
scratch allocation. This is a full-model measurement at 16K, not a throughput
claim for 256K. Peak Metal allocation does not measure process RSS.

At a 262,144-token capacity, full attention uses this schedule:

| History covered by the batch | Decoder batch | Attention query group |
| --- | ---: | ---: |
| Through 65,536 tokens | 128 | 128 |
| Through 131,072 tokens | 128 | 64 |
| Through 262,144 tokens | 128 | 32 |

Each group reuses the same 780 MiB partition workspace and writes its own rows
in the full-batch output. Query offsets do not change the original chunk base:
every group can read earlier fresh keys and values. The cache write runs only
after all groups finish, so a local ring cannot overwrite history still needed
by another group. This adds no copies, device allocations or CPU waits between
groups. Each query retains its original dot products, softmax, partition order,
FP32 accumulation and KV rounding.

The complete 256K allocation audit now uses 2,048 batches of 128 tokens, compared
with 5,632 batches in the preceding schedule. Peak tracked allocation remains
exactly 28,856,387,244 bytes. The peak stays fixed after the first full batch;
the final vocabulary projection also stays below it. Audits at capacities of
128, 4,096, 65,536 and 131,072 tokens pass the same check. These are allocation
simulations of the actual decoder schedule, not full-context GPU executions or
process-RSS measurements. [Allocation records](m4-max-prefill-query-groups-memory.json)
include the previous 256K schedule and all five new capacities.

Native GPU attention tests run all 128 queries at 64K, 128K and 256K histories.
Each result matches the original unsplit kernel bit for bit and passes an
independent analytic reference. The test frees the larger reference workspace
before it runs the candidate with 780 MiB. Other checks cover both KV formats,
uneven query groups, output guards, local-ring wrapping, relative bias, causal
masking and compensated reductions at an unaligned 65K position. Text and image
fixtures compare complete continuations across batch limits of 32, 64 and 128,
including later turns and image placeholders that cross query-group boundaries.

Full-checkpoint comparisons match all 49,214,268 saved logits bit for bit across
FP16 and FP32 KV storage. Each precision covers an 8,192-token prompt, code and
conversation prompts, intermediate checkpoints and 32 continuation steps per
case. The [measurement record](m4-max-prefill-query-groups.json) lists the checked
positions and zero-mismatch totals. These full-model checks use a 16K context;
the 256K checks above cover attention and allocation, not a full-model prompt.
All 48 workspace tests pass, including the Metal tests, with no tests skipped.
Formatting and Clippy with warnings denied pass. The release CLI is rebuilt.

To repeat the timing comparison, save the `rink` executable from `21dfc5f`, then
run the same command with that executable and the new build. Run one GPU job at
a time:

```sh
cargo run --release -p rink-cli -- bench --model checkpoints/inkling-2-mini \
  --context 16384 --prompt-tokens 16352 --decode-tokens 32 --repetitions 3
cargo run --release -p rink-inkling --example prefill_memory -- \
  checkpoints/inkling-2-mini 262144 128
cargo test --release --workspace -- --include-ignored --test-threads=1
```
