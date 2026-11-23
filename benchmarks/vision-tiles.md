# Faster image encoding and native expert prefill

These changes keep the checkpoint weights, image resolution, model passes,
FP32 arithmetic and existing device tensor buffers. The baseline is `c729831`.
Measurements use the 48 GiB Apple M4 Max and checkpoint revision
`6e7c8248f75265bdd4fbab5273304243b5d8c2cb`.

## Full-model results

All image runs use the same 5,344 by 3,044 image, 10,318 image tokens and a
10,342-token prompt. Each process loads the model, warms an 8K text request,
resets, then processes the image at 256K capacity with FP16 KV and batch 1,024.
Model loading and image file decoding are excluded. GPU jobs run sequentially.

| Order | Build | Encoding | Prompt prefill | Total preparation |
| --- | --- | ---: | ---: | ---: |
| Baseline first | Baseline | 54.02 s | 21.17 s | 75.91 s |
| Baseline first | Vision update | 36.61 s | 21.22 s | 58.78 s |
| Candidate first | Vision update | 36.61 s | 23.36 s | 60.94 s |
| Candidate first | Baseline | 41.42 s | 21.70 s | 64.30 s |

Encoding improves in both orders: 32.2% and 11.6% less time. Both updated runs
encode in 36.6 seconds. Baseline variation is substantial, so the complete trial
results are included. All four runs peak at exactly **30,915,985,408 bytes
(28.79 GiB)** and match every image continuation logit. These runs isolate the
encoder changes; the native expert change below is measured separately.
[Image records](m4-max-vision-tiles-image.json)

A separate validation run of the combined build encodes in 36.31 seconds,
prefills in 21.79 seconds and prepares the image in 58.91 seconds total. It
retains the same 28.79 GiB peak and matches all 6,401,856 continuation logits.
This additional run is not part of the paired encoder comparison above.

The text comparison runs three 512-token trials followed by three 8K trials per
build, then repeats with reversed build order. Each trial starts with empty
history and decodes 32 tokens. Capacity, KV precision and batch limit are the
same as the image comparison. Per-kernel timestamp profiling is disabled.

| Prompt | Baseline median | Updated median | Throughput gain | Identical peak Metal allocation |
| --- | ---: | ---: | ---: | ---: |
| 512 tokens | 642.1 tok/s | 661.1 tok/s | 3.0% | 26.99 GiB |
| 8,192 tokens | 550.8 tok/s | 569.5 tok/s | 3.4% | 27.46 GiB |

There are six trials per build per prompt length. The separate 8K run orders
give 5.4% and 1.3% improvements; the corresponding 512-token improvements are
2.9% and 2.7%. All 24 generated sequences match. The earlier 512-token trials
retain workspace, so this suite's 8K peak differs from an 8K-only workload.
These are short-prompt measurements at 256K reserved capacity, not throughput
with 256K of active history. Decode selects its existing kernels.
[Text records](m4-max-vision-tiles-prefill.json)

## Encoder changes

Native 64-channel attention heads process 64 queries per workgroup at sequence
lengths of 2,048 encoder positions or more. Complete query tiles load directly into matrix
fragments. The last partial tile pads its queries before loading the fragments.
This avoids keeping all query fragments live in registers for complete tiles.

After every matrix reader finishes, a barrier allows K/V and score staging to
be reused for the output tile. Shared storage falls from 28.375 KiB in the
previous 32-query kernel to 16.75 KiB. No device buffer is added. The FP32 dot
products, 32-key online-softmax updates and ordered value accumulations stay the
same. Smaller images retain the existing 16-query path.

Large encoder projections reuse the decoder's existing 32- or 64-token matrix
tiles. Small projections and images retain the existing smaller tiles. Bias,
GELU, normalization, positional encoding and image preprocessing are unchanged.

## Native expert specialization

Four entry points specialize the checkpoint's NVFP4 and MXFP8 up/down
projections. The compiler receives fixed dimensions and format, while batch
size and actual expert routes remain dynamic. Runtime guards require the exact
native shape and eight routes per token; other shapes use the generic kernels.

The selected kernels call the existing FP32 matrix helpers with the same
adaptive 16/32/64-slot tiles, operation order, quantized weights and lookup
tables. They add no tensor buffers and retain the 16 KiB shared-memory limit.
The single-token decode paths are unchanged.

## Kernel measurements

The final isolated benchmarks exclude one warmup and alternate timing order
over four rounds. Large image attention is 1.09–1.45× faster across the measured
production sizes of 2,048 to 41,273 positions. At 41,273 positions, large dense
projections are 1.45–1.51× faster. Native expert gains depend on batch size and
routing: approximately 0–14% across the measured cases. All outputs match bit
for bit, with intact guards and no device allocations during dispatch.
[Kernel records](m4-max-vision-tiles-kernels.json)

Small-image attention keeps the existing kernel: the new tile is slower at
129 positions. Other experiments, including cached query fragments, different
dense prefetch schedules and expert activation fusion, did not show sufficient
benefit and are not included. Synthetic kernel timings do not predict full-model
speedups; the full-model comparisons above determine the reported gains.

## Validation and memory

All **59 workspace tests pass**, including normally ignored Metal tests run
serially. The FP16- and FP32-KV native regressions each match 24,607,134 logits
bit for bit against the established baseline, at 16K context. Together with the
combined build's image continuation, **55,616,124 logits match bit for bit**.
The five-image
independent CPU reference checks 1,495,296 features with zero tolerance violations;
maximum absolute error is 1.08e-5. Existing numerical bounds are unchanged.
[Validation records](m4-max-vision-tiles-validation.json)

The tests independently poison outputs and guards. They cover complete and
partial 64-query tiles, the runtime selection boundary, non-power-of-two head
counts, and an independent FP64 attention calculation. The native CPU vision
reference now includes an 840 by 1,000 image: its 2,101 encoder positions and 525
merged tokens exercise both large attention and projection paths, with tails.
Native expert tests compare both quantized formats and both projection directions
across sparse routes, empty experts and partial 16/32/64-slot tiles, with poisoned
output guards. Generic shapes retain the existing tests and runtime fallback.

The full allocation schedule through 256K and 1M context retains the baseline
peaks of 27.37 and 53.66 GiB. Both schedules retain 1,024-token prefill batches.
Warmed decode, repeated 8K prefill and repeated images still make no Rust heap
allocations or device allocations in the allocation audit. This simulation does
not run GPU math, measure process RSS or establish 1M throughput. The 1M
reservation exceeds this Mac's physical memory.
[Allocation records](m4-max-vision-tiles-memory.json)

## Reproduce

Run GPU jobs sequentially. Keep private image/logit snapshots outside the repository.

```sh
cargo test --release --workspace -- --include-ignored --test-threads=1
cargo run --release -p rink-inkling --example bench_vision_attention
cargo run --release -p rink-inkling --example bench_vision_matrices -- 41273
cargo run --release -p rink-inkling --example bench_expert_specialized
cargo run --release -- bench --context 262144 --prompt-tokens 512,8192 \
  --decode-tokens 32 --repetitions 3 --prefill-batch 1024
python scripts/vision_reference.py native checkpoints/inkling-2-mini /tmp/vision-reference
cargo run --release -p rink-inkling --example compare_vision -- \
  checkpoints/inkling-2-mini /tmp/vision-reference
cargo run --release -p rink-inkling --example bench_image -- \
  compare checkpoints/inkling-2-mini /path/to/image.png /tmp/image-logits.bin 1024 warm
cargo run --release -p rink-inkling --example prefill_memory -- \
  checkpoints/inkling-2-mini 262144
```

The image comparison needs a snapshot recorded with the baseline build first.
The `warm` option prefills 8K text tokens before resetting the conversation and
processing the image. Kernel benchmarks alternate timing order, exclude a warmup
and check every output bit. Their synthetic data and repeated buffers do not
predict full-model speedups.
