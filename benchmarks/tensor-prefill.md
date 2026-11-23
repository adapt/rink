# TensorOps prefill projections

This pass compares with `a5b4787` on the 48 GiB Apple M4 Max, macOS 27.0.1,
using checkpoint revision `6e7c8248f75265bdd4fbab5273304243b5d8c2cb`.

## Full-model text results

| Prompt | Reference median | Updated median | Gain | Identical peak Metal allocation |
| --- | ---: | ---: | ---: | ---: |
| 512 tokens | 705.9 tok/s | 761.3 tok/s | 7.8% | 26.99 GiB |
| 8,192 tokens | 553.1 tok/s | 569.1 tok/s | 2.9% | 27.46 GiB |

There are six trials per version per prompt length. All 24 generated sequences
match. Five of six short-prompt pairs and all six 8K pairs favor the update.
Timings vary, including a larger first-pair gain; all trials are retained.
The median of the six paired throughput gains is 5.2% at 512 tokens and 2.9%
at 8K. The table instead compares each version's median throughput.
These results establish a modest improvement, not the 2x target.
[Text trial records](m4-max-tensor-prefill-text.json)

## Selected changes

Large dense BF16 text projections can use Metal's TensorOps with FP32 inputs and
outputs. The matrix descriptor disables relaxed precision. Native quantized
expert weights, model passes, KV precision, image resolution and batch limits
are unchanged. No full-precision weight copy or additional device tensor is
allocated.

Selection requires at least 512 tokens, at least 1,024 output rows, and a column
count divisible by 32. The 64-by-64 output tiles use 128 threads. Smaller shapes
and other formats keep the existing kernels.

The image encoder retains its existing kernels. The TensorOps and fused-GELU
encoder candidates passed correctness checks, but their full-image timings
did not establish a reliable gain over the existing encoder.

TensorOps is compiled as an optional, separate library. If the device or system
cannot compile it, model loading retains the existing kernels. A regression
test injects this failure and checks the fallback. The isolated benchmarks fail
explicitly when the optional library is unavailable, preventing a fallback from
being reported as a TensorOps measurement.

The implementation follows Apple's [TensorOps programming guidance](https://developer.apple.com/download/files/Metal-Performance-Primitives-Programming-Guide.pdf).
It uses the existing Rust accelerator interface and buffer bindings. On M4,
TensorOps uses shader implementations; M5 neural-accelerator performance claims
do not establish a speedup on this machine.

## Kernel measurements

TensorOps dense kernels are 1.21–1.43x faster across the measured shapes at
batches of 512, 1,024 and 41,273. These measurements include vision shapes;
production selection is restricted to text projections. All output bits and
guard bytes match. The rejected fused bias/GELU candidate is 1.07–1.16x faster
than TensorOps plus a separate activation at the measured 1,024- and
41,273-token shapes. It also passes the exact comparisons. Both candidates
allocate zero device buffers during dispatch.

These benchmarks alternate timing order, exclude one warmup, and retain all
four timing rounds. Repeated synthetic inputs may remain cached, so their
speedups do not predict full-model throughput.
[Kernel records](m4-max-tensor-prefill-kernels.json)

## Image candidates

A full-model pair used a 5,344-by-3,044 image with 10,318 image tokens,
256K capacity, an 8K text warmup and 32 continuation tokens. The encoder
changed from 37.02 to 37.50 seconds; image prefill changed from 20.01 to
21.00 seconds. Peak Metal allocation was identical at 28.79 GiB, and all
6,401,856 snapshot logits matched. This pair did not demonstrate a speedup.
[Full-image candidate records](m4-max-tensor-prefill-image.json)

A separate comparison loaded only the encoder and ran base, TensorOps and
TensorOps plus fused GELU, then reversed the order. All six feature arrays
matched exactly and peak allocation was identical. The base trials ranged
from 34.99 to 39.97 seconds, TensorOps from 36.37 to 38.14 seconds, and fused
TensorOps from 35.20 to 35.62 seconds. Fusion helped relative to unfused
TensorOps, but the large timing drift and full-model result do not establish
a reliable improvement over the existing encoder. Neither encoder candidate
is enabled in the final change.
[Encoder trial records](m4-max-tensor-prefill-encoder.json)

## Measurement method

The text benchmark alternates the old and updated dense projections within
one loaded model. It runs six trials per version at 512 and 8,192 prompt tokens,
reversing order each round. Each trial starts with empty history and generates
32 continuation tokens. Prompt tokens repeat one fixed sentence. The model
reserves 256K capacity, uses FP16 KV and 1,024-token prefill batches, and runs
four decode warmup tokens first.

All other kernels and tensor buffers are shared. The old path selects the
compact or wide kernel using the rules from `a5b4787`. Per-kernel timestamp
profiling is disabled. First-use allocations and all timing trials are retained.
The 8K trials retain workspace from the earlier 512-token trials.

These are fresh-prompt measurements with 256K reserved capacity, not throughput
at 256K of active history. This pass does not change single-token decode kernels.

## Validation and memory

All 60 workspace tests pass, including normally ignored Metal tests run serially.
Projection checks cover cancellation, wide exponents, token/row tails, unaligned
input offsets and independently poisoned output guards. The fallback test
rejects the optional library while retaining the base kernels.

The native FP16- and FP32-KV regressions match all 49,214,268 text logits bit for
bit. A final full-image regression with the original encoder matches another
6,401,856 logits, for 55,616,124 in total. Peak Metal allocation remains exactly
30,915,985,408 bytes (28.79 GiB). The independent CPU vision oracle checks
1,495,296 features across five images, including a 2,101-position image.
It has zero tolerance violations and the unchanged maximum absolute error of
1.08e-5. No numerical bounds are loosened. Formatting and Clippy with warnings
denied pass. A final direct kernel check confirms that the standalone TensorOps
library compiles and preserves all output and guard bits without allocations.
[Validation records](m4-max-tensor-prefill-validation.json)

The complete allocation simulations at 256K and 1M exactly match the previous
build, including peaks of 27.37 and 53.66 GiB. They retain 1,024-token batches.
Warmed decode, repeated 8K prefill and eight repeated image requests still make
zero Rust heap allocations/reallocations and zero device-buffer allocations.
These simulations exclude GPU execution, driver storage, page rounding and
process RSS. The 1M reservation still exceeds this Mac's physical memory; the
simulation does not establish 1M throughput or quality.
[Allocation records](m4-max-tensor-prefill-memory.json)

## Experiments not selected

- FP32 matrix attention over 32 queries accelerated the longer isolated cases,
  but changed partition values by up to about 3.1e-5 in these synthetic inputs.
  The score and relative-bias summation order changes. This prototype has no
  full-model quality validation and is not enabled in production.
- Variants that retain the original score arithmetic and use matrix operations
  only for the value sum match the existing output bits. They did not show a
  consistent speed improvement across the tested histories and cache formats.
- Expert projections using cooperative TensorOps inputs and register
  dequantization were slower than the current native kernels. A second version
  with bounded shared-memory staging also failed to improve throughput.
- TensorOps and fused-GELU image projections passed exact comparisons, including
  bias/no-bias, saturation, offsets and partial tiles. Isolated gains did not
  translate into a reliable full-image gain, so the encoder changes were removed.

The faster matrix-attention prototype is a research result, not evidence that
prefill can safely double. The selected changes preserve the current exact-logit
regression contract.
[Rejected experiment records](m4-max-tensor-prefill-experiments.json)

## Reproduce

Run GPU commands sequentially:

```sh
cargo test --release --workspace -- --include-ignored --test-threads=1
cargo run --release -p rink-inkling --example compare_tensor_prefill -- 6
cargo run --release -p rink-inkling --example bench_tensor_prefill -- 1024
cargo run --release -p rink-inkling --example bench_tensor_prefill -- 41273
cargo run --release -p rink-inkling --example prefill_memory -- \
  checkpoints/inkling-2-mini 262144
cargo run --release -p rink-inkling --example allocation_audit -- \
  checkpoints/inkling-2-mini 262144
```

The full-model image comparison uses `bench_image compare` against a private
snapshot recorded before the changes. Image content and generated token IDs
are excluded from committed measurement records.
