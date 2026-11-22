# Reuse inference storage

The allocation audit found repeated CPU allocations during every token, a full
copy of normalized image pixels, and separate reservations for text and image
working buffers. This change removes that work without changing kernels,
precision, batch limits, or the model's loop schedule.

Measured on the 48 GiB Apple M4 Max, using checkpoint revision
`6e7c8248f75265bdd4fbab5273304243b5d8c2cb`. The baseline is `6edd566`.
Unless stated otherwise, the context capacity is 262,144, the batch limit is
1,024, and KV storage is FP16.

## CPU allocations

| Operation | Before | After |
| --- | ---: | ---: |
| Rust heap allocations/reallocations per warmed decode token | 9,011 | 0 |
| Rust heap allocations/reallocations during a repeated 8K prefill | 81,691 | 0 |
| Rust heap allocations/reallocations per checkpoint tensor lookup | 6,247 | 1 |
| Extra CPU patch staging for a 5,344 by 3,044 image | 188.93 MiB | 0 |

The counting allocator runs the real model control flow with size-only device
buffers. It excludes Metal/Objective-C internals, the CLI and tokenizer, and GPU
execution. The image staging saving follows from the direct Metal initialization
path; the simulator skips host initialization callbacks. These counts do not
measure whole-process RSS or establish an end-to-end speedup.

Kernel arguments use a fixed stack array. Weight names use bounded stack
formatting, configuration and schedules share immutable data, and greedy
readback uses eight stack bytes. Full-logit readback writes directly into the
returned float vector. Each checkpoint shard retains its validated metadata;
only the requested tensor's shape is copied for the SafeTensors view.

Image preprocessing writes the original normalization, transparency and padding
operations directly into shared Metal storage. It creates no full CPU float
copy. The `generate` command drops its image and tokenizer encoding after
prefill. Prompt-token uploads reuse one buffer and synchronize before host writes.

## Working storage

| Size-only allocation audit | Before | After |
| --- | ---: | ---: |
| Text prefill through 256K | 28.83 GiB | 27.37 GiB |
| Image after text prefill | 30.93 GiB | 28.79 GiB |
| Repeated image requests, after warmup | 30.93 GiB | 28.79 GiB |

The image audit uses synthetic pixels at the same dimensions as the earlier
image benchmark. After the first image request, all eight repeated image
requests make no new device allocations. The audit asserts that the final
requests have stable peaks and no new allocations.

One pool serves image encoding and text prefill. Text arenas divide blocks into
bounded, 256-byte-aligned views. Live vision tensors use separate blocks to
preserve independent Metal resources. Text layers reuse one inner arena.
Four outer output slots protect the current value, the prelude, and the
previous loop pass while another slot receives the next output. This replaces
the separate output slot for every layer visit. Idle blocks are released before
the pool grows when no available block fits.

An arena resets only after its last reads have been encoded on the ordered
stream. Phase changes and CPU access synchronize as required. Pending Metal
commands retain the underlying buffers. Views retain allocation ownership and
enforce their logical bounds for host reads and writes.

Scratch, KV/convolution state and lookup tables use overwrite-only allocation.
Convolution reads only initialized lags; attention reads only committed history
or fresh K/V values. The ordinary zeroed allocation API remains available.
Tests fill overwrite-only storage with NaNs to detect initialization assumptions.

At 4K, 64K, 256K and 1M capacities, the size-only text peaks are respectively
18.75, 20.80, 27.37 and 53.66 GiB. Each stays stable from the first full prefill
batch to the end of context. The 1M case is an allocation audit; it exceeds this
Mac's working-set limit and is not a GPU throughput or quality measurement.

These totals exclude CPU images, tokenizer state, file mappings, page rounding
and driver overhead. Larger images and batches require more working storage.
The pool retains reusable blocks, so a small later request can retain storage
from an earlier large request.

## Native Metal measurements

An 8K text prompt followed by 32 decode tokens, over three trials, has a median
prefill rate of 409.2 tokens/s and a decode rate of 35.1 tokens/s. The earlier
baseline measured 398.7 and 34.9 tokens/s. These are small timing differences,
not a claim of a large speedup. Peak Metal allocation is 27.38 GiB.

Two image-after-text runs peak at 28.79 GiB each, including vision weights and
the retained pool, compared with 30.93 GiB each for the original engine under
the same warmup. The updated engine prefills 10,342 prompt tokens at 370.5 and
381.9 tokens/s; the original runs reach 381.8 and 383.8 tokens/s.

Encoding takes 55.94 and 55.35 seconds after the change, compared with 50.47 and
55.43 seconds before it. The original run varies between about 1.7 and 2.4
seconds per vision layer. These checks do not establish a consistent image
speedup. All four runs match the saved image snapshot exactly. Allocation totals
come from actual Metal counters; they still exclude CPU and driver memory.

## Validation

FP16 and FP32 full-model regressions compare 24,607,134 logits each with the saved
references. All 49,214,268 logits match bit for bit. Cases include an 8K prompt,
code, conversation, chunk boundaries and continuation tokens.

The image-after-text run also matches its saved prompt, continuation tokens and
6,401,856 logits bit for bit. The combined total is 55,616,124 matching logits.

The tests cover poisoned storage, multiple image turns, changes in batch size,
loop state, reset, nested buffer views, alias ordering, guard regions, host I/O
bounds, cross-device ownership, and independent cached shard metadata. Existing
scalar/reference tests and long-context kernel checks also pass. All 54 workspace
tests pass, including the normally ignored Metal tests. Formatting, Clippy with
warnings denied, and release builds also pass.

Run the audits with:

```sh
cargo run --release -p rink-inkling --example allocation_audit -- checkpoints/inkling-2-mini
cargo run --release -p rink-inkling --example prefill_memory -- checkpoints/inkling-2-mini 262144
```

The image benchmark accepts `warm` to prefill an 8K text request and reset the
conversation before its image run. This retains working storage and verifies
the image-after-text case against the same private image snapshot. Private
images, token IDs and logit snapshots are excluded from the committed results.

[Native measurements](m4-max-allocation-reuse.json),
[allocation counts](m4-max-allocation-reuse-audit.json) and
[full-context allocation audit](m4-max-allocation-reuse-memory.json).
