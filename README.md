# Rink

A Rust inference workspace with model-owned kernels and a separate accelerator API.
The first model is **Inkling 2 Mini**, running text generation on Apple Silicon using
Metal. Inference runs entirely in Rust and Metal; no Python, MLX, or external model
server is involved.

This first implementation supports the original Hugging Face checkpoint, including
native NVFP4 and MXFP8 expert weights, the three-pass loop, independent KV and
convolution state per layer visit, attention residuals, cross-pass message gates,
local/global relative attention, and top-8 routing. The single-sequence decoder
processes prompts in batches of up to 32 tokens using SIMD matrix multiplication,
grouped native-quantized expert projections, and tiled attention. Routing and wide
normalizations run in parallel, and controller projections are fused with gating.
During decoding, expert gate/up projections include SwiGLU, and convolution
updates incorporate expert summation and residual additions.
Scratch buffers and conversation caches are reused, and greedy sampling runs on the
GPU. KV caches use float16 by default; activations, convolution history, and attention
accumulation use float32. Attention shares one temporary workspace across all layer
visits. Image input, CUDA execution, speculative decoding, and batching across
requests are follow-up work.

## Run

Use a Rust toolchain supporting edition 2024 and macOS with a Metal-capable Apple GPU.
The original checkpoint is about 19.5 GB. The default 256K chat context targets Macs
with 48 GB or more unified memory; on a 32 GB Mac, use a smaller `--context`, such as
`4096`.
Cache memory depends on the requested context. Metal allocations are bounded by the
device's recommended working set, so the model's advertised 1M-token context is not
a promise that that context fits on a particular Mac.

Download the original files with the Hugging Face CLI (download only; not a runtime
dependency), or place the same files in a local directory:

```sh
hf download thinkingmachines/Inkling-2-Mini \
  --revision 6e7c8248f75265bdd4fbab5273304243b5d8c2cb \
  --local-dir checkpoints/inkling-2-mini \
  --include config.json tokenizer.json model.safetensors.index.json model-00001-of-00001.safetensors

cargo run --release -- inspect --model checkpoints/inkling-2-mini
cargo run --release -- chat

# Or run a single prompt:
cargo run --release -- generate \
  --model checkpoints/inkling-2-mini \
  --prompt 'Explain why the sky is blue in one sentence.' \
  --effort 0.1 --max-tokens 128 --context 2048
```

`chat` opens a terminal conversation. Enter one message per line; replies stream as
they are generated. Weights load once, and conversation history stays in the model's
cache until `/reset` or exit. `/quit` exits, `/help` lists commands, and `/continue`
resumes a reply that hit the token limit. Finish or reset a paused reply before
sending another message. Protocol markers are hidden; add `--show-thinking` to see
the reasoning as well as the answer.

Chat defaults to `--model checkpoints/inkling-2-mini --effort 0.9 --max-tokens 512
--context 262144 --kv-cache f16`. The token limit includes reasoning and protocol
tokens; the context limit covers the entire conversation. A new message must leave room for the reply
budget. When the context fills, use `/reset` or restart with a larger `--context`.
History is kept in memory and is not saved to disk.

`chat`, `generate`, and `bench` accept `--kv-cache f16` (default) or `--kv-cache f32`
for precision comparisons. Both cached and newly projected keys/values are rounded
to the selected storage precision before attention, consistently across prompt
chunks and decoding. At 256K, the FP16 KV cache uses about 8.16 GiB; the shared
32-token attention workspace uses at most 780 MiB for this model. Weights and other
scratch buffers require additional memory. Cache capacity is allocated at load time;
attention workspace grows when a larger prompt batch is first used and is then reused.
`generate` and `bench` retain their smaller context defaults; pass `--context 262144`
explicitly when using them to test the chat capacity.

`generate` renders the released single-user-message chat format and stops at EOS.
It leaves protocol markers visible, including thinking and answer boundaries.
`--raw` accepts an already rendered prompt or a plain completion prompt. Its default
effort is `0.9`; both commands accept `--effort` in `[0, 0.99]`. For single-prompt
generation, the complete prompt and requested continuation must fit within
`--context`; oversized requests fail before loading weights. Checkpoints and build
output are ignored by Git.

Only the original tensor layout is accepted; MLX conversions, GGUF files, and the
variable-loop research checkpoint are not supported. Checkpoint files must remain
unchanged while the model loads them. The vision and drafter weights are not loaded.

## Workspace and API

| Crate | Responsibility |
| --- | --- |
| `rink-core` | Accelerator/buffer contracts, ordered kernel dispatch, model session interface, greedy sampling |
| `rink-metal` | Metal devices, buffers, shader compilation, command submission and synchronization |
| `rink-inkling` (`models/inkling`) | Configuration, native weight layout, loop schedule, caches, decoder and model-specific Metal kernels |
| `rink-cli` | Checkpoint inspection, tokenization, terminal chat, streamed generation and performance benchmarks |

```rust,ignore
use rink_core::AutoregressiveModel;
use rink_inkling::Inkling;
use rink_metal::Metal;
use std::path::Path;

let mut model = Inkling::load(Metal::new()?, Path::new("checkpoints/inkling-2-mini"), 2048)?;
let next = model.prefill_greedy(&[123, 456])?; // Process a prompt and select its next token.
let following = model.step_greedy(next)?;    // Consume that token and select another.
model.reset()?; // Start a new sequence while retaining cache and scratch allocations.
# Ok::<(), anyhow::Error>(())
```

`Inkling<A>` is generic over `Accelerator`; the decoder never imports Metal types.
Accelerators manage buffers, compile model-supplied kernels, enqueue ordered work,
and synchronize on host reads. The low-level dispatch method is unsafe because
its caller owns the kernel ABI and buffer bounds; the public model API validates
configuration, tensor shapes, token IDs, and sequence capacity. A failed decoding
step poisons the session until `reset()`.
`step()` and `prefill()` also expose full logits for callers implementing their own
sampling. The greedy methods return only a token ID and avoid reading the vocabulary
back to the CPU. `reset()` invalidates previous sequence state without reallocating
or clearing the underlying buffers; kernels read only entries valid for the new
sequence.

To add another model, create a crate under `models/`, add it to the workspace, and
implement `AutoregressiveModel` with that model's configuration, loader and kernels.
To add CUDA, implement `Accelerator` in a new backend crate and supply CUDA variants
of the model's kernel library. `AcceleratorKind::Cuda` reserves that backend identity;
it does not select a working CUDA implementation today. Each session currently uses
one accelerator; multi-device model sharding is not implemented.

## Validation

Run fixed-work performance trials with the same checkpoint, context, and token counts:

```sh
cargo run --release -- bench --prompt-tokens 32,128 --decode-tokens 32 \
  --repetitions 3 --context 4096 > benchmark.json
```

The benchmark loads once, warms the runtime, and resets the conversation before each
trial. JSON records load/reset/prefill/decode timings, generated token IDs, Metal
allocation and dispatch counts, CPU allocation/encoding time, GPU command-buffer
execution time, synchronization wait time, and readback bytes. GPU and wait timings
overlap and must not be added together. Decode uses fixed token counts through EOS.
`--reference` forces single-token prefill and CPU sampling for comparison.
It still uses the current math kernels, so it is not the original implementation.
For diagnostic GPU time by kernel, add `--profile-kernels`. This mode timestamps
each dispatch using a separate compute pass and changes scheduling; disable it
when measuring normal throughput. The JSON `kernels` entries report invocation
counts and GPU durations for each phase.
Recorded M4 Max results, methodology, and remaining bottlenecks are in
[the benchmark report](benchmarks/README.md), including before/after measurements
and raw JSON for each optimization pass.

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
# On an Apple Silicon machine, explicitly include hardware tests:
cargo test --workspace -- --include-ignored
```

Hardware tests fail if a Metal GPU is unavailable; they never silently fall back to
CPU. The checked-in tiny checkpoint and scalar Python reference exercise all decoder
components, three loop passes, separate pass state, full and sliding attention,
cache wraparound, reset, log scaling, vocabulary padding, and context bounds. Kernel
tests separately cover native FP4/FP8 decoding, BF16 matrix multiplication, partial
SIMD tiles, partitioned causal attention, GPU argmax, and biased top-k selection with
unbiased mixture weights. Further tests cover routing ties and nonfinite inputs,
wide normalization, fused controllers with mixed weight formats, and grouped
FP4/FP8/float32 experts with empty expert groups, sparse routes, partial tiles, and
input/output reordering. Batched prefill is compared against single-token execution
across chunk boundaries, sliding-window wraparound, reset, and subsequent decoding.
Fused expert activation is compared against separate projection and SwiGLU kernels;
fused residual convolutions verify both outputs and history across chunks and resets.
FP16 and FP32 caches have independent scalar decoder references. Cache tests cover
rounding, ring writes, memory guards, and attention at 8K, 64K, and 256K positions;
a memory test checks that workspace grows with the largest layer rather than the
number of layer visits, and is reused across chunks, decoding, and reset.
Regenerate the fixtures with
`python3 scripts/make_test_fixture.py` (standard library only).
Chat session tests cover exact token history across turns, EOS, continuation, reset,
context bounds, and recovery from failed decoding steps.

Float32 activations and direct dequantization intentionally differ from the release
GPU runtimes' activation quantization and BF16 rounding; bitwise agreement with those
runtimes is not expected. The performance measurements compare revisions of this
implementation; they do not establish accuracy or performance parity with the
release runtimes.

Full-checkpoint smoke test, November 20, 2026: on an Apple M4 Max with 48 GiB unified
memory, the pinned checkpoint above (SHA256
`65e60f51aa1284d4fb022584908c315270334d991828bb774af60584523539d1`) loaded and generated
the final answer `4` to `What is 2 + 2? Answer briefly.` This is a functional smoke
test, not an accuracy evaluation. The optimized decoder also reproduces every
generated token ID from the recorded 32- and 128-token baseline benchmark trials.
Terminal chat was also verified with this checkpoint: recall across turns, reset,
continuation across eight-token chunks, and both answer-only and thinking displays.

## References

- [Inkling 2 Mini release](https://thinkingmachines.ai/news/inkling-2-mini/)
- [Original checkpoint and configuration](https://huggingface.co/thinkingmachines/Inkling-2-Mini/tree/6e7c8248f75265bdd4fbab5273304243b5d8c2cb)
- [Hugging Face looped decoder reference](https://github.com/huggingface/transformers/blob/6173a2f679ff4ed393afc499aefc569da7e68bc1/src/transformers/models/inkling2_loop/modeling_inkling2_loop.py)
- [Native quantization and message-query conversion](https://github.com/huggingface/transformers/blob/6173a2f679ff4ed393afc499aefc569da7e68bc1/src/transformers/core_model_loading.py)
