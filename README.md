# Rink

A Rust inference workspace with model-owned kernels and a separate accelerator API.
The first model is **Inkling 2 Mini**, running text generation on Apple Silicon using
Metal. Inference runs entirely in Rust and Metal; no Python, MLX, or external model
server is involved.

This first implementation supports the original Hugging Face checkpoint, including
native NVFP4 and MXFP8 expert weights, the three-pass loop, independent KV and
convolution state per layer visit, attention residuals, cross-pass message gates,
local/global relative attention, and top-8 routing. It is a correctness-oriented,
single-sequence decoder: prompt tokens are processed sequentially, activations and
caches use float32, and sampling is greedy. Image input, CUDA execution, speculative
decoding, batching, and optimized prompt processing are follow-up work.

## Run

Use a Rust toolchain supporting edition 2024 and macOS with a Metal-capable Apple GPU.
The original checkpoint is about 19.5 GB; 32 GB or more unified memory is recommended.
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
cargo run --release -- generate \
  --model checkpoints/inkling-2-mini \
  --prompt 'Explain why the sky is blue in one sentence.' \
  --effort 0.1 --max-tokens 128 --context 2048
```

The CLI renders the released single-user-message chat format and stops at the
configured EOS token. Protocol tokens remain visible, including thinking and answer
boundaries. `--raw` accepts an already rendered prompt or a plain completion prompt.
`--effort` accepts a number in `[0, 0.99]` (release default: `0.9`). The complete
prompt and requested continuation must fit within `--context`; oversized requests
fail before loading weights. Checkpoints and build output are ignored by Git.

Only the original tensor layout is accepted; MLX conversions, GGUF files, and the
variable-loop research checkpoint are not supported. Checkpoint files must remain
unchanged while the model loads them. The vision and drafter weights are not loaded.

## Workspace and API

| Crate | Responsibility |
| --- | --- |
| `rink-core` | Accelerator/buffer contracts, ordered kernel dispatch, model session interface, greedy sampling |
| `rink-metal` | Metal devices, buffers, shader compilation, command submission and synchronization |
| `rink-inkling` (`models/inkling`) | Configuration, native weight layout, loop schedule, caches, decoder and model-specific Metal kernels |
| `rink-cli` | Local checkpoint inspection, tokenization, chat rendering and streamed text generation |

```rust,ignore
use rink_core::AutoregressiveModel;
use rink_inkling::Inkling;
use rink_metal::Metal;
use std::path::Path;

let mut model = Inkling::load(Metal::new()?, Path::new("checkpoints/inkling-2-mini"), 2048)?;
let logits = model.step(123)?; // One input token; logits predict the following token.
let next = rink_core::greedy(&logits)?;
model.reset()?;               // Clears every loop pass's KV and convolution state.
# Ok::<(), anyhow::Error>(())
```

`Inkling<A>` is generic over `Accelerator`; the decoder never imports Metal types.
Accelerators manage buffers, compile model-supplied kernels, enqueue ordered work,
and synchronize on host reads. The low-level dispatch method is unsafe because
its caller owns the kernel ABI and buffer bounds; the public model API validates
configuration, tensor shapes, token IDs, and sequence capacity. A failed decoding
step poisons the session until `reset()`.

To add another model, create a crate under `models/`, add it to the workspace, and
implement `AutoregressiveModel` with that model's configuration, loader and kernels.
To add CUDA, implement `Accelerator` in a new backend crate and supply CUDA variants
of the model's kernel library. `AcceleratorKind::Cuda` reserves that backend identity;
it does not select a working CUDA implementation today. Each session currently uses
one accelerator; multi-device model sharding is not implemented.

## Validation

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
tests separately cover native FP4/FP8 decoding, BF16 matrix multiplication, and biased
top-k selection with unbiased mixture weights. Regenerate the fixtures with
`python3 scripts/make_test_fixture.py` (standard library only).

Float32 activations and direct dequantization intentionally differ from the release
GPU runtimes' activation quantization and BF16 rounding; bitwise agreement with those
runtimes is not expected. This implementation does not claim benchmark parity or
optimized inference performance.

Full-checkpoint smoke test, November 20, 2026: on an Apple M4 Max with 48 GiB unified
memory, the pinned checkpoint above (SHA256
`65e60f51aa1284d4fb022584908c315270334d991828bb774af60584523539d1`) loaded and generated
the final answer `4` to `What is 2 + 2? Answer briefly.` With `--effort 0
--max-tokens 64 --context 128`, the 24-token prompt took 3.64 seconds to process and
39 generated tokens took 3.35 seconds, stopping at EOS. These counts include
reasoning and protocol tokens; timings exclude loading. This is a functional smoke
test, not an accuracy evaluation or a general performance benchmark.

## References

- [Inkling 2 Mini release](https://thinkingmachines.ai/news/inkling-2-mini/)
- [Original checkpoint and configuration](https://huggingface.co/thinkingmachines/Inkling-2-Mini/tree/6e7c8248f75265bdd4fbab5273304243b5d8c2cb)
- [Hugging Face looped decoder reference](https://github.com/huggingface/transformers/blob/6173a2f679ff4ed393afc499aefc569da7e68bc1/src/transformers/models/inkling2_loop/modeling_inkling2_loop.py)
- [Native quantization and message-query conversion](https://github.com/huggingface/transformers/blob/6173a2f679ff4ed393afc499aefc569da7e68bc1/src/transformers/core_model_loading.py)
