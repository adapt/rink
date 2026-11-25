# Bonsai 2.5 27B

This crate runs the language model from PrismML's
[official Hadamard-aware MLX pack](https://huggingface.co/prism-ml/Ternary-Bonsai-2.5-27B-mlx-2bit/tree/aa432fd0e17612f6684e728d4d3c42b6656bcabb).
It owns the model configuration, weight layout, chat format, forward pass, and
Metal kernels. It uses the shared `rink-core` accelerator and checkpoint APIs.
It has no dependency on the Inkling crate or on a Python runtime.

See the [M4 Max measurements](../../benchmarks/bonsai.md) for measured throughput,
memory use, and native checkpoint comparisons.

## Run

Use the download command in the root README. Select the checkpoint directory:

```sh
cargo run --release -- chat --model checkpoints/bonsai-2.5-27b
cargo run --release -- generate --model checkpoints/bonsai-2.5-27b \
  --prompt 'What is 2 plus 2?' --effort 0
cargo run --release -- bench --model checkpoints/bonsai-2.5-27b \
  --prompt-tokens 32,128 --decode-tokens 32 --repetitions 3
```

Model selection reads `model_type` in `config.json`; the directory name does not
matter. `inspect` shows the architecture and cache size before loading weights.
GGUF files and ordinary Qwen checkpoints are not accepted by this loader.

Text chat keeps history in both the KV cache and recurrent state. `/continue`
resumes a reply. `/reset` clears both caches. Both native end tokens stop replies.
The default effort `0.9` selects `xhigh` reasoning. Effort `0` disables reasoning;
values below `1/3` select `low`, and values from `1/3` to below `2/3` select `medium`.

This version supports text input, batched prefill, and the `bash` and `read` tools.
Enable tools with `--tool bash,read --cwd /path/to/project`. Commands run directly
with your user permissions. Tool timeouts, output limits, cancellation, and
`/continue` use the same rules as Inkling. Calls execute after a complete assistant
response; a pause inside a call does not execute it. Tool replies extend the cached
conversation without replaying earlier calls.

Bonsai uses its native XML-style calls and groups tool results in a user message.
The next assistant response uses the selected reasoning effort. `read` supports
text files; an image returns an unsupported-input error. Images and MTP are not
implemented. Bonsai rejects `--image` and `--kv-cache f32`.

Prefill processes up to 512 tokens per batch by default. `--prefill-batch` selects
a power of two from 32 through 4,096. The workspace is allocated once. Larger
batches need more activation memory. Recurrent updates stay in causal order;
full attention masks future tokens. Only the last prompt token uses the vocabulary
projection. Decode selects the next token on the GPU and reads back four bytes.

## Storage and arithmetic

The checkpoint has 64 decoder layers: 48 gated recurrent-attention layers and
16 full-attention layers. Packed matrices stay in affine 2-bit/group-128 form.
Each projection applies the declared signed Hadamard transform to its input.
Embedding lookup applies the inverse transform after unpacking a single row.
The loader validates every packed-module manifest entry, tensor shape, and sign
vector. It does not upload the unused vision tower or an MTP head.

Activations, auxiliary weights, and KV storage use FP16, as in the published
runtime. Dot products, normalization, attention softmax, Hadamard transforms,
and recurrent state use FP32 arithmetic. The packed projection fast path applies
only when the stored FP16 bias is exactly the negative scale and all four weight
values are representable. Decode factors that scale after a partial dot product.
Prefill unpacks those weights with exact FP16 arithmetic and FP32 matrix
accumulators. Other affine matrices retain their generic kernels. Compatible
gate and up projections share one kernel with SiLU; both projection results
still round to FP16 before the activation. Recurrent heads of width 128 use
vector loads and retain FP32 state. Reduction orders can differ from MLX; the
represented weights and arithmetic precision do not change. The context limit is 262,144 tokens.
At that capacity, KV and recurrent caches reserve 16.14 GiB. Attention workspace
grows linearly with the configured capacity; it does not hold a quadratic score
matrix. The same allocations are reused throughout the conversation.

## Verification

The tiny fixture uses deterministic synthetic weights, not model weights.
`scripts/bonsai_reference.py` generates expected logits with the published
`Packed` loader and MLX's Qwen3.5 implementation. The GPU test checks both
attention types, three value heads per recurrent key head, signed transforms,
positions on both sides of an attention partition boundary, split prefill,
invalid requests, and reset. It compares greedy tokens and all saved logits.
FP16 reductions can differ between runtimes; the test uses an explicit numerical
tolerance. This is not a claim of bitwise parity with MLX.

Run the model tests on a Mac with Metal:

```sh
cargo test -p rink-bonsai -- --include-ignored --test-threads=1
```

The native chat test also needs the downloaded tokenizer. Its 12 expected token
sequences were produced by the released Jinja template and tokenizer, covering
all four effort modes, Unicode, and multiline text. Tool fixtures check declarations,
multiple calls, typed arguments, and grouped results in all four effort modes.
Parser tests cover malformed calls, pauses, and literal delimiters in input.
The CLI tests check execution once, context limits, reset, and cached continuation.

Run native tool checks with `python3 scripts/check_bonsai_cli.py` after a release
build. The check uses synthetic files in a temporary directory and includes Bash,
file reading, missing files, unsupported images, and Ctrl-C cancellation.
Regenerate chat fixtures with `python scripts/bonsai_chat_reference.py checkpoints/bonsai-2.5-27b`
in a Python environment with Transformers installed. This generator also needs
the release's `chat_template.jinja` and `tokenizer_config.json` in that directory.

To regenerate the numerical fixture, install the release's
`runtime/requirements.txt` in a temporary Python environment. Download the release's
`runtime/` files beside that environment, then run:

```sh
python scripts/bonsai_reference.py \
  --runtime /path/to/release/runtime --model checkpoints/bonsai-2.5-27b \
  --fixture models/bonsai/tests/fixtures/tiny \
  --output models/bonsai/tests/fixtures/tiny/expected.json
```

For full-checkpoint comparisons, omit `--fixture` and pass `--tokens` with a JSON
array of token IDs. Compare the saved reference with:

```sh
cargo run --release -p rink-bonsai --example compare_reference -- \
  checkpoints/bonsai-2.5-27b /path/to/reference.json
# Compare batched prefill at the same saved positions:
cargo run --release -p rink-bonsai --example compare_reference -- \
  checkpoints/bonsai-2.5-27b /path/to/reference.json 512
```

Reference versions: PrismML revision `aa432fd0e17612f6684e728d4d3c42b6656bcabb`,
MLX `0.32.0`, MLX-LM `0.31.3`, MLX-VLM `0.6.3`. The model pack is licensed by
PrismML under Apache 2.0; rink's MIT license does not replace the model license.
