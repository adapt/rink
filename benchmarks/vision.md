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

The 44 workspace tests include these image checks:

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
