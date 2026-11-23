# Correctness audit

Checked on November 23, 2026, on an Apple M4 Max with 48 GiB unified memory.
The starting revision is `90e888c`. The checkpoint is the original
`thinkingmachines/Inkling-2-Mini` release at
`6e7c8248f75265bdd4fbab5273304243b5d8c2cb`.
The checkpoint and tokenizer SHA-256 digests, and the config and index Git blob
digests, match the cached release metadata. All four file sizes also match.

## Finding and fix

`generate` built a string with message markers, then tokenized the entire string.
A literal marker in the user's question could therefore become a control token.
For image generation, mentioning `<|unused_200054|>` could add a second image
placeholder block and reject an otherwise valid prompt. Interactive chat already
handled these strings as ordinary text.

Normal generation now uses the same codec as chat. `--raw` still accepts an
explicitly rendered prompt. A regression test failed on the old behavior and
passes with the fix. The checks cover Unicode, literal role markers, literal
image markers, image framing, and raw input. No inference kernels or arithmetic
changed. The fix is committed as `2e28dec`.

## Automated coverage

All 83 existing workspace tests passed, including all 41 normally ignored Metal
tests, run serially. After the fix, all 14 CLI tests passed, including two new
regressions. That is 85 distinct passing tests. Formatting, Clippy across all
targets with warnings denied, and whitespace checks also passed.

| Area | Checks |
| --- | --- |
| Checkpoints | Shard offsets, shapes and dtypes; malformed and truncated files; index path traversal. |
| Model configuration | Release layer/pass schedule, independent caches, unsupported configurations and dimensions. |
| Decoder | Independent scalar reference; FP16 and FP32 KV; decode/prefill agreement; stable greedy selection; reset and continuation. |
| Optimized kernels | Native NVFP4/MXFP8, BF16 matrices, routing ties, nonfinite candidates, convolution history, offsets, tails and output guards. |
| Long history | Analytic attention at 256K positions; ring wrap; causal masking; relative bias; partition reductions; batch limits from 32 to 4,096. |
| Storage reuse | Poisoned allocations, bounded buffer views, resource lifetimes, shared workspace, reset and repeated image/text turns. |
| Vision | Pixel normalization, transparency, padding, patch order, EXIF orientation, dimension/depth limits, CPU encoder reference, mixed image/text prefill. |
| Chat protocol | Native framing, effort and tool declarations, literal marker isolation, incomplete/invalid calls, multiple calls and exact EOS history. |
| Tool loop | Capacity checked before execution, bounded text, complete image reservations, corrupt-image fallback, cancellation, round limits and paused replies. |
| Host tools | Bash output/status, fresh shells, timeout, SIGINT and child cleanup; read pagination, UTF-8 boundaries, snapshots, FIFO rejection and unsupported files. |

## Native model checks

FP16 and FP32 cache runs each match all 24,607,134 saved logits bit for bit:
49,214,268 comparisons with zero mismatches. The native vision encoder passes
all five freshly generated CPU references: 1,495,296 features, zero tolerance
violations, and maximum absolute error `1.0788440704345703e-5`.

Real CLI checks also pass: normal/raw generation parity, image generation with
a literal marker, default chat settings, manual image input, visual recall,
reset, and recovery after a missing image. With tools enabled, the model reads
selected text lines, identifies two images, recalls an image without another
tool call, and reads a file created by Bash. Six token-limit continuations
complete successfully. Ctrl-C cancels the running Bash command; continuation
reports the cancellation, and the delayed background child does not survive.
Answer checks exclude the displayed tool output.

The full-checkpoint numerical comparisons and CLI results are recorded in
[the audit results](m4-max-correctness.json). The reference types are distinct:

- Text logit snapshots detect changes from the earlier rink engine. They are
  not an independent implementation of the released text model. Their hashes
  match the snapshots documented in the earlier
  [long-context validation](m4-max-prefill-budget-logits.json).
- Vision expectations are regenerated with the independent CPU PyTorch script
  from the original checkpoint. Pixel patches must match exactly. Every feature
  must satisfy `abs(actual - expected) <= 2e-5 + 2e-5 * abs(expected)`.
- The CLI harness uses the real checkpoint and generated synthetic files. It
  checks observable behavior, including actual image understanding and Bash
  execution. It is not a numerical oracle or a general model-quality benchmark.

## Reproduce

Run GPU commands one at a time, without another GPU workload:

```sh
cargo test --workspace --offline -- --include-ignored --test-threads=1
cargo clippy --workspace --all-targets --offline -- -D warnings
cargo fmt --all -- --check
cargo build --release --offline -p rink-cli
python3 scripts/check_cli.py
```

The CLI harness uses only Python's standard library. It loads the default local
checkpoint, checks the default 256K chat capacity and effort 0.9, then uses a
small reply budget to force tool-call continuations. `--binary` and `--model`
can select another build or checkpoint directory. Logs and synthetic inputs
stay in a new temporary directory. Bash has direct host access; the test asks
it to create files and run a cancellable child inside that directory.

Regenerate the independent native vision reference with a Python environment
that has PyTorch, NumPy, Pillow and SafeTensors:

```sh
python scripts/vision_reference.py native checkpoints/inkling-2-mini /tmp/rink-vision-reference
cargo run --release --offline -p rink-inkling --example compare_vision -- \
  checkpoints/inkling-2-mini /tmp/rink-vision-reference
```

For text regression, run `compare_revisions compare` against the saved FP16 and
FP32 snapshots. New baselines must be recorded with a separately identified
reference build using `compare_revisions record`; recording and comparing the
same build does not validate an optimization. See the example's usage comments.

## Limits

This audit does not establish correctness for every possible input. The native
text snapshot cases reach 8,192 prompt tokens with 16K cache capacity. The 256K
attention checks use native kernel geometry and controlled data; the CLI check
reserves 256K but uses short conversations. Neither is a full-model conversation
with 256K active tokens. Full native inference at 1M was not run; its cache and
workspace exceed this Mac's available memory.

The independent full-checkpoint oracle covers vision. Full-checkpoint text is
checked against prior rink snapshots, while the independent scalar text oracle
uses small fixtures. This does not claim complete logit parity with an upstream
runtime, or an evaluation of model quality. FP16 KV is checked with its intended
rounding; it is not expected to equal FP32 KV bit for bit.
