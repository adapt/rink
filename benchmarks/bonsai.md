# Bonsai 2.5 performance

Measured on November 24, 2026, on an Apple M4 Max with 48 GiB of unified memory.
The checkpoint is PrismML's MLX pack at revision
`aa432fd0e17612f6684e728d4d3c42b6656bcabb`. Its SHA-256 was checked before use.
[Raw measurements and reference checks](m4-max-bonsai-fused.json) accompany this
report. The targets of 1,000 tok/s prefill and 60 tok/s decode are **not met**.

| Prompt tokens | Before prefill | Current prefill | Local MLX prefill | Before decode | Current decode | Local MLX decode |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 200.4 | 230.4 | 263.2 | 35.1 | 36.8 | 35.6 |
| 512 | 224.6 | 240.7 | 266.8 | 34.4 | 36.6 | 34.0 |
| 1,024 | 205.6 | 234.2 | 221.7 | 34.2 | 36.5 | 34.2 |

All rates are tokens per second, with 4,096 tokens of reserved context in Rink.
Figures are medians of three wall-clock trials. The before binary is commit
`85229e7`. Each trial starts with an empty conversation and uses 32 fixed decode
steps, including steps after EOS. Four warmup steps precede the trials. Kernel
profiling is off. Load and reset time are excluded. Generated token IDs match in
all paired before/after and MLX trials.

The MLX figures are our local measurements, not published PrismML numbers. The
harness uses the released packed runtime and shared Hadamard rotations with
MLX 0.32.0, MLX-LM 0.31.3, and MLX-VLM 0.6.3. It reads back each selected token,
as Rink does. MLX prefills the whole prompt; Rink uses batches of at most 512.
Throughput varies across runs: earlier retained-kernel trials reached about
255 tok/s at 512 tokens. MLX's 1,024-token results also varied substantially.
These results show a decode improvement over MLX, but not an overall prefill win.

With 262,144 tokens of reserved capacity, current medians are 231.2/37.3 tok/s
for 128-token prefill/decode and 210.6/37.7 for 512-token prefill/decode. These
short prompts do **not** measure throughput with 256K tokens of active history.

Metal allocation is 7.64 GiB at 4K capacity and 23.56 GiB at 256K. The model
buffers and cache policy are unchanged; the device reports an additional 256 KiB
with the new pipelines. These figures are not process RSS or total system memory.
MLX grows its caches, so its peak memory is not a like-for-like capacity comparison.
Neither prefill nor decode makes device allocations after loading. Decode reads
back four bytes per step.

## Changes

The retained kernels use a specialized 128-wide GDN recurrence, a smaller tile
for dense gate projections, and exact symmetric dequantization in packed prefill
GEMMs. Gate and up projections share their input tile and fuse SiLU into the
output store. This removes 128 dispatches per decoded token (1,527 to 1,399).
Fusion requires matching rotation signs and eligible symmetric weights; other
matrices retain the separate affine projection path.

Weights remain packed. FP16 projection boundaries, FP32 accumulators and recurrent
state, and disabled Metal fast math are retained. Both fused projections round
to FP16 before activation, as the separate kernels do. Different reduction orders
produce small numerical differences from the reference runtime. Prefill remains
512 tokens per batch and uses the existing workspace.

Larger tiles, alternate layouts, split-K projections, and several recurrence
mappings did not provide repeatable gains. Those experiments are not enabled.

## Reproduce

```sh
cargo build --release -p rink-cli
target/release/rink bench --model checkpoints/bonsai-2.5-27b \
  --prompt-tokens 128,512,1024 --decode-tokens 32 --repetitions 3 \
  --context 4096 --prefill-batch 512
target/release/rink bench --model checkpoints/bonsai-2.5-27b \
  --prompt-tokens 128,512 --decode-tokens 32 --repetitions 3 --context 262144
python scripts/bonsai_benchmark_mlx.py --model checkpoints/bonsai-2.5-27b \
  --runtime /path/to/released/runtime --prompt-tokens 128,512,1024 \
  --decode-tokens 32 --repetitions 3
```

For the before result, build `85229e7` and use the same Rink commands.
Historical measurements: [initial implementation](m4-max-bonsai.json),
[first projection changes](m4-max-bonsai-projections.json), and
[batched prefill](m4-max-bonsai-batched.json).

## Correctness

The independent oracle uses the release's `Packed` module and the MLX versions
above. It loads the same checkpoint, casts auxiliary weights to FP16 as the
published loader does, and runs one token at a time. Inputs cover arithmetic and
a conversation with a remembered codeword plus distractors.

Batched comparisons check 9,932,800 logits at 40 positions across 25-, 474-,
and 1,025-token inputs. The longest check includes two full 512-token batches
followed by cached decode. All 40 greedy predictions match; maximum relative RMS
logit error is 0.6186%. A separate token-at-a-time run over the 474-token input
matches all 21 saved predictions, with maximum relative RMS error 0.5669%.
The gate remains 1% relative RMS error with exact greedy agreement. These checks
do not establish equivalence for every prompt or a full 256K conversation.

Nine GPU tests cover attention, cached continuation, reset, packed projections,
recurrence, fusion, and sampling. Fused MLP outputs match the separate kernels
byte for byte on the tested inputs. Scalar checks cover native widths, all four
weight codes, signed scales, generic affine biases, partial tiles, and guards.
Host checks exhaust all admitted FP16 scales for symmetric dequantization;
GPU checks include subnormal and largest admitted scale boundaries. Existing
tolerances are unchanged. Workspace tests, formatting, and strict Clippy pass.

```sh
cargo test --release -p rink-bonsai --test metal -- --ignored --test-threads=1
```
