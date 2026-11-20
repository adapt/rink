#![cfg(target_os = "macos")]
use rink_core::{Accelerator, AutoregressiveModel};
use rink_inkling::{
    Inkling,
    kernels::{Kernels, Params},
};
use rink_metal::{Metal, MetalBuffer};

fn upload(d: &mut Metal, values: &[f32]) -> MetalBuffer {
    d.upload(bytemuck::cast_slice(values)).unwrap()
}
fn read(d: &mut Metal, b: &MetalBuffer) -> Vec<f32> {
    d.read(b)
        .unwrap()
        .chunks_exact(4)
        .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
        .collect()
}
fn close(a: &[f32], b: &[f32], tol: f32) {
    assert_eq!(a.len(), b.len());
    for (i, (&a, &b)) in a.iter().zip(b).enumerate() {
        assert!(
            a.is_finite() && (a - b).abs() < tol,
            "index {i}: actual {a}, expected {b}, tolerance {tol}"
        );
    }
}

// These tests are explicit hardware tests, never silently passed when no GPU exists.
#[test]
#[ignore = "requires a Metal GPU"]
fn parallel_residual_reads_match_single_simd_with_offsets_and_tails() {
    let mut device = Metal::new().unwrap();
    let kernels = Kernels::compile(&mut device).unwrap();
    for width in [17, 2304] {
        for batch in [1, 3] {
            for blocks in [1, 2, 7, 32] {
                let values: Vec<_> = (0..(blocks + 3) * batch * width)
                    .map(|i| ((i * 17 % 211) as f32 - 105.) / 33.)
                    .collect();
                let bank = upload(&mut device, &values);
                let live = upload(&mut device, &values[..batch * width]);
                let query = upload(&mut device, &values[..2 * width]);
                let expected = device.allocate(batch * width * 4).unwrap();
                let actual = device.allocate(batch * width * 4).unwrap();
                let params = Params::new(&[width, blocks, width, 3, batch], &[]);
                // SAFETY: block bank includes a three-block offset and all saved
                // blocks; the live state and query cover their requested rows.
                unsafe {
                    kernels
                        .run(
                            &mut device,
                            "block_read",
                            &[&expected, &bank, &live, &query],
                            params,
                            batch * 32,
                            32,
                        )
                        .unwrap();
                    kernels
                        .run(
                            &mut device,
                            "block_read_parallel",
                            &[&actual, &bank, &live, &query],
                            params,
                            batch * 256,
                            256,
                        )
                        .unwrap();
                }
                close(
                    &read(&mut device, &actual),
                    &read(&mut device, &expected),
                    0.000001,
                );
            }
        }
    }
}

#[test]
#[ignore = "requires a Metal GPU; run cargo test --workspace -- --include-ignored"]
fn tiny_decoder_matches_scalar_reference_and_reset() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny");
    #[derive(serde::Deserialize)]
    struct Expected {
        tokens: Vec<u32>,
        logits: Vec<Vec<f32>>,
    }
    let e: Expected = serde_json::from_str(include_str!("fixtures/tiny/expected.json")).unwrap();
    let mut model = Inkling::load(Metal::new().unwrap(), &path, e.tokens.len()).unwrap();
    let mut allocations = 0;
    for (i, (&id, expected)) in e.tokens.iter().zip(&e.logits).enumerate() {
        close(&model.step(id).unwrap(), expected, 0.0002);
        if i == 0 {
            allocations = model.accelerator().stats().allocations;
        }
        assert_eq!(
            model.accelerator().stats().allocations,
            allocations,
            "scratch should be reused after the first token"
        );
    }
    assert_eq!(model.position(), e.tokens.len());
    assert!(model.step(1).is_err());
    model.reset().unwrap();
    assert_eq!(
        model.accelerator().stats().allocations,
        allocations,
        "reset should reuse cache allocations"
    );
    assert_eq!(model.position(), 0);
    assert!(model.step(15).is_err());
    close(&model.step(e.tokens[0]).unwrap(), &e.logits[0], 0.0002);
}

#[test]
#[ignore = "requires a Metal GPU"]
fn native_quantized_experts_match_scalar_values() {
    let mut d = Metal::new().unwrap();
    let k = Kernels::compile(&mut d).unwrap();
    // All 16 FP4 codes, with asymmetric signs, block scales and per-expert factors.
    let packed: Vec<u8> = (0..32)
        .map(|i| (i % 16) as u8 | (((15 - i % 16) as u8) << 4))
        .collect();
    let weights = d.upload(&packed).unwrap();
    let scales = d.upload(&[0x38, 0x40, 0x30, 0x48]).unwrap(); // 1, 2, 0.5, 4
    let factors = upload(&mut d, &[0.25, 2.]);
    let x: Vec<f32> = (0..32).map(|i| (i as f32 - 13.) / 10.).collect();
    let input = upload(&mut d, &x);
    let routing = upload(&mut d, &[1., 0.4, 0., 0.6]);
    let out = d.allocate(8).unwrap();
    // SAFETY: two experts, one 32-column row each, two selected slots, matching buffers.
    unsafe {
        k.run(
            &mut d,
            "expert_matvec",
            &[&out, &weights, &scales, &factors, &input, &routing],
            Params::new(&[1, 32, 2, 0, 0, 1], &[]),
            64,
            32,
        )
        .unwrap();
    }
    let lut = [0., 0.5, 1., 1.5, 2., 3., 4., 6.];
    let mut expected = Vec::new();
    for expert in [1, 0] {
        let mut sum = 0.;
        for (i, &x) in x.iter().enumerate() {
            let idx = expert * 32 + i;
            let code = (packed[idx / 2] >> (4 * (idx % 2))) & 15;
            sum += lut[(code & 7) as usize]
                * if code & 8 != 0 { -1. } else { 1. }
                * [1., 2., 0.5, 4.][idx / 16]
                * [0.25, 2.][expert]
                * x;
        }
        expected.push(sum);
    }
    close(&read(&mut d, &out), &expected, 0.0001);

    let batch_input = upload(
        &mut d,
        &x.iter()
            .copied()
            .chain(x.iter().map(|v| -2. * v))
            .collect::<Vec<_>>(),
    );
    let batch_routes = upload(&mut d, &[1., 0.4, 0., 0.6, 0., 0.25, 1., 0.75]);
    let batch_out = d.allocate(16).unwrap();
    // SAFETY: two tokens, each selecting two experts, with independent 32-value inputs.
    unsafe {
        k.run(
            &mut d,
            "expert_matvec",
            &[
                &batch_out,
                &weights,
                &scales,
                &factors,
                &batch_input,
                &batch_routes,
            ],
            Params::new(&[1, 32, 2, 0, 0, 2], &[]),
            128,
            32,
        )
        .unwrap();
    }
    close(
        &read(&mut d, &batch_out),
        &[
            expected[0],
            expected[1],
            -2. * expected[1],
            -2. * expected[0],
        ],
        0.0002,
    );

    let fp8: Vec<u8> = (0..64)
        .map(|i| [0x00, 0x01, 0x38, 0xbc, 0x7e, 0x80, 0x10, 0xf0][i % 8])
        .collect();
    let weights = d.upload(&fp8).unwrap();
    let scales = d.upload(&[126, 129]).unwrap();
    // SAFETY: same geometry, now with 64 FP8 weights and two 32-element block scales.
    unsafe {
        k.run(
            &mut d,
            "expert_matvec",
            &[&out, &weights, &scales, &factors, &input, &routing],
            Params::new(&[1, 32, 2, 1, 0, 1], &[]),
            64,
            32,
        )
        .unwrap();
    }
    let values = [0., 1. / 512., 1., -1.5, 448., -0., 0.03125, -128.];
    let base: f32 = x.iter().enumerate().map(|(i, x)| x * values[i % 8]).sum();
    close(&read(&mut d, &out), &[base * 4., base * 0.5], 0.002);
    // SAFETY: same two-token geometry, now using FP8 weights and scales.
    unsafe {
        k.run(
            &mut d,
            "expert_matvec",
            &[
                &batch_out,
                &weights,
                &scales,
                &factors,
                &batch_input,
                &batch_routes,
            ],
            Params::new(&[1, 32, 2, 1, 0, 2], &[]),
            128,
            32,
        )
        .unwrap();
    }
    close(
        &read(&mut d, &batch_out),
        &[base * 4., base * 0.5, -base, -base * 8.],
        0.004,
    );
}

#[test]
#[ignore = "requires a Metal GPU"]
fn routing_bias_selects_experts_without_changing_mixture_weights() {
    let mut d = Metal::new().unwrap();
    let k = Kernels::compile(&mut d).unwrap();
    let logits = upload(&mut d, &[0., 1., -1.]);
    let bias = upload(&mut d, &[0., -10., 10.]);
    let out = d.allocate(16).unwrap();
    // SAFETY: three scores and biases, two output pairs; one thread executes selection.
    unsafe {
        k.run(
            &mut d,
            "route",
            &[&out, &logits, &bias],
            Params::new(&[3, 2, 1], &[8.]),
            1,
            1,
        )
        .unwrap();
    }
    let s = 1. / (1. + 1f32.exp());
    close(
        &read(&mut d, &out),
        &[2., 8. * s / (s + 0.5), 0., 4. / (s + 0.5)],
        0.00001,
    );
}

#[test]
#[ignore = "requires a Metal GPU"]
fn matvec_handles_bfloat16_and_non_aligned_dimensions() {
    let mut d = Metal::new().unwrap();
    let k = Kernels::compile(&mut d).unwrap();
    let x: Vec<f32> = (0..37).map(|i| (i as f32).cos()).collect();
    let w: Vec<half::bf16> = (0..111)
        .map(|i| half::bf16::from_f32((i as f32).sin()))
        .collect();
    let bytes: Vec<u8> = w.iter().flat_map(|v| v.to_bits().to_le_bytes()).collect();
    let weights = d.upload(&bytes).unwrap();
    let input = upload(&mut d, &x);
    let out = d.allocate(12).unwrap();
    // SAFETY: three rows of 37 BF16 elements, 37 F32 inputs, 3 F32 outputs; padded grid is guarded.
    unsafe {
        k.run(
            &mut d,
            "matvec",
            &[&out, &weights, &input],
            Params::new(&[3, 37, 1], &[]),
            96,
            128,
        )
        .unwrap();
    }
    let expected: Vec<f32> = w
        .chunks_exact(37)
        .map(|row| row.iter().zip(&x).map(|(a, b)| a.to_f32() * b).sum())
        .collect();
    close(&read(&mut d, &out), &expected, 0.00001);
}

#[test]
#[ignore = "requires a Metal GPU"]
fn prefill_and_gpu_sampling_match_scalar_reference_across_reset() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny");
    let expected: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/tiny/expected.json")).unwrap();
    let tokens: Vec<u32> = serde_json::from_value(expected["tokens"].clone()).unwrap();
    let logits: Vec<Vec<f32>> = serde_json::from_value(expected["logits"].clone()).unwrap();
    let mut model = Inkling::load(Metal::new().unwrap(), &path, tokens.len()).unwrap();
    assert!(model.prefill(&[]).is_err());
    assert!(model.prefill(&[tokens[0], 999]).is_err());
    assert_eq!(model.position(), 0);
    close(&model.prefill(&tokens[..4]).unwrap(), &logits[3], 0.0002);
    assert_eq!(model.position(), 4);
    close(
        &model.prefill(&tokens[4..]).unwrap(),
        logits.last().unwrap(),
        0.0002,
    );
    model.reset().unwrap();
    model.accelerator_mut().reset_stats();
    assert_eq!(
        model.prefill_greedy(&tokens[..5]).unwrap(),
        rink_core::greedy(&logits[4]).unwrap()
    );
    assert_eq!(model.accelerator().stats().readback_bytes, 8);
    for i in 5..tokens.len() {
        assert_eq!(
            model.step_greedy(tokens[i]).unwrap(),
            rink_core::greedy(&logits[i]).unwrap()
        );
    }
    assert!(model.step_greedy(1).is_err());
}

#[test]
#[ignore = "requires a Metal GPU"]
fn gpu_argmax_handles_ties_tail_and_nonfinite_values() {
    let mut d = Metal::new().unwrap();
    let k = Kernels::compile(&mut d).unwrap();
    let mut values = vec![-5.; 2059];
    values[1023] = 9.;
    values[2048] = 9.;
    values[2058] = 8.;
    let partial = d.allocate(3 * 12).unwrap();
    let result = d.allocate(8).unwrap();
    for invalid in [false, true] {
        if invalid {
            values[2058] = f32::NAN;
        }
        let x = upload(&mut d, &values);
        // SAFETY: three 1024-token partitions, 12 bytes per partial and two u32 outputs.
        unsafe {
            k.run(
                &mut d,
                "argmax_partial",
                &[&partial, &x],
                Params::new(&[values.len()], &[]),
                96,
                32,
            )
            .unwrap();
            k.run(
                &mut d,
                "argmax_finish",
                &[&result, &partial],
                Params::new(&[3], &[]),
                32,
                32,
            )
            .unwrap();
        }
        let bytes = d.read(&result).unwrap();
        assert_eq!(u32::from_le_bytes(bytes[..4].try_into().unwrap()), 1023);
        assert_eq!(
            u32::from_le_bytes(bytes[4..].try_into().unwrap()),
            u32::from(invalid)
        );
    }
}

#[test]
#[ignore = "requires a Metal GPU"]
fn vectorized_dense_projection_handles_row_and_column_tiles() {
    let mut d = Metal::new().unwrap();
    let k = Kernels::compile(&mut d).unwrap();
    let (rows, cols) = (131, 132);
    let x: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.2).cos()).collect();
    let w: Vec<half::bf16> = (0..rows * cols)
        .map(|i| half::bf16::from_f32((i as f32 * 0.03).sin()))
        .collect();
    let weights = d
        .upload(
            &w.iter()
                .flat_map(|v| v.to_bits().to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let input = upload(&mut d, &x);
    let out = d.allocate(rows * 4).unwrap();
    // SAFETY: rows may be partial, columns are a multiple of four; guarded 4-row tiles.
    unsafe {
        k.run(
            &mut d,
            "matvec4",
            &[&out, &weights, &input],
            Params::new(&[rows, cols, 1], &[]),
            rows.div_ceil(4) * 32,
            128,
        )
        .unwrap();
    }
    let expected: Vec<f32> = w
        .chunks_exact(cols)
        .map(|row| row.iter().zip(&x).map(|(a, b)| a.to_f32() * b).sum())
        .collect();
    close(&read(&mut d, &out), &expected, 0.00005);
}

#[test]
#[ignore = "requires a Metal GPU"]
fn partitioned_attention_matches_scalar_with_tails_ring_and_causal_chunks() {
    let mut device = Metal::new().unwrap();
    let kernels = Kernels::compile(&mut device).unwrap();
    let (qh, kh, dim, rel_dim, extent) = (6, 2, 40, 4, 64);
    for (position, capacity, batch) in [(513, 257, 1), (126, 257, 5), (600, 257, 7)] {
        let vals = |n: usize, phase: f32| -> Vec<f32> {
            (0..n)
                .map(|i| (i as f32 * 0.017 + phase).sin() * 0.3)
                .collect()
        };
        let q = vals(batch * qh * dim, 0.1);
        let keys = vals(capacity * kh * dim, 0.2);
        let values = vals(capacity * kh * dim, 0.3);
        let new_keys = vals(batch * kh * dim, 0.8);
        let new_values = vals(batch * kh * dim, 0.9);
        let rel = vals(batch * qh * rel_dim, 0.4);
        let profiles = vals(rel_dim * extent, 0.5);
        let sinks = vals(batch * qh, 0.6);
        let gate = vals(batch * qh * dim, 0.7);
        let buffers: Vec<_> = [
            &q,
            &keys,
            &values,
            &new_keys,
            &new_values,
            &rel,
            &profiles,
            &sinks,
            &gate,
        ]
        .iter()
        .map(|v| upload(&mut device, v))
        .collect();
        let n = (position + batch).min(capacity);
        let parts = n.div_ceil(128);
        let partial = device.allocate(batch * qh * parts * (dim + 2) * 4).unwrap();
        let out = device.allocate(batch * qh * dim * 4).unwrap();
        let geometry = [
            qh, kh, dim, position, capacity, n, rel_dim, extent, parts, batch,
        ];
        // SAFETY: all heads/partitions and new-token arrays match the declared geometry.
        unsafe {
            kernels
                .run(
                    &mut device,
                    "attention_partition",
                    &[
                        &partial,
                        &buffers[0],
                        &buffers[1],
                        &buffers[2],
                        &buffers[3],
                        &buffers[4],
                        &buffers[5],
                        &buffers[6],
                    ],
                    Params::new(&geometry, &[]),
                    batch * qh * parts * 128,
                    128,
                )
                .unwrap();
            kernels
                .run(
                    &mut device,
                    "attention_finish",
                    &[&out, &partial, &buffers[7], &buffers[8]],
                    Params::new(&geometry, &[]),
                    batch * qh * 32,
                    32,
                )
                .unwrap();
        }
        let mut expected = Vec::new();
        for token in 0..batch {
            let pos = position + token;
            let visible = (pos + 1).min(capacity);
            for head in 0..qh {
                let query = token * qh + head;
                let kv_head = head / (qh / kh);
                let mut scores = Vec::new();
                for kp in pos + 1 - visible..=pos {
                    let key = if kp >= position {
                        &new_keys[((kp - position) * kh + kv_head) * dim..]
                    } else {
                        &keys[((kp % capacity) * kh + kv_head) * dim..]
                    };
                    let mut score: f32 = (0..dim).map(|i| q[query * dim + i] * key[i]).sum::<f32>()
                        / (dim as f32).sqrt();
                    if pos - kp < extent {
                        score += (0..rel_dim)
                            .map(|i| rel[query * rel_dim + i] * profiles[i * extent + pos - kp])
                            .sum::<f32>();
                    }
                    scores.push(score);
                }
                let maximum = scores.iter().copied().fold(sinks[query], f32::max);
                let weights: Vec<f32> = scores.iter().map(|s| (s - maximum).exp()).collect();
                let denominator = weights.iter().sum::<f32>() + (sinks[query] - maximum).exp();
                for i in 0..dim {
                    let mut sum = 0.;
                    for (j, kp) in (pos + 1 - visible..=pos).enumerate() {
                        let value = if kp >= position {
                            new_values[((kp - position) * kh + kv_head) * dim + i]
                        } else {
                            values[((kp % capacity) * kh + kv_head) * dim + i]
                        };
                        sum += weights[j] * value;
                    }
                    expected.push(sum / denominator / (1. + (-gate[query * dim + i]).exp()));
                }
            }
        }
        close(&read(&mut device, &out), &expected, 0.00001);
    }
}

#[test]
#[ignore = "requires a Metal GPU"]
fn matrix_prefill_matches_scalar_for_partial_tiles() {
    let mut device = Metal::new().unwrap();
    let kernels = Kernels::compile(&mut device).unwrap();
    let (batch, rows, cols) = (13, 19, 37);
    let x: Vec<f32> = (0..batch * cols).map(|i| (i as f32 * 0.2).cos()).collect();
    let w: Vec<half::bf16> = (0..rows * cols)
        .map(|i| half::bf16::from_f32((i as f32 * 0.03).sin()))
        .collect();
    let weights = device
        .upload(
            &w.iter()
                .flat_map(|v| v.to_bits().to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let input = upload(&mut device, &x);
    let output = device.allocate(batch * rows * 4).unwrap();
    // SAFETY: rounded 8x8 tiles guard both the output boundaries and the reduction tail.
    unsafe {
        kernels
            .run(
                &mut device,
                "matmul",
                &[&output, &weights, &input],
                Params::new(&[rows, cols, 1, batch, 0], &[]),
                rows.div_ceil(8) * batch.div_ceil(8) * 32,
                32,
            )
            .unwrap();
    }
    let mut expected = Vec::new();
    for token in 0..batch {
        for row in 0..rows {
            expected.push(
                (0..cols)
                    .map(|c| x[token * cols + c] * w[row * cols + c].to_f32())
                    .sum(),
            );
        }
    }
    close(&read(&mut device, &output), &expected, 0.00002);
}

#[test]
#[ignore = "requires a Metal GPU"]
fn batched_prefill_matches_token_steps_across_chunk_boundaries() {
    let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny");
    let temporary = std::env::temp_dir().join(format!(
        "rink-prefill-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir(&temporary).unwrap();
    let mut config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(fixture.join("config.json")).unwrap()).unwrap();
    config["text_config"]["model_max_length"] = 128.into();
    std::fs::write(
        temporary.join("config.json"),
        serde_json::to_vec(&config).unwrap(),
    )
    .unwrap();
    std::fs::copy(
        fixture.join("model.safetensors"),
        temporary.join("model.safetensors"),
    )
    .unwrap();
    let mut model = Inkling::load(Metal::new().unwrap(), &temporary, 96).unwrap();
    let tokens: Vec<u32> = (0..65).map(|i| (i * 7 % 14) as u32).collect();
    let reference: Vec<_> = tokens.iter().map(|&t| model.step(t).unwrap()).collect();
    for chunk_size in [2, 7, 31, 32, 33, 65] {
        model.reset().unwrap();
        let mut position = 0;
        for chunk in tokens.chunks(chunk_size) {
            position += chunk.len();
            close(
                &model.prefill(chunk).unwrap(),
                &reference[position - 1],
                0.0003,
            );
            assert_eq!(model.position(), position);
        }
        assert_eq!(model.step_greedy(3).unwrap(), {
            model.reset().unwrap();
            for &t in &tokens {
                model.step(t).unwrap();
            }
            rink_core::greedy(&model.step(3).unwrap()).unwrap()
        });
    }
    std::fs::remove_dir_all(temporary).unwrap();
}
