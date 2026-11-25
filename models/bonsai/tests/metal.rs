#![cfg(target_os = "macos")]
use rink_bonsai::Bonsai;
use rink_core::{AutoregressiveModel, greedy};
use serde::Deserialize;
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny")
}
#[derive(Deserialize)]
struct Expected {
    tokens: Vec<u32>,
    logits: BTreeMap<usize, Vec<f32>>,
}

#[test]
#[ignore = "requires a Metal GPU"]
fn hybrid_forward_matches_independent_mlx_and_resets_both_caches() {
    let expected: Expected =
        serde_json::from_slice(&std::fs::read(fixture().join("expected.json")).unwrap()).unwrap();
    let mut model = Bonsai::load(rink_metal::Metal::new().unwrap(), fixture(), 512).unwrap();
    model.set_prefill_batch_limit(1).unwrap();
    let mut last = Vec::new();
    let mut first = Vec::new();
    let mut maximum = 0.0f32;
    for (i, &token) in expected.tokens.iter().enumerate() {
        let actual = model.step(token).unwrap();
        if i == 0 {
            first = actual.clone();
        }
        if let Some(want) = expected.logits.get(&(i + 1)) {
            assert_eq!(
                greedy(&actual).unwrap(),
                greedy(want).unwrap(),
                "greedy at position {}",
                i + 1
            );
            for (j, (&a, &b)) in actual.iter().zip(want).enumerate() {
                maximum = maximum.max((a - b).abs());
                assert!(
                    (a - b).abs() <= 0.01 + 0.005 * b.abs(),
                    "position {} logit {j}: {a} vs {b}",
                    i + 1
                );
            }
        }
        last = actual;
    }
    eprintln!("Bonsai tiny: maximum MLX logit error {maximum}");
    assert_eq!(model.position(), 300);
    assert!(model.step(256).is_err());
    assert_eq!(model.position(), 300);
    assert!(model.prefill(&vec![1; 213]).is_err());
    assert_eq!(model.position(), 300);
    model.reset().unwrap();
    assert_eq!(model.position(), 0);
    assert_eq!(model.step(expected.tokens[0]).unwrap(), first);
    model.reset().unwrap();
    assert_eq!(model.prefill(&expected.tokens).unwrap(), last);
    model.reset().unwrap();
    model.prefill(&expected.tokens[..127]).unwrap();
    assert_eq!(model.prefill(&expected.tokens[127..]).unwrap(), last);
}

#[test]
#[ignore = "requires a Metal GPU"]
fn batched_prefill_matches_mlx_and_preserves_cached_continuation() {
    let expected: Expected =
        serde_json::from_slice(&std::fs::read(fixture().join("expected.json")).unwrap()).unwrap();
    let check = |actual: &[f32], want: &[f32]| {
        assert_eq!(greedy(actual).unwrap(), greedy(want).unwrap());
        for (i, (&a, &b)) in actual.iter().zip(want).enumerate() {
            assert!(
                (a - b).abs() <= 0.01 + 0.005 * b.abs(),
                "logit {i}: {a} vs {b}"
            );
        }
    };
    let mut model = Bonsai::load(rink_metal::Metal::new().unwrap(), fixture(), 512).unwrap();
    for (&position, want) in &expected.logits {
        let start = model.position();
        check(
            &model.prefill(&expected.tokens[start..position]).unwrap(),
            want,
        );
    }
    let mut reference = Bonsai::load(rink_metal::Metal::new().unwrap(), fixture(), 512).unwrap();
    reference.set_prefill_batch_limit(1).unwrap();
    reference.prefill(&expected.tokens).unwrap();
    let continuation: Vec<_> = (0..10).map(|i| reference.step(i).unwrap()).collect();
    for batch in [2, 32, 64, 128, 256, 512] {
        model.reset().unwrap();
        model.set_prefill_batch_limit(batch).unwrap();
        let position = model.position();
        assert!(model.prefill(&[1, 256]).is_err());
        assert_eq!(model.position(), position);
        let last = model.prefill(&expected.tokens).unwrap();
        check(&last, &expected.logits[&300]);
        for (token, want) in continuation.iter().enumerate() {
            check(&model.step(token as u32).unwrap(), want);
        }
        model.reset().unwrap();
        model.prefill(&expected.tokens[..127]).unwrap();
        check(
            &model.prefill(&expected.tokens[127..]).unwrap(),
            &expected.logits[&300],
        );
    }
}

#[test]
#[ignore = "requires a Metal GPU"]
fn packed_projection_and_inverse_embedding_match_scalar_reference() {
    use half::f16;
    use rink_bonsai::kernels::{Kernels, Params};
    use rink_core::Accelerator;
    fn bits(values: &[f32]) -> Vec<u16> {
        values.iter().map(|&v| f16::from_f32(v).to_bits()).collect()
    }
    fn read(device: &mut rink_metal::Metal, b: &rink_metal::MetalBuffer) -> Vec<f32> {
        device
            .read(b)
            .unwrap()
            .chunks_exact(2)
            .map(|b| f16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).to_f32())
            .collect()
    }
    fn transform(x: &[f32], signs: &[f32], block: usize, inverse: bool) -> Vec<f32> {
        let mut out = Vec::new();
        for (x, signs) in x.chunks(block).zip(signs.chunks(block)) {
            let mut v: Vec<f64> = x
                .iter()
                .zip(signs)
                .map(|(&x, &s)| x as f64 * if inverse { 1.0 } else { s as f64 })
                .collect();
            let mut stride = 1;
            while stride < block {
                for base in (0..block).step_by(2 * stride) {
                    for j in 0..stride {
                        let a = v[base + j];
                        let b = v[base + j + stride];
                        v[base + j] = a + b;
                        v[base + j + stride] = a - b;
                    }
                }
                stride *= 2;
            }
            out.extend(v.into_iter().zip(signs).map(|(v, &s)| {
                f16::from_f64(v / (block as f64).sqrt() * if inverse { s as f64 } else { 1.0 })
                    .to_f32()
            }));
        }
        out
    }
    let mut device = rink_metal::Metal::new().unwrap();
    let kernels = Kernels::compile(&mut device).unwrap();
    for (rows, width, block) in [
        (1, 128, 128),
        (5, 384, 128),
        (17, 5120, 1024),
        (5, 8192, 4096),
        (5, 17408, 1024),
    ] {
        let embedding_row = (rows - 1).min(3);
        let x: Vec<f32> = (0..width)
            .map(|i| f16::from_f32((i % 29) as f32 / 16.0 - 0.75).to_f32())
            .collect();
        let signs: Vec<f32> = (0..width)
            .map(|i| if i % 3 == 0 { -1.0 } else { 1.0 })
            .collect();
        let mut words = vec![0u32; rows * width / 16];
        for i in 0..rows * width {
            words[i / 16] |= (((i % 7) % 4) as u32) << (2 * (i % 16));
        }
        let scales: Vec<f32> = (0..rows * width / 128)
            .map(|i| f16::from_f32((i % 11 + 1) as f32 / 257.0).to_f32())
            .collect();
        // Exercise all four codes and independently rounded affine biases, not
        // just the symmetric ternary subset used by the native checkpoint.
        let biases: Vec<f32> = scales
            .iter()
            .enumerate()
            .map(|(i, &s)| f16::from_f32(-0.73 * s + (i % 3) as f32 / 997.0).to_f32())
            .collect();
        let xb = device.upload(bytemuck::cast_slice(&bits(&x))).unwrap();
        let sb = device.upload(bytemuck::cast_slice(&bits(&signs))).unwrap();
        let w = device.upload(bytemuck::cast_slice(&words)).unwrap();
        let scalesb = device.upload(bytemuck::cast_slice(&bits(&scales))).unwrap();
        let biasesb = device.upload(bytemuck::cast_slice(&bits(&biases))).unwrap();
        let rotated = device.allocate(width * 2).unwrap();
        let projected = device
            .upload(bytemuck::cast_slice(&vec![0x7e00u16; rows + 64]))
            .unwrap();
        let embedding = device.allocate(width * 2).unwrap();
        let inverse = device.allocate(width * 2).unwrap();
        // SAFETY: all shapes, packed extents and block counts match the shader ABI.
        unsafe {
            kernels
                .run(
                    &mut device,
                    if block == 1024 {
                        "hadamard1024"
                    } else {
                        "hadamard"
                    },
                    &[&xb, &sb, &rotated],
                    Params::new(&[block, 0, width], &[]),
                    width / block * 256,
                    256,
                )
                .unwrap();
            kernels
                .run(
                    &mut device,
                    "packed_mv",
                    &[&w, &scalesb, &biasesb, &rotated, &projected],
                    Params::new(&[width, rows], &[]),
                    rows * 32,
                    128,
                )
                .unwrap();
            kernels
                .run(
                    &mut device,
                    "embedding",
                    &[&w, &scalesb, &biasesb, &embedding],
                    Params::new(&[width, embedding_row], &[]),
                    width,
                    256,
                )
                .unwrap();
            kernels
                .run(
                    &mut device,
                    "hadamard",
                    &[&embedding, &sb, &inverse],
                    Params::new(&[block, 1], &[]),
                    width / block * 256,
                    256,
                )
                .unwrap();
        }
        let transformed = transform(&x, &signs, block, false);
        for (a, b) in read(&mut device, &rotated).iter().zip(&transformed) {
            assert!((a - b).abs() < 0.002);
        }
        let weight = |i: usize| {
            f16::from_f32(
                ((words[i / 16] >> (2 * (i % 16))) & 3) as f32 * scales[i / 128] + biases[i / 128],
            )
            .to_f32()
        };
        let output = device.read(&projected).unwrap();
        assert!(output[rows * 2..].chunks_exact(2).all(|b| b == [0, 0x7e]));
        for (row, actual) in read(&mut device, &projected)
            .into_iter()
            .take(rows)
            .enumerate()
        {
            let expected = f16::from_f64(
                (0..width)
                    .map(|i| weight(row * width + i) as f64 * transformed[i] as f64)
                    .sum(),
            )
            .to_f32();
            assert!(
                (actual - expected).abs() <= 0.002 + 0.001 * expected.abs(),
                "width {width} row {row}: {actual} vs {expected}"
            );
        }
        let embedded: Vec<f32> = (0..width)
            .map(|i| weight(embedding_row * width + i))
            .collect();
        for (actual, expected) in read(&mut device, &inverse)
            .iter()
            .zip(transform(&embedded, &signs, block, true))
        {
            assert!((actual - expected).abs() < 0.002);
        }
        // The symmetric path factors scales out of the dot product. Check it
        // independently, including code 3, negative scales and a partial row pair.
        let symmetric_scales: Vec<f32> = scales
            .iter()
            .enumerate()
            .map(|(i, &s)| if i % 3 == 0 { -s } else { s })
            .collect();
        let symmetric_biases: Vec<f32> = symmetric_scales.iter().map(|s| -s).collect();
        let ss = device
            .upload(bytemuck::cast_slice(&bits(&symmetric_scales)))
            .unwrap();
        let sb = device
            .upload(bytemuck::cast_slice(&bits(&symmetric_biases)))
            .unwrap();
        let out = device
            .upload(bytemuck::cast_slice(&vec![0x7e00u16; rows + 64]))
            .unwrap();
        // SAFETY: the affine metadata satisfies the symmetric-path contract;
        // the output has one entry per row, followed by an untouched guard.
        unsafe {
            kernels
                .run(
                    &mut device,
                    "packed_mv_symmetric",
                    &[&w, &ss, &sb, &rotated, &out],
                    Params::new(&[width, rows], &[]),
                    rows.div_ceil(2) * 32,
                    128,
                )
                .unwrap();
        }
        let bytes = device.read(&out).unwrap();
        assert!(bytes[rows * 2..].chunks_exact(2).all(|b| b == [0, 0x7e]));
        for (row, actual) in read(&mut device, &out).into_iter().take(rows).enumerate() {
            let expected = f16::from_f64(
                (0..width)
                    .map(|i| {
                        let j = row * width + i;
                        let code = ((words[j / 16] >> (2 * (j % 16))) & 3) as f32;
                        let weight = f16::from_f32(
                            code * symmetric_scales[j / 128] + symmetric_biases[j / 128],
                        );
                        weight.to_f64() * transformed[i] as f64
                    })
                    .sum(),
            )
            .to_f32();
            assert!(
                (actual - expected).abs() <= 0.002 + 0.001 * expected.abs(),
                "symmetric width {width} row {row}: {actual} vs {expected}"
            );
        }
    }
}

#[test]
#[ignore = "requires a Metal GPU"]
fn matrix_tiles_match_scalar_products_and_guard_partial_tiles() {
    use half::f16;
    use rink_bonsai::kernels::{Kernels, Params};
    use rink_core::Accelerator;
    fn bits(v: &[f16]) -> Vec<u16> {
        v.iter().map(|x| x.to_bits()).collect()
    }
    let mut device = rink_metal::Metal::new().unwrap();
    let kernels = Kernels::compile(&mut device).unwrap();
    for (batch, rows, cols) in [
        (2, 3, 128),
        (31, 65, 384),
        (33, 48, 5120),
        (65, 3, 17408),
        (128, 65, 128),
    ] {
        let words: Vec<u32> = (0..rows * cols / 16)
            .map(|i| (i as u32).wrapping_mul(0x9e3779b9).wrapping_add(0x12345678))
            .collect();
        let scales: Vec<f16> = (0..rows * cols / 128)
            .map(|i| f16::from_f32(0.003 + (i % 31) as f32 * 0.0002))
            .collect();
        let biases: Vec<f16> = scales
            .iter()
            .enumerate()
            .map(|(i, s)| f16::from_f32(-s.to_f32() * (0.8 + (i % 7) as f32 * 0.03)))
            .collect();
        let x: Vec<f16> = (0..batch * cols)
            .map(|i| f16::from_f32(((i * 17 % 113) as f32 - 56.) / 37.))
            .collect();
        let w: Vec<f16> = (0..rows * cols)
            .map(|i| {
                f16::from_f32(
                    ((words[i / 16] >> (2 * (i % 16))) & 3) as f32 * scales[i / 128].to_f32()
                        + biases[i / 128].to_f32(),
                )
            })
            .collect();
        let packed = device.upload(bytemuck::cast_slice(&words)).unwrap();
        let dense = device.upload(bytemuck::cast_slice(&bits(&w))).unwrap();
        let sb = device.upload(bytemuck::cast_slice(&bits(&scales))).unwrap();
        let bb = device.upload(bytemuck::cast_slice(&bits(&biases))).unwrap();
        let xb = device.upload(bytemuck::cast_slice(&bits(&x))).unwrap();
        let expected: Vec<f32> = (0..batch * rows)
            .map(|i| {
                f16::from_f64(
                    (0..cols)
                        .map(|k| {
                            x[(i / rows) * cols + k].to_f64() * w[(i % rows) * cols + k].to_f64()
                        })
                        .sum(),
                )
                .to_f32()
            })
            .collect();
        for name in ["packed_mm32", "packed_mm64", "dense_mm"] {
            let out = device
                .upload(bytemuck::cast_slice(&vec![0x7e00u16; batch * rows + 64]))
                .unwrap();
            let buffers = if name == "dense_mm" {
                vec![&dense, &xb, &out]
            } else {
                vec![&packed, &sb, &bb, &xb, &out]
            };
            let tile = if name == "packed_mm32" { 32 } else { 64 };
            // SAFETY: packed/dense extents, bounded row/token tails, and tile geometry match the kernel ABI.
            unsafe {
                kernels
                    .run(
                        &mut device,
                        name,
                        &buffers,
                        Params::new(&[cols, rows, batch], &[]),
                        rows.div_ceil(64) * batch.div_ceil(tile) * 128,
                        128,
                    )
                    .unwrap();
            }
            let bytes = device.read(&out).unwrap();
            assert!(
                bytes[batch * rows * 2..]
                    .chunks_exact(2)
                    .all(|b| b == [0, 0x7e]),
                "{name} output guard"
            );
            for (i, (bytes, &want)) in bytes[..batch * rows * 2]
                .chunks_exact(2)
                .zip(&expected)
                .enumerate()
            {
                let actual = f16::from_bits(u16::from_le_bytes(bytes.try_into().unwrap())).to_f32();
                assert!(
                    (actual - want).abs() <= 0.002 + 0.001 * want.abs(),
                    "{name} [{batch},{rows},{cols}] output {i}: {actual} vs {want}"
                );
            }
        }
    }
}

#[test]
#[ignore = "requires a Metal GPU"]
fn device_greedy_preserves_ties_rejects_nonfinite_and_matches_cached_steps() {
    use half::f16;
    use rink_bonsai::kernels::{Kernels, Params};
    use rink_core::Accelerator;
    let mut device = rink_metal::Metal::new().unwrap();
    let kernels = Kernels::compile(&mut device).unwrap();
    for n in [1usize, 31, 1024, 1025, 248320] {
        for bad in [
            None,
            Some(f32::NAN),
            Some(f32::INFINITY),
            Some(f32::NEG_INFINITY),
        ] {
            let mut values = vec![-1.0; n];
            values[n / 2] = 2.;
            values[n - 1] = 2.;
            if let Some(bad) = bad {
                values[0] = bad;
            }
            let bits: Vec<u16> = values.iter().map(|&x| f16::from_f32(x).to_bits()).collect();
            let logits = device.upload(bytemuck::cast_slice(&bits)).unwrap();
            let parts = n.div_ceil(1024);
            let partial = device.allocate(parts * 8).unwrap();
            let token = device.allocate(4).unwrap();
            // SAFETY: exact vocabulary prefix, one 8-byte choice per partition, and a 4-byte result.
            unsafe {
                kernels
                    .run(
                        &mut device,
                        "greedy_part",
                        &[&logits, &partial],
                        Params::new(&[n], &[]),
                        parts * 256,
                        256,
                    )
                    .unwrap();
                kernels
                    .run(
                        &mut device,
                        "greedy_finish",
                        &[&partial, &token],
                        Params::new(&[parts], &[]),
                        256,
                        256,
                    )
                    .unwrap();
            }
            let mut bytes = [0u8; 4];
            device.read_into(&token, &mut bytes).unwrap();
            let actual = u32::from_le_bytes(bytes);
            if bad.is_some() {
                assert_eq!(actual, u32::MAX);
            } else {
                assert_eq!(actual, greedy(&values).unwrap());
            }
        }
    }
    let expected: Expected =
        serde_json::from_slice(&std::fs::read(fixture().join("expected.json")).unwrap()).unwrap();
    let mut reference = Bonsai::load(rink_metal::Metal::new().unwrap(), fixture(), 512).unwrap();
    let mut fast = Bonsai::load(rink_metal::Metal::new().unwrap(), fixture(), 512).unwrap();
    assert_eq!(
        fast.prefill_greedy(&expected.tokens).unwrap(),
        greedy(&reference.prefill(&expected.tokens).unwrap()).unwrap()
    );
    for token in 0..16 {
        assert_eq!(
            fast.step_greedy(token).unwrap(),
            greedy(&reference.step(token).unwrap()).unwrap()
        );
    }
    fast.reset().unwrap();
    reference.reset().unwrap();
    assert_eq!(
        fast.prefill_greedy(&expected.tokens).unwrap(),
        greedy(&reference.prefill(&expected.tokens).unwrap()).unwrap()
    );
}

#[test]
#[ignore = "requires a Metal GPU"]
fn attention_partitions_match_scalar_softmax_across_context_boundaries() {
    use half::f16;
    use rink_bonsai::kernels::{Kernels, Params};
    use rink_core::Accelerator;
    let mut device = rink_metal::Metal::new().unwrap();
    let kernels = Kernels::compile(&mut device).unwrap();
    let heads = 4usize;
    let width = 128usize;
    let batch = 4usize;
    let q: Vec<f16> = (0..batch * heads * width)
        .map(|i| f16::from_f32(((i * 31 % 97) as f32 - 48.) / 19.))
        .collect();
    let gates: Vec<f16> = (0..batch * heads * 2 * width)
        .map(|i| f16::from_f32(((i * 13 % 71) as f32 - 35.) / 32.))
        .collect();
    let qb = device
        .upload(bytemuck::cast_slice(
            &q.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        ))
        .unwrap();
    let gb = device
        .upload(bytemuck::cast_slice(
            &gates.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        ))
        .unwrap();
    for position in [254usize, 4094, 8190] {
        let length = position + batch;
        let keys: Vec<f16> = (0..length * width)
            .map(|i| f16::from_f32(((i * 17 % 113) as f32 - 56.) / 43.))
            .collect();
        let values: Vec<f16> = (0..length * width)
            .map(|i| f16::from_f32(((i * 29 % 149) as f32 - 74.) / 67.))
            .collect();
        let kb = device
            .upload(bytemuck::cast_slice(
                &keys.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            ))
            .unwrap();
        let vb = device
            .upload(bytemuck::cast_slice(
                &values.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            ))
            .unwrap();
        let mut expected = Vec::new();
        for token in 1..batch {
            for head in 0..heads {
                let scores: Vec<f64> = (0..position + token + 1)
                    .map(|t| {
                        (0..width)
                            .map(|j| {
                                q[(token * heads + head) * width + j].to_f64()
                                    * keys[t * width + j].to_f64()
                            })
                            .sum::<f64>()
                            / (width as f64).sqrt()
                    })
                    .collect();
                let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let probs: Vec<f64> = scores.iter().map(|s| (s - maximum).exp()).collect();
                let den: f64 = probs.iter().sum();
                for j in 0..width {
                    let value = f16::from_f64(
                        probs
                            .iter()
                            .enumerate()
                            .map(|(t, p)| p * values[t * width + j].to_f64())
                            .sum::<f64>()
                            / den,
                    );
                    let gate = gates[(token * heads + head) * 2 * width + width + j].to_f32();
                    let gate = f16::from_f32(1. / (1. + (-gate).exp()));
                    expected.push(f16::from_f32(value.to_f32() * gate.to_f32()).to_f32());
                }
            }
        }
        for partition in [32usize, 256] {
            let count = batch - 1;
            let parts = length.div_ceil(partition);
            let partial = device
                .allocate(count * heads * parts * (width + 2) * 4)
                .unwrap();
            let out = device
                .upload(bytemuck::cast_slice(&vec![
                    0x7e00u16;
                    batch * heads * width + 64
                ]))
                .unwrap();
            // SAFETY: the dispatched queries are [1,4), caches contain the complete
            // causal prefix, and all partitions have distinct scratch ranges.
            unsafe {
                kernels
                    .run(
                        &mut device,
                        "attention_part",
                        &[&qb, &kb, &vb, &partial],
                        Params::new(
                            &[parts, width, heads, 1, position + 1, 1, count, partition],
                            &[(width as f32).sqrt().recip()],
                        ),
                        count * parts * heads * 128,
                        128,
                    )
                    .unwrap();
                kernels
                    .run(
                        &mut device,
                        "attention_finish",
                        &[&partial, &gb, &out],
                        Params::new(&[parts, width, heads, 1, count], &[]),
                        count * heads * width,
                        256,
                    )
                    .unwrap();
            }
            let bytes = device.read(&out).unwrap();
            for guard in [
                &bytes[..heads * width * 2],
                &bytes[batch * heads * width * 2..],
            ] {
                assert!(guard.chunks_exact(2).all(|b| b == [0, 0x7e]));
            }
            for (i, (bytes, &want)) in bytes[heads * width * 2..batch * heads * width * 2]
                .chunks_exact(2)
                .zip(&expected)
                .enumerate()
            {
                let actual = f16::from_bits(u16::from_le_bytes(bytes.try_into().unwrap())).to_f32();
                assert!(
                    (actual - want).abs() <= 0.002 + 0.001 * want.abs(),
                    "position {position}, partition {partition}, output {i}: {actual} vs {want}"
                );
            }
        }
    }
}
