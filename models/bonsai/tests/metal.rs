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
                    "hadamard",
                    &[&xb, &sb, &rotated],
                    Params::new(&[block, 0], &[]),
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
    }
}
