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
            Params::new(&[1, 32, 2, 0, 0], &[]),
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
            Params::new(&[1, 32, 2, 1, 0], &[]),
            64,
            32,
        )
        .unwrap();
    }
    let values = [0., 1. / 512., 1., -1.5, 448., -0., 0.03125, -128.];
    let base: f32 = x.iter().enumerate().map(|(i, x)| x * values[i % 8]).sum();
    close(&read(&mut d, &out), &[base * 4., base * 0.5], 0.002);
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
            Params::new(&[3, 2], &[8.]),
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
