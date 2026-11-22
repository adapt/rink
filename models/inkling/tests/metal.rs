#![cfg(target_os = "macos")]
use rink_core::{Accelerator, AutoregressiveModel};
use rink_inkling::{
    Inkling, KvCachePrecision,
    kernels::{Kernels, Params},
};
use rink_metal::{Metal, MetalBuffer};

fn upload(d: &mut Metal, values: &[f32]) -> MetalBuffer {
    d.upload(bytemuck::cast_slice(values)).unwrap()
}
fn upload_cache(d: &mut Metal, values: &[f32], fp16: bool) -> MetalBuffer {
    if fp16 {
        d.upload(
            &values
                .iter()
                .flat_map(|&v| half::f16::from_f32(v).to_bits().to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap()
    } else {
        upload(d, values)
    }
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

#[test]
#[ignore = "requires a Metal GPU"]
fn fused_residual_convolution_preserves_mixtures_history_chunks_and_reset() {
    let mut d = Metal::new().unwrap();
    let k = Kernels::compile(&mut d).unwrap();
    for channels in [1usize, 17, 129] {
        for taps in [1usize, 4, 7] {
            for top in [0usize, 1, 3] {
                for add in [false, true] {
                    let values = |n, phase: usize| -> Vec<f32> {
                        (0..n)
                            .map(|i| ((i * 13 + phase) % 37) as f32 / 41. - 0.4)
                            .collect()
                    };
                    let weights = upload(&mut d, &values(channels * taps, 5));
                    let initial = values(channels * taps, 11);
                    let old_state = upload(&mut d, &initial);
                    let new_state = upload(&mut d, &initial);
                    let mut position = 0;
                    // A chunk longer than the ring, multiple continuations, then
                    // a reset without clearing either state allocation.
                    for (batch, reset) in
                        [(1, false), (2, false), (9, false), (3, false), (5, true)]
                    {
                        if reset {
                            position = 0;
                        }
                        let n = channels * batch;
                        let input = upload(&mut d, &values(n * top.max(1), position));
                        let residual = upload(&mut d, &values(n, position + 7));
                        let routing = upload(
                            &mut d,
                            &(0..batch * top.max(1))
                                .flat_map(|i| [0., (i % 5) as f32 / 7.])
                                .collect::<Vec<_>>(),
                        );
                        let mixture = d.allocate(n * 4).unwrap();
                        let convolved = d.allocate(n * 4).unwrap();
                        let old = d.allocate(n * 4).unwrap();
                        let new = upload(&mut d, &vec![f32::NAN; n]);
                        // SAFETY: each channel owns its ring slots. Input and
                        // routing lengths match top-k; unused buffers are valid
                        // placeholders. Both states start with identical history.
                        unsafe {
                            if top > 0 {
                                k.run(
                                    &mut d,
                                    "expert_sum",
                                    &[&mixture, &input, &routing],
                                    Params::new(&[channels, top, batch], &[]),
                                    n,
                                    128,
                                )
                                .unwrap();
                            }
                            k.run(
                                &mut d,
                                "convolution",
                                &[
                                    &convolved,
                                    if top > 0 { &mixture } else { &input },
                                    &weights,
                                    &old_state,
                                    &input,
                                ],
                                Params::new(&[channels, taps, position, 0, channels, batch], &[]),
                                channels,
                                128,
                            )
                            .unwrap();
                            if add {
                                k.run(
                                    &mut d,
                                    "add",
                                    &[&old, &residual, &convolved],
                                    Params::new(&[n], &[]),
                                    n,
                                    128,
                                )
                                .unwrap();
                            }
                            k.run(
                                &mut d,
                                "residual_convolution",
                                &[&new, &input, &weights, &new_state, &residual, &routing],
                                Params::new(
                                    &[channels, taps, position, batch, top, usize::from(add)],
                                    &[],
                                ),
                                channels,
                                128,
                            )
                            .unwrap();
                        }
                        close(
                            &read(&mut d, &new),
                            &read(&mut d, if add { &old } else { &convolved }),
                            0.000003,
                        );
                        close(
                            &read(&mut d, &new_state),
                            &read(&mut d, &old_state),
                            0.000001,
                        );
                        position += batch;
                    }
                }
            }
        }
    }
}

#[test]
#[ignore = "requires a Metal GPU"]
fn parallel_routing_keeps_nonfinite_candidates_in_bounds() {
    let mut d = Metal::new().unwrap();
    let k = Kernels::compile(&mut d).unwrap();
    for nan_logits in [false, true] {
        let logits = upload(&mut d, &[if nan_logits { f32::NAN } else { 0. }; 33]);
        let bias = upload(
            &mut d,
            &[if nan_logits { 0. } else { f32::NEG_INFINITY }; 33],
        );
        let out = d.allocate(32 * 8).unwrap();
        // SAFETY: one row of 33 candidates and 32 selected output pairs.
        unsafe {
            k.run(
                &mut d,
                "route_parallel",
                &[&out, &logits, &bias],
                Params::new(&[33, 32, 1], &[8.]),
                32,
                32,
            )
            .unwrap();
        }
        for (index, pair) in read(&mut d, &out).chunks_exact(2).enumerate() {
            assert_eq!(pair[0], index as f32);
            if nan_logits {
                assert!(pair[1].is_nan());
            } else {
                assert_eq!(pair[1], 0.25);
            }
        }
    }
}

#[test]
#[ignore = "requires a Metal GPU"]
fn grouped_and_fused_experts_match_vectors_with_sparse_routes_and_partial_tiles() {
    let mut d = Metal::new().unwrap();
    let k = Kernels::compile(&mut d).unwrap();
    let experts = 37usize;
    let top = 3usize;
    for (format, cols, rows) in [
        (0usize, 16usize, 11usize),
        (0, 32, 11),
        (0, 160, 11),
        (1, 32, 11),
        (1, 160, 11),
        (2, 17, 11),
        (2, 160, 11),
        (0, 16, 2),
        (0, 32, 14),
        (0, 160, 22),
        (1, 32, 14),
        (1, 160, 22),
        (2, 17, 22),
        (2, 160, 14),
        (0, 160, 37),
        (1, 160, 65),
        (2, 37, 65),
        (0, 2304, 32),
        (1, 1152, 32),
    ] {
        let count = experts * rows * cols;
        let weights = match format {
            0 => d
                .upload(
                    &(0..count / 2)
                        .map(|i| ((i * 7 % 16) | ((i * 11 % 16) << 4)) as u8)
                        .collect::<Vec<_>>(),
                )
                .unwrap(),
            1 => d
                .upload(
                    &(0..count)
                        .map(|i| [0x00, 0x01, 0x38, 0xbc, 0x10, 0xb0, 0x3c, 0x40][i % 8])
                        .collect::<Vec<_>>(),
                )
                .unwrap(),
            _ => upload(
                &mut d,
                &(0..count)
                    .map(|i| ((i * 17 % 43) as f32 - 21.) / 29.)
                    .collect::<Vec<_>>(),
            ),
        };
        let scale_values: Vec<u8> = match format {
            0 => (0..count / 16).map(|i| [0x30, 0x38, 0x40][i % 3]).collect(),
            1 => (0..count / 32).map(|i| [126, 127, 128][i % 3]).collect(),
            _ => vec![0; 4],
        };
        let scales = d.upload(&scale_values).unwrap();
        let factors = upload(
            &mut d,
            &(0..experts)
                .map(|i| (i % 7 + 1) as f32 / 8.)
                .collect::<Vec<_>>(),
        );
        let table = d.allocate(experts * 4096 * 4).unwrap();
        if format == 0 {
            // SAFETY: the table has 4096 entries for each uploaded factor.
            unsafe {
                k.run(
                    &mut d,
                    "nvfp4_decode_table",
                    &[&table, &factors],
                    Params::new(&[experts], &[]),
                    experts * 4096,
                    128,
                )
                .unwrap();
            }
        }
        for batch in [1usize, 2, 8, 17, 32, 33, 64, 128] {
            for sparse in [false, true] {
                let routes: Vec<_> = (0..batch * top)
                    .flat_map(|slot| {
                        let e = if sparse {
                            [0, 5, experts - 1][slot % top]
                        } else {
                            (slot / top * 7 + slot % top * 11) % experts
                        };
                        [e as f32, 1. / top as f32]
                    })
                    .collect();
                let routing = upload(&mut d, &routes);
                let slots = batch * top;
                let max_tiles = (slots + 7 * experts.min(slots)) / 8;
                let groups = d.upload(&vec![0xff; experts * (batch + 1) * 4]).unwrap();
                let tiles = d.upload(&vec![0xff; (1 + 2 * max_tiles) * 4]).unwrap();
                // SAFETY: each token selects distinct valid experts, group lists
                // have room for every token, and the tile list uses a proven bound.
                unsafe {
                    k.run(
                        &mut d,
                        "expert_groups",
                        &[&groups, &routing],
                        Params::new(&[experts, top, batch], &[]),
                        experts * 32,
                        32,
                    )
                    .unwrap();
                    k.run(
                        &mut d,
                        "expert_tiles",
                        &[&tiles, &groups],
                        Params::new(&[experts, batch], &[]),
                        32,
                        32,
                    )
                    .unwrap();
                }
                let ints = |bytes: Vec<u8>| {
                    bytes
                        .chunks_exact(4)
                        .map(|v| u32::from_le_bytes(v.try_into().unwrap()))
                        .collect::<Vec<_>>()
                };
                let actual_groups = ints(d.read(&groups).unwrap());
                let actual_tiles = ints(d.read(&tiles).unwrap());
                let mut expected_tiles = Vec::new();
                for e in 0..experts {
                    let expected: Vec<_> = (0..slots)
                        .filter(|&slot| routes[2 * slot] as usize == e)
                        .map(|slot| slot as u32)
                        .collect();
                    let base = e * (batch + 1);
                    assert_eq!(actual_groups[base] as usize, expected.len());
                    assert_eq!(
                        &actual_groups[base + 1..base + 1 + expected.len()],
                        &expected
                    );
                    for first in (0..expected.len()).step_by(8) {
                        expected_tiles.extend([e as u32, first as u32]);
                    }
                }
                assert_eq!(actual_tiles[0] as usize, expected_tiles.len() / 2);
                assert_eq!(&actual_tiles[1..1 + expected_tiles.len()], &expected_tiles);
                for down in [0, 1] {
                    let input = upload(
                        &mut d,
                        &(0..if down == 1 {
                            slots * cols
                        } else {
                            batch * cols
                        })
                            .map(|i| ((i * 13 % 41) as f32 - 20.) / 31.)
                            .collect::<Vec<_>>(),
                    );
                    let old = d.allocate(slots * rows * 4).unwrap();
                    let new = upload(&mut d, &vec![f32::NAN; slots * rows]);
                    let tiled = upload(&mut d, &vec![f32::NAN; slots * rows + 17]);
                    let p = Params::new(&[rows, cols, top, format, down, batch], &[]);
                    // SAFETY: native weights and scales use validated block widths;
                    // inputs match shared/per-slot mode and outputs cover every slot.
                    unsafe {
                        k.run(
                            &mut d,
                            "expert_matvec",
                            &[&old, &weights, &scales, &factors, &input, &routing],
                            p,
                            slots * rows * 32,
                            128,
                        )
                        .unwrap();
                        k.run(
                            &mut d,
                            "expert_matmul",
                            &[&new, &weights, &scales, &factors, &input, &groups, &tiles],
                            p,
                            rows.div_ceil(8) * max_tiles * 32,
                            32,
                        )
                        .unwrap();
                        k.run(
                            &mut d,
                            "expert_matmul_tiled",
                            &[&tiled, &weights, &scales, &factors, &input, &groups, &tiles],
                            p,
                            rows.div_ceil(32) * max_tiles * 128,
                            128,
                        )
                        .unwrap();
                    }
                    if format == 0 {
                        let lookup = upload(&mut d, &vec![f32::NAN; slots * rows + 17]);
                        // SAFETY: same validated geometry as the reference, plus its complete table.
                        unsafe {
                            k.run(
                                &mut d,
                                "expert_matvec_nvfp4",
                                &[
                                    &lookup, &weights, &scales, &factors, &input, &routing, &table,
                                ],
                                p,
                                slots * rows * 32,
                                128,
                            )
                            .unwrap();
                        }
                        let bytes = d.read(&lookup).unwrap();
                        assert_eq!(&bytes[..slots * rows * 4], d.read(&old).unwrap());
                        assert!(
                            read(&mut d, &lookup)[slots * rows..]
                                .iter()
                                .all(|v| v.to_bits() == f32::NAN.to_bits())
                        );
                        // SAFETY: grouped NVFP4 has the same tiles and output
                        // guards as the reference, and reuses its exact table.
                        unsafe {
                            k.run(
                                &mut d,
                                "expert_matmul_nvfp4",
                                &[
                                    &lookup, &weights, &scales, &factors, &input, &groups, &tiles,
                                    &table,
                                ],
                                p,
                                rows.div_ceil(32) * max_tiles * 128,
                                128,
                            )
                            .unwrap();
                        }
                        assert_eq!(
                            d.read(&lookup).unwrap(),
                            d.read(&tiled).unwrap(),
                            "NVFP4 grouped: rows={rows} cols={cols} batch={batch} sparse={sparse} down={down}"
                        );
                    }
                    let tiled_values = read(&mut d, &tiled);
                    let reference = read(&mut d, &new);
                    for (i, (&a, &b)) in tiled_values.iter().zip(&reference).enumerate() {
                        assert_eq!(
                            a.to_bits(),
                            b.to_bits(),
                            "format={format} rows={rows} cols={cols} batch={batch} sparse={sparse} down={down} index={i}"
                        );
                        assert!(a.is_finite());
                    }
                    assert!(
                        tiled_values[slots * rows..]
                            .iter()
                            .all(|v| v.to_bits() == f32::NAN.to_bits())
                    );
                    close(&read(&mut d, &new), &read(&mut d, &old), 0.0002);
                    if rows % 2 == 0 && down == 0 {
                        let width = rows / 2;
                        let caps = [1.7, 2.3];
                        let expected_vector = d.allocate(slots * width * 4).unwrap();
                        let fused_vector = upload(&mut d, &vec![f32::NAN; slots * width]);
                        let activated = Params::new(&[width, top, batch], &caps);
                        let fused = Params::new(&[rows, cols, top, format, 0, batch], &caps);
                        // SAFETY: paired gate/up rows are even. The fused output
                        // has half the row count and retains original routing slots.
                        unsafe {
                            k.run(
                                &mut d,
                                "swiglu",
                                &[&expected_vector, &old],
                                activated,
                                slots * width,
                                128,
                            )
                            .unwrap();
                            k.run(
                                &mut d,
                                "expert_swiglu_matvec",
                                &[&fused_vector, &weights, &scales, &factors, &input, &routing],
                                fused,
                                slots * width * 32,
                                128,
                            )
                            .unwrap();
                        }
                        if format == 0 {
                            let lookup = upload(&mut d, &vec![f32::NAN; slots * width + 17]);
                            // SAFETY: paired rows use the same complete matrix table and routes.
                            unsafe {
                                k.run(
                                    &mut d,
                                    "expert_swiglu_nvfp4",
                                    &[
                                        &lookup, &weights, &scales, &factors, &input, &routing,
                                        &table,
                                    ],
                                    fused,
                                    slots * width * 32,
                                    128,
                                )
                                .unwrap();
                            }
                            let bytes = d.read(&lookup).unwrap();
                            assert_eq!(&bytes[..slots * width * 4], d.read(&fused_vector).unwrap());
                            assert!(
                                read(&mut d, &lookup)[slots * width..]
                                    .iter()
                                    .all(|v| v.to_bits() == f32::NAN.to_bits())
                            );
                        }
                        close(
                            &read(&mut d, &fused_vector),
                            &read(&mut d, &expected_vector),
                            0.000003,
                        );
                    }
                }
            }
        }
    }
}

#[test]
#[ignore = "requires a Metal GPU"]
fn fused_controllers_match_separate_projections_for_dtypes_and_tiles() {
    let mut d = Metal::new().unwrap();
    let k = Kernels::compile(&mut d).unwrap();
    for (width, rank) in [(17usize, 7usize), (132, 64), (129, 35), (2304, 64)] {
        for batch in [1usize, 3, 9] {
            for (fdtype, gdtype) in [(0, 0), (1, 1), (1, 2)] {
                let encode = |values: Vec<f32>, dtype| -> Vec<u8> {
                    match dtype {
                        0 => bytemuck::cast_slice(&values).to_vec(),
                        1 => values
                            .iter()
                            .flat_map(|&v| half::bf16::from_f32(v).to_bits().to_le_bytes())
                            .collect(),
                        _ => values
                            .iter()
                            .flat_map(|&v| half::f16::from_f32(v).to_bits().to_le_bytes())
                            .collect(),
                    }
                };
                let f = d
                    .upload(&encode(
                        (0..width * rank)
                            .map(|i| ((i * 11 % 37) as f32 - 18.) / 512.)
                            .collect(),
                        fdtype,
                    ))
                    .unwrap();
                let g = d
                    .upload(&encode(
                        (0..width * rank)
                            .map(|i| ((i * 7 % 31) as f32 - 15.) / 256.)
                            .collect(),
                        gdtype,
                    ))
                    .unwrap();
                let low = upload(
                    &mut d,
                    &(0..batch * rank)
                        .map(|i| ((i * 7 % 19) as f32 - 9.) / 16.)
                        .collect::<Vec<_>>(),
                );
                let x = upload(
                    &mut d,
                    &(0..batch * width)
                        .map(|i| ((i * 13 % 101) as f32 - 50.) / 32.)
                        .collect::<Vec<_>>(),
                );
                let msg = upload(
                    &mut d,
                    &(0..batch * width)
                        .map(|i| ((i * 17 % 59) as f32 - 29.) / 32.)
                        .collect::<Vec<_>>(),
                );
                let forget = d.allocate(batch * width * 4).unwrap();
                let gain = d.allocate(batch * width * 4).unwrap();
                let old = d.allocate(batch * width * 4).unwrap();
                let new = d.allocate(batch * width * 4).unwrap();
                let vectorized = rank % 4 == 0 && width >= 128;
                let (linear, threads, group) = if batch > 1 {
                    ("matmul", width.div_ceil(8) * batch.div_ceil(8) * 32, 32)
                } else if vectorized {
                    ("matvec4", width.div_ceil(4) * 32, 128)
                } else {
                    ("matvec", width * 32, 128)
                };
                // SAFETY: all weights, activations, outputs, and dispatches follow
                // the projection and controller ABIs, including partial tiles.
                unsafe {
                    k.run(
                        &mut d,
                        linear,
                        &[&forget, &f, &low],
                        Params::new(&[width, rank, fdtype, batch, 0], &[]),
                        threads,
                        group,
                    )
                    .unwrap();
                    k.run(
                        &mut d,
                        linear,
                        &[&gain, &g, &low],
                        Params::new(&[width, rank, gdtype, batch, 0], &[]),
                        threads,
                        group,
                    )
                    .unwrap();
                    k.run(
                        &mut d,
                        "controller",
                        &[&old, &x, &msg, &forget, &gain],
                        Params::new(&[batch * width], &[]),
                        batch * width,
                        128,
                    )
                    .unwrap();
                    k.run(
                        &mut d,
                        if batch > 1 {
                            "controller_matmul"
                        } else {
                            "controller_matvec"
                        },
                        &[&new, &x, &msg, &low, &f, &g],
                        Params::new(
                            &[width, rank, fdtype, gdtype, batch, usize::from(vectorized)],
                            &[],
                        ),
                        threads,
                        group,
                    )
                    .unwrap();
                }
                close(&read(&mut d, &new), &read(&mut d, &old), 0.00001);
            }
        }
    }
}

#[test]
#[ignore = "requires a Metal GPU"]
fn parallel_routing_preserves_ties_bias_and_mixture_weights() {
    let mut d = Metal::new().unwrap();
    let k = Kernels::compile(&mut d).unwrap();
    for n in [1usize, 3, 31, 32, 33, 128, 256] {
        for top in [n.min(8), n.min(32)] {
            let scores: Vec<_> = (0..3 * n)
                .map(|i| {
                    if i < n {
                        0.
                    } else {
                        ((i * 17 % 31) as f32 - 15.) * 10.
                    }
                })
                .collect();
            let bias: Vec<_> = (0..n).map(|i| (i % 3) as f32 - 1.).collect();
            let logits = upload(&mut d, &scores);
            let bias = upload(&mut d, &bias);
            let old = d.allocate(3 * top * 8).unwrap();
            let new = d.allocate(3 * top * 8).unwrap();
            let p = Params::new(&[n, top, 3], &[8.]);
            // SAFETY: three rows of n scores, n biases, and top output pairs per token.
            unsafe {
                k.run(&mut d, "route", &[&old, &logits, &bias], p, 3, 32)
                    .unwrap();
                k.run(
                    &mut d,
                    "route_parallel",
                    &[&new, &logits, &bias],
                    p,
                    3 * 32,
                    32,
                )
                .unwrap();
            }
            let old = read(&mut d, &old);
            let new = read(&mut d, &new);
            for (a, b) in old.chunks_exact(2).zip(new.chunks_exact(2)) {
                assert_eq!(a[0], b[0], "expert identity or tie order changed");
            }
            close(&new, &old, 0.000001);
        }
    }
}

#[test]
#[ignore = "requires a Metal GPU"]
fn wide_normalization_matches_reference_with_zero_rows_and_tails() {
    let mut d = Metal::new().unwrap();
    let k = Kernels::compile(&mut d).unwrap();
    for width in [1usize, 17, 128, 513, 2304, 8192] {
        let x: Vec<_> = (0..3 * width)
            .map(|i| {
                if i < width {
                    0.
                } else {
                    ((i * 13 % 101) as f32 - 50.) / 7.
                }
            })
            .collect();
        let weights: Vec<_> = (0..width).map(|i| 0.5 + (i % 7) as f32 / 10.).collect();
        let input = upload(&mut d, &x);
        let weights = upload(&mut d, &weights);
        for weighted in [0, 1] {
            let old = d.allocate(3 * width * 4).unwrap();
            let new = d.allocate(3 * width * 4).unwrap();
            let p = Params::new(&[width, 3, weighted], &[1.0e-6]);
            // SAFETY: three width-element rows, width weights, and matching outputs.
            unsafe {
                k.run(&mut d, "rms", &[&old, &input, &weights], p, 3 * 32, 32)
                    .unwrap();
                k.run(
                    &mut d,
                    "rms_parallel",
                    &[&new, &input, &weights],
                    p,
                    3 * 256,
                    256,
                )
                .unwrap();
            }
            close(&read(&mut d, &new), &read(&mut d, &old), 0.00001);
        }
        let msg = upload(&mut d, &x.iter().map(|v| v * 0.3 + 1.).collect::<Vec<_>>());
        let old = d.allocate(6 * width * 4).unwrap();
        let new = d.allocate(6 * width * 4).unwrap();
        let p = Params::new(&[width, 3], &[1.0e-6]);
        // SAFETY: three width-element rows in each input, concatenated output rows.
        unsafe {
            k.run(&mut d, "concat_norm", &[&old, &input, &msg], p, 3 * 32, 32)
                .unwrap();
            k.run(
                &mut d,
                "concat_norm_parallel",
                &[&new, &input, &msg],
                p,
                3 * 256,
                256,
            )
            .unwrap();
        }
        close(&read(&mut d, &new), &read(&mut d, &old), 0.00001);
    }
}

// These tests are explicit hardware tests, never silently passed when no GPU exists.
#[test]
#[ignore = "requires a Metal GPU"]
fn parallel_residual_reads_match_single_simd_with_offsets_and_tails() {
    let mut device = Metal::new().unwrap();
    let kernels = Kernels::compile(&mut device).unwrap();
    for spread in [false, true] {
        for width in [17, 127, 128, 129, 2304] {
            for batch in [1, 3, 32] {
                for blocks in [1, 2, 7, 9, 32] {
                    let values: Vec<_> = (0..(blocks + 3) * batch * width)
                        .map(|i| {
                            let value = ((i * 17 % 211) as f32 - 105.) / 33.;
                            if spread {
                                value * 2f32.powi((i % 13) as i32 - 6)
                            } else {
                                value
                            }
                        })
                        .collect();
                    let bank = upload(&mut device, &values);
                    let live = upload(&mut device, &values[..batch * width]);
                    let query = upload(&mut device, &values[..2 * width]);
                    let expected = device.allocate(batch * width * 4).unwrap();
                    let initial = vec![f32::NAN; batch * width + 17];
                    let actual = upload(&mut device, &initial);
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
                        &read(&mut device, &actual)[..batch * width],
                        &read(&mut device, &expected),
                        0.000001,
                    );
                    for shards in [0usize, 1, 4, 8] {
                        let prefetched = upload(&mut device, &initial);
                        // SAFETY: the prefetch variant has the same buffers, with
                        // disjoint output shards. NaN guards detect excess writes.
                        unsafe {
                            kernels
                                .run(
                                    &mut device,
                                    "block_read_prefetch",
                                    &[&prefetched, &bank, &live, &query],
                                    Params::new(&[width, blocks, width, 3, batch, shards], &[]),
                                    batch * shards.max(1) * 256,
                                    256,
                                )
                                .unwrap();
                        }
                        assert_eq!(
                            device.read(&actual).unwrap(),
                            device.read(&prefetched).unwrap(),
                            "width={width}, batch={batch}, blocks={blocks}, shards={shards}, spread={spread}"
                        );
                    }
                }
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
    for precision in [KvCachePrecision::F32, KvCachePrecision::F16] {
        let e: Expected = serde_json::from_str(match precision {
            KvCachePrecision::F32 => include_str!("fixtures/tiny/expected.json"),
            KvCachePrecision::F16 => include_str!("fixtures/tiny/expected-f16.json"),
        })
        .unwrap();
        let mut model =
            Inkling::load_with_kv_cache(Metal::new().unwrap(), &path, e.tokens.len(), precision)
                .unwrap();
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
        serde_json::from_str(include_str!("fixtures/tiny/expected-f16.json")).unwrap();
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
    for fp16 in [false, true] {
        let round = |x| {
            if fp16 {
                half::f16::from_f32(x).to_f32()
            } else {
                x
            }
        };
        for (position, capacity, batch) in [
            (0, 257, 1),
            (0, 1, 1),
            (3, 3, 5),
            (256, 257, 3),
            (262145, 1024, 1),
            (127, 512, 1),
            (129, 512, 3),
            (1025, 1024, 1),
            (33, 1024, 32),
            (513, 257, 1),
            (126, 257, 5),
            (600, 257, 7),
            (129, 257, 128),
            (33, 1024, 128),
            (32767, 32768, 128),
            (65537, 131072, 128),
            (8191, 8192, 1),
            (65535, 65536, 1),
            (262143, 262144, 1),
        ] {
            let (qh, kh, dim, rel_dim, extent) = if capacity == 512 {
                (6, 2, 160, 17, 64)
            } else if capacity == 1024 {
                (24, 8, 128, 16, 1024)
            } else if capacity > 257 {
                (3, 1, 8, 4, 64)
            } else {
                (6, 2, 40, 4, 64)
            };
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
            .enumerate()
            .map(|(i, v)| {
                if i == 1 || i == 2 {
                    upload_cache(&mut device, v, fp16)
                } else {
                    upload(&mut device, v)
                }
            })
            .collect();
            let n = (position + batch).min(capacity);
            let parts = n.div_ceil(128);
            let partial = device.allocate(batch * qh * parts * (dim + 2) * 4).unwrap();
            let out = device.allocate(batch * qh * dim * 4).unwrap();
            let geometry = [
                qh,
                kh,
                dim,
                position,
                capacity,
                n,
                rel_dim,
                extent,
                parts,
                batch,
                usize::from(fp16),
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
            for (name, threads) in [
                ("attention_partition_wide", 512),
                ("attention_partition_prefetch", 256),
            ] {
                let length = batch * qh * parts * (dim + 2);
                let candidate = upload(&mut device, &vec![f32::NAN; length + 17]);
                // SAFETY: each candidate keeps the same partition/output geometry.
                unsafe {
                    kernels
                        .run(
                            &mut device,
                            name,
                            &[
                                &candidate,
                                &buffers[0],
                                &buffers[1],
                                &buffers[2],
                                &buffers[3],
                                &buffers[4],
                                &buffers[5],
                                &buffers[6],
                            ],
                            Params::new(&geometry, &[]),
                            batch * qh * parts * threads,
                            threads,
                        )
                        .unwrap();
                }
                let bytes = device.read(&candidate).unwrap();
                assert_eq!(
                    &bytes[..length * 4],
                    device.read(&partial).unwrap(),
                    "{name} fp16={fp16} position={position} capacity={capacity} batch={batch}"
                );
                assert!(
                    bytes[length * 4..]
                        .chunks_exact(4)
                        .all(|b| b == f32::NAN.to_le_bytes())
                );
            }
            // Reuse one partial buffer for query subsets. Compare with the
            // original unsplit kernel, including nonzero offsets, tails and
            // ring wrapping; all inputs remain based at the full chunk start.
            let reference = device.read(&out).unwrap();
            for (group, trim_parts) in [
                (1, false),
                (7, false),
                (32, false),
                (64, false),
                (32, true),
                (64, true),
            ]
            .into_iter()
            .filter(|&(n, _)| n < batch)
            {
                let length = group * qh * parts * (dim + 2);
                let partials = upload(&mut device, &vec![f32::NAN; length + 17]);
                let output_length = batch * qh * dim;
                let output = upload(&mut device, &vec![f32::NAN; output_length + 17]);
                for offset in (0..batch).step_by(group) {
                    let count = group.min(batch - offset);
                    let mut p = Params::new(&geometry, &[]);
                    p.u[9] = count as u32;
                    p.u[11] = offset as u32;
                    // Smaller old decoder chunks could end before the next
                    // partition. Also check those shorter reductions at an
                    // unaligned long-context base where compensation is used.
                    let group_parts = if trim_parts {
                        (position + offset + count).min(capacity).div_ceil(128)
                    } else {
                        parts
                    };
                    p.u[8] = group_parts as u32;
                    // SAFETY: each group fits the shared partials and accesses
                    // only its query rows in the full-batch input/output arrays.
                    unsafe {
                        kernels
                            .run(
                                &mut device,
                                "attention_partition_prefetch",
                                &[
                                    &partials,
                                    &buffers[0],
                                    &buffers[1],
                                    &buffers[2],
                                    &buffers[3],
                                    &buffers[4],
                                    &buffers[5],
                                    &buffers[6],
                                ],
                                p,
                                count * qh * group_parts * 256,
                                256,
                            )
                            .unwrap();
                        kernels
                            .run(
                                &mut device,
                                "attention_finish",
                                &[&output, &partials, &buffers[7], &buffers[8]],
                                p,
                                count * qh * 32,
                                32,
                            )
                            .unwrap();
                    }
                }
                let actual = device.read(&output).unwrap();
                assert_eq!(
                    &actual[..output_length * 4],
                    reference,
                    "query group={group}, trim_parts={trim_parts}, fp16={fp16}, position={position}, capacity={capacity}"
                );
                assert!(
                    actual[output_length * 4..]
                        .chunks_exact(4)
                        .all(|b| b == f32::NAN.to_le_bytes())
                );
                assert!(
                    device.read(&partials).unwrap()[length * 4..]
                        .chunks_exact(4)
                        .all(|b| b == f32::NAN.to_le_bytes())
                );
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
                        let mut score: f32 = (0..dim)
                            .map(|i| q[query * dim + i] * round(key[i]))
                            .sum::<f32>()
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
                            sum += weights[j] * round(value);
                        }
                        expected.push(sum / denominator / (1. + (-gate[query * dim + i]).exp()));
                    }
                }
            }
            close(&read(&mut device, &out), &expected, 0.00001);
        }
    }
}

#[test]
#[ignore = "requires a Metal GPU"]
fn kv_cache_writes_round_and_preserve_ring_guards_at_long_positions() {
    let mut device = Metal::new().unwrap();
    let kernels = Kernels::compile(&mut device).unwrap();
    for fp16 in [false, true] {
        let round = |x| {
            if fp16 {
                half::f16::from_f32(x).to_f32()
            } else {
                x
            }
        };
        for (capacity, position, batch) in
            [(3, 0, 1), (3, 2, 7), (257, 262111, 33), (257, 1048574, 2)]
        {
            let channels = 17;
            let len = capacity * channels;
            let mut expected = vec![-13.375; len + 19];
            let keys = upload_cache(&mut device, &expected, fp16);
            let values = upload_cache(&mut device, &expected, fp16);
            let input: Vec<_> = (0..batch * channels)
                .map(|i| (i as f32 * 0.41).sin() * 7.)
                .collect();
            let k = upload(&mut device, &input);
            let v = upload(&mut device, &input.iter().map(|x| -x).collect::<Vec<_>>());
            // SAFETY: the ring and fresh rows match the ABI, with extra sentinel
            // elements beyond the logical cache. Oversized chunks write each slot once.
            unsafe {
                kernels
                    .run(
                        &mut device,
                        "cache_write",
                        &[&keys, &values, &k, &v],
                        Params::new(
                            &[channels, capacity, position, batch, usize::from(fp16)],
                            &[],
                        ),
                        batch * channels,
                        128,
                    )
                    .unwrap();
            }
            for token in batch.saturating_sub(capacity)..batch {
                for channel in 0..channels {
                    expected[((position + token) % capacity) * channels + channel] =
                        round(input[token * channels + channel]);
                }
            }
            let read_cache = |d: &mut Metal, b: &MetalBuffer| -> Vec<f32> {
                if fp16 {
                    d.read(b)
                        .unwrap()
                        .chunks_exact(2)
                        .map(|b| {
                            half::f16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).to_f32()
                        })
                        .collect()
                } else {
                    read(d, b)
                }
            };
            assert_eq!(read_cache(&mut device, &keys), expected);
            for token in batch.saturating_sub(capacity)..batch {
                for channel in 0..channels {
                    expected[((position + token) % capacity) * channels + channel] =
                        round(-input[token * channels + channel]);
                }
            }
            assert_eq!(read_cache(&mut device, &values), expected);
        }
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
    // Larger batches must preserve the complete continuation, including loop
    // messages, convolution state and a partial final chunk, in both KV formats.
    config["text_config"]["model_max_length"] = 768.into();
    std::fs::write(
        temporary.join("config.json"),
        serde_json::to_vec(&config).unwrap(),
    )
    .unwrap();
    // Reach the 64- and 32-query workspace limits while the decoder retains
    // 128-token batches. Include a second prompt with an unaligned cache base.
    let tokens: Vec<u32> = (0..641).map(|i| (i * 7 % 14) as u32).collect();
    for precision in [KvCachePrecision::F16, KvCachePrecision::F32] {
        let mut model =
            Inkling::load_with_kv_cache(Metal::new().unwrap(), &temporary, 768, precision).unwrap();
        let mut expected = None;
        for limit in [32, 64, 128] {
            model.set_prefill_batch_limit(limit).unwrap();
            model.reset().unwrap();
            let mut actual = vec![model.prefill(&tokens[..257]).unwrap()];
            actual.push(model.prefill(&tokens[257..]).unwrap());
            for token in [3, 4, 5] {
                actual.push(model.step(token).unwrap());
            }
            let bits: Vec<_> = actual.iter().flatten().map(|v| v.to_bits()).collect();
            if let Some(expected) = &expected {
                assert_eq!(&bits, expected, "batch={limit}, kv={precision}");
            } else {
                expected = Some(bits);
            }
        }
    }
    std::fs::remove_dir_all(temporary).unwrap();
}

#[test]
#[ignore = "requires a Metal GPU"]
fn large_context_shares_attention_workspace_and_reuses_it_after_reset() {
    let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny");
    let temporary = std::env::temp_dir().join(format!("rink-cache-memory-{}", std::process::id()));
    std::fs::create_dir_all(&temporary).unwrap();
    let mut config = rink_inkling::Config::load(fixture.join("config.json")).unwrap();
    config.text_config.model_max_length = 262144;
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
    let c = &config.text_config;
    let tokens: Vec<_> = (0..32).map(|i| (i * 7 % 14) as u32).collect();
    let mut measurements = Vec::new();
    let mut expected: Option<Vec<f32>> = None;
    for capacity in [128, 262144] {
        let mut model = Inkling::load(Metal::new().unwrap(), &temporary, capacity).unwrap();
        let loaded = model.accelerator().stats().allocated_bytes;
        model.accelerator_mut().reset_stats();
        let logits = model.prefill(&tokens).unwrap();
        let scratch = model.accelerator().stats().allocated_bytes;
        if let Some(expected) = &expected {
            close(&logits, expected, 0.00001);
        } else {
            expected = Some(logits);
        }
        let allocations = model.accelerator().stats().allocations;
        model.prefill(&tokens).unwrap();
        model.step(2).unwrap();
        model.reset().unwrap();
        close(
            &model.prefill(&tokens).unwrap(),
            expected.as_ref().unwrap(),
            0.00001,
        );
        assert_eq!(
            model.accelerator().stats().allocations,
            allocations + 2,
            "only the two new prompt-token uploads should allocate"
        );
        measurements.push((loaded, scratch));
    }
    let full_visits = c.schedule().iter().filter(|v| !c.is_local(v.layer)).count();
    // Growing the context changes only the full-attention caches, at two bytes
    // per key/value element. Looped visits still have independent history.
    assert_eq!(
        measurements[1].0 - measurements[0].0,
        (full_visits * (262144 - 128) * c.num_key_value_heads * c.head_dim * 2 * 2) as u64
    );
    let largest_workspace = 32 * c.num_attention_heads * (262144 / 128) * (c.head_dim + 2) * 4;
    assert!(
        measurements[1].1 - measurements[0].1 <= largest_workspace as u64,
        "attention workspace must scale with the largest layer, not the number of visits"
    );
    std::fs::remove_dir_all(temporary).unwrap();
}

#[test]
#[ignore = "requires a Metal GPU and about 5 GiB for the unsplit 256K reference"]
fn full_256k_attention_geometry_retrieves_the_first_token() {
    let mut device = Metal::new().unwrap();
    let kernels = Kernels::compile(&mut device).unwrap();
    for (capacity, query_group) in [(65536usize, 128usize), (131072, 64), (262144, 32)] {
        let batch = 128;
        let (qh, kh, dim) = (24usize, 8usize, 128usize);
        let position = capacity - batch;
        let elements = capacity * kh * dim;
        let mut data = vec![0u16; elements];
        for head in 0..kh {
            data[head * dim] = half::f16::from_f32(20.).to_bits();
        }
        // The last batch of cache entries are not yet written. Attention must use the
        // fresh input rows for these positions, with causal masking within the chunk.
        data[position * kh * dim..].fill(half::f16::NAN.to_bits());
        let keys = device.upload(bytemuck::cast_slice(&data)).unwrap();
        data.fill(half::f16::from_f32(0.25).to_bits());
        let first_value = |head: usize, channel: usize| {
            if channel % 2 == 0 {
                1. + head as f32 / 8.
            } else {
                -1.
            }
        };
        for head in 0..kh {
            for channel in 0..dim {
                data[head * dim + channel] =
                    half::f16::from_f32(first_value(head, channel)).to_bits();
            }
        }
        data[position * kh * dim..].fill(half::f16::NAN.to_bits());
        let values = device.upload(bytemuck::cast_slice(&data)).unwrap();
        drop(data);
        let mut queries = vec![0.; batch * qh * dim];
        for query in queries.chunks_exact_mut(dim) {
            query[0] = (dim as f32).sqrt();
        }
        let query = upload(&mut device, &queries);
        let new_keys = upload(&mut device, &vec![0.; batch * kh * dim]);
        let new_values = upload(&mut device, &vec![0.75; batch * kh * dim]);
        let rel = upload(&mut device, &vec![0.; batch * qh]);
        let profiles = upload(&mut device, &[0.]);
        let sinks = upload(&mut device, &vec![0.; batch * qh]);
        let gate = upload(&mut device, &vec![0.; batch * qh * dim]);
        let parts = capacity.div_ceil(128);
        let partials = device.allocate(batch * qh * parts * (dim + 2) * 4).unwrap();
        let output = device.allocate(batch * qh * dim * 4).unwrap();
        let geometry = Params::new(
            &[
                qh, kh, dim, position, capacity, capacity, 1, 1, parts, batch, 1,
            ],
            &[],
        );
        // SAFETY: full released-model geometry with FP16 KV buffers, FP32 fresh
        // rows. The unsplit reference uses a larger workspace for comparison only.
        // The production path below stays within 780 MiB at every history length.
        unsafe {
            kernels
                .run(
                    &mut device,
                    "attention_partition",
                    &[
                        &partials,
                        &query,
                        &keys,
                        &values,
                        &new_keys,
                        &new_values,
                        &rel,
                        &profiles,
                    ],
                    geometry,
                    batch * qh * parts * 128,
                    128,
                )
                .unwrap();
            kernels
                .run(
                    &mut device,
                    "attention_finish",
                    &[&output, &partials, &sinks, &gate],
                    geometry,
                    batch * qh * 32,
                    32,
                )
                .unwrap();
        }
        let mut expected = Vec::with_capacity(batch * qh * dim);
        for token in 0..batch {
            let n = position + token + 1;
            // The first cached key scores 20; every other key and the sink score 0.
            // Summing these weights analytically avoids a second huge attention matrix.
            let first_weight = 20f64.exp();
            let denominator = first_weight + n as f64;
            for head in 0..qh {
                for channel in 0..dim {
                    let numerator = first_weight
                        * f64::from(first_value(head / (qh / kh), channel))
                        + (position - 1) as f64 * 0.25
                        + (token + 1) as f64 * 0.75;
                    expected.push((0.5 * numerator / denominator) as f32);
                }
            }
        }
        let reference_output = read(&mut device, &output);
        close(&reference_output, &expected, 0.00001);
        drop(partials);
        let workspace_bytes = query_group * qh * parts * (dim + 2) * 4;
        assert_eq!(workspace_bytes, 780 * 1024 * 1024);
        let partials = device.allocate(workspace_bytes).unwrap();
        for offset in (0..batch).step_by(query_group) {
            let mut p = geometry;
            p.u[9] = query_group as u32;
            p.u[11] = offset as u32;
            // SAFETY: each query group fits the 780 MiB partial buffer; the
            // inputs/output retain all 128 rows and the original KV base.
            unsafe {
                kernels
                    .run(
                        &mut device,
                        "attention_partition_prefetch",
                        &[
                            &partials,
                            &query,
                            &keys,
                            &values,
                            &new_keys,
                            &new_values,
                            &rel,
                            &profiles,
                        ],
                        p,
                        query_group * qh * parts * 256,
                        256,
                    )
                    .unwrap();
                kernels
                    .run(
                        &mut device,
                        "attention_finish",
                        &[&output, &partials, &sinks, &gate],
                        p,
                        query_group * qh * 32,
                        32,
                    )
                    .unwrap();
            }
        }
        assert_eq!(
            bytemuck::cast_slice::<_, u8>(&read(&mut device, &output)),
            bytemuck::cast_slice::<_, u8>(&reference_output)
        );

        // Also compare the decode path at the released model's full cache geometry.
        // Only the first fresh token is visible; unwritten later slots contain NaNs.
        let length = qh * parts * (dim + 2);
        let initial = vec![f32::NAN; length + 17];
        let reference = upload(&mut device, &initial);
        let candidate = upload(&mut device, &initial);
        let decode_geometry = Params::new(
            &[
                qh,
                kh,
                dim,
                position,
                capacity,
                position + 1,
                1,
                1,
                parts,
                1,
                1,
            ],
            &[],
        );
        for (name, buffer, threads) in [
            ("attention_partition", &reference, 128),
            ("attention_partition_prefetch", &candidate, 256),
        ] {
            // SAFETY: the same native caches and first fresh token serve both kernels;
            // each output has one token's partitions plus unwritten sentinel guards.
            unsafe {
                kernels
                    .run(
                        &mut device,
                        name,
                        &[
                            buffer,
                            &query,
                            &keys,
                            &values,
                            &new_keys,
                            &new_values,
                            &rel,
                            &profiles,
                        ],
                        decode_geometry,
                        qh * parts * threads,
                        threads,
                    )
                    .unwrap();
            }
        }
        assert_eq!(
            device.read(&candidate).unwrap(),
            device.read(&reference).unwrap()
        );
    }
}

#[test]
#[ignore = "requires a Metal GPU"]
fn projection_scheduling_preserves_every_output_bit() {
    let mut d = Metal::new().unwrap();
    let k = Kernels::compile(&mut d).unwrap();
    // Wide exponents, cancellation, tails, dtype conversions, and nonzero input
    // offsets exercise arithmetic that a greedy-token comparison can miss.
    let values = |n: usize, seed: u32| -> Vec<f32> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                let sign = if state & 1 == 0 { 1. } else { -1. };
                sign * ((state >> 8) % 1024) as f32 / 1024. * 2f32.powi((state % 13) as i32 - 6)
            })
            .collect()
    };
    for dtype in 0..3 {
        for (rows, cols) in [
            (1usize, 4usize),
            (5, 1),
            (9, 3),
            (17, 31),
            (33, 33),
            (7, 132),
            (24, 2304),
            (64, 4608),
            (127, 128),
            (128, 256),
            (131, 2304),
            (131, 3072),
            (384, 4608),
            (37, 29),
        ] {
            let w = values(rows * cols, 91);
            let weights = match dtype {
                0 => upload(&mut d, &w),
                1 => d
                    .upload(
                        &w.iter()
                            .flat_map(|v| half::bf16::from_f32(*v).to_bits().to_le_bytes())
                            .collect::<Vec<_>>(),
                    )
                    .unwrap(),
                _ => d
                    .upload(
                        &w.iter()
                            .flat_map(|v| half::f16::from_f32(*v).to_bits().to_le_bytes())
                            .collect::<Vec<_>>(),
                    )
                    .unwrap(),
            };
            for batch in [1usize, 2, 7, 8, 15, 16, 17, 31, 32, 33, 64, 128] {
                let offsets: &[usize] = if batch == 1 { &[0, 12] } else { &[0, 1, 3, 12] };
                for &offset in offsets {
                    let input = upload(&mut d, &values(offset + batch * cols, 37));
                    let initial = vec![f32::NAN; batch * rows + 17];
                    let old = upload(&mut d, &initial);
                    let new = upload(&mut d, &initial);
                    let p = Params::new(&[rows, cols, dtype, batch, offset], &[]);
                    if batch == 1 {
                        // SAFETY: group size only changes placement of independent
                        // SIMD rows. The scalar ABI supports any column count.
                        unsafe {
                            k.run(
                                &mut d,
                                "matvec",
                                &[&old, &weights, &input],
                                p,
                                rows * 32,
                                128,
                            )
                            .unwrap();
                            k.run(
                                &mut d,
                                "matvec",
                                &[&new, &weights, &input],
                                p,
                                rows * 32,
                                32,
                            )
                            .unwrap();
                        }
                        assert_eq!(
                            d.read(&new).unwrap(),
                            d.read(&old).unwrap(),
                            "scalar threadgroup placement changed output bits"
                        );
                        // SAFETY: prefetching retains the scalar projection ABI,
                        // including short columns and non-aligned tails.
                        unsafe {
                            k.run(
                                &mut d,
                                "matvec_prefetch",
                                &[&new, &weights, &input],
                                p,
                                rows * 32,
                                32,
                            )
                            .unwrap();
                        }
                        assert_eq!(
                            d.read(&new).unwrap(),
                            d.read(&old).unwrap(),
                            "prefetch changed scalar output bits"
                        );
                        if dtype == 1 && [2304, 4608].contains(&cols) {
                            let name = if cols == 2304 {
                                "matvec_scalar_bf16_2304"
                            } else {
                                "matvec_scalar_bf16_4608"
                            };
                            // SAFETY: BF16 storage and fixed column count match this specialization.
                            unsafe {
                                k.run(&mut d, name, &[&new, &weights, &input], p, rows * 32, 32)
                                    .unwrap();
                            }
                            assert_eq!(
                                d.read(&new).unwrap(),
                                d.read(&old).unwrap(),
                                "{name} rows={rows} offset={offset}"
                            );
                        }
                    }
                    if batch == 1 && cols % 4 != 0 {
                        continue;
                    }
                    // SAFETY: padded outputs include sentinel guards. Packed vector
                    // inputs have four-float aligned offsets/columns. Matrix tiles
                    // guard all row, token, and reduction tails.
                    unsafe {
                        if batch == 1 {
                            k.run(
                                &mut d,
                                "matvec4",
                                &[&old, &weights, &input],
                                p,
                                rows.div_ceil(4) * 32,
                                128,
                            )
                            .unwrap();
                            k.run(
                                &mut d,
                                "matvec_packed",
                                &[&new, &weights, &input],
                                p,
                                rows * 32,
                                128,
                            )
                            .unwrap();
                        } else {
                            k.run(
                                &mut d,
                                "matmul",
                                &[&old, &weights, &input],
                                p,
                                rows.div_ceil(8) * batch.div_ceil(8) * 32,
                                32,
                            )
                            .unwrap();
                            k.run(
                                &mut d,
                                "matmul_tiled",
                                &[&new, &weights, &input],
                                p,
                                rows.div_ceil(32) * batch.div_ceil(16) * 256,
                                256,
                            )
                            .unwrap();
                        }
                    }
                    let expected = read(&mut d, &old);
                    let actual = read(&mut d, &new);
                    for (i, (&a, &b)) in actual.iter().zip(&expected).enumerate() {
                        assert_eq!(
                            a.to_bits(),
                            b.to_bits(),
                            "dtype={dtype} rows={rows} cols={cols} batch={batch} offset={offset} index={i}: {a} != {b}"
                        );
                        if i < batch * rows {
                            assert!(a.is_finite());
                        }
                    }
                    if batch == 1 && dtype == 1 && [2304, 3072].contains(&cols) {
                        let name = if cols == 2304 {
                            "matvec_bf16_2304"
                        } else {
                            "matvec_bf16_3072"
                        };
                        // SAFETY: aligned input offset, BF16 storage, fixed columns, and padded rows.
                        unsafe {
                            k.run(&mut d, name, &[&new, &weights, &input], p, rows * 32, 128)
                                .unwrap();
                        }
                        assert_eq!(
                            d.read(&new).unwrap(),
                            d.read(&old).unwrap(),
                            "{name} rows={rows} offset={offset}"
                        );
                    }
                    if batch > 1 {
                        let name = "matmul_packed";
                        let packed = upload(&mut d, &initial);
                        // SAFETY: four-wide staging handles partial vectors;
                        // output tile geometry and guards are unchanged.
                        unsafe {
                            k.run(
                                &mut d,
                                name,
                                &[&packed, &weights, &input],
                                p,
                                rows.div_ceil(32) * batch.div_ceil(16) * 256,
                                256,
                            )
                            .unwrap();
                        }
                        assert_eq!(
                            d.read(&packed).unwrap(),
                            d.read(&old).unwrap(),
                            "{name}: dtype={dtype} rows={rows} cols={cols} batch={batch} offset={offset}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
#[ignore = "requires a Metal GPU"]
fn nvfp4_table_preserves_codes_scales_and_signed_zero() {
    let mut d = Metal::new().unwrap();
    let k = Kernels::compile(&mut d).unwrap();
    let factors = [0.0f32, 1.0 / 7.0, 1.0, 1.0 / 4096.0, 4096.0];
    let f = upload(&mut d, &factors);
    let table = upload(&mut d, &vec![f32::NAN; factors.len() * 4096 + 17]);
    // SAFETY: the output includes every code/scale/factor tuple and sentinel guards.
    unsafe {
        k.run(
            &mut d,
            "nvfp4_decode_table",
            &[&table, &f],
            Params::new(&[factors.len()], &[]),
            factors.len() * 4096,
            128,
        )
        .unwrap();
    }
    let values = read(&mut d, &table);
    for (expert, &factor) in factors.iter().enumerate() {
        for scale in 0u32..256 {
            for code in 0u32..16 {
                let exponent = (scale >> 3) & 15;
                let mantissa = scale & 7;
                let value = if exponent == 0 {
                    mantissa as f32 / 512.0
                } else if exponent == 15 && mantissa == 7 {
                    f32::NAN
                } else {
                    (1.0 + mantissa as f32 / 8.0) * 2f32.powi(exponent as i32 - 7)
                };
                let scale_value = if scale & 128 != 0 { -value } else { value };
                let magnitude = [0.0f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0][(code & 7) as usize];
                let expected =
                    magnitude * if code & 8 != 0 { -1.0 } else { 1.0 } * scale_value * factor;
                let actual = values[expert * 4096 + scale as usize * 16 + code as usize];
                if expected.is_nan() {
                    assert!(actual.is_nan());
                } else {
                    assert_eq!(
                        actual.to_bits(),
                        expected.to_bits(),
                        "expert={expert} scale={scale} code={code}"
                    );
                }
            }
        }
    }
    assert!(
        values[factors.len() * 4096..]
            .iter()
            .all(|v| v.to_bits() == f32::NAN.to_bits())
    );
}

#[test]
#[ignore = "requires a Metal GPU"]
fn native_mxfp8_projections_preserve_all_bits_and_route_inputs() {
    let mut d = Metal::new().unwrap();
    let k = Kernels::compile(&mut d).unwrap();
    let (experts, top, rows) = (3usize, 2usize, 2304usize);
    for down in [false, true] {
        let cols = if down { 1152 } else { 2304 };
        let count = experts * rows * cols;
        let w = d
            .upload(
                &(0..count)
                    .map(|i| {
                        [
                            0x00, 0x80, 0x01, 0x81, 0x38, 0xbc, 0x7e, 0xfe, 0x40, 0x39, 0x10,
                        ][i % 11]
                    })
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let scales = d
            .upload(
                &(0..count / 32)
                    .map(|i| [121, 125, 127, 129, 133][i % 5])
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let factors = upload(&mut d, &[1.]);
        for batch in [1usize, 3, 7] {
            let slots = batch * top;
            let width = if down { rows } else { rows / 2 };
            let x = upload(
                &mut d,
                &(0..cols * if down { slots } else { batch })
                    .map(|i| ((i * 13 % 127) as f32 - 63.) / 131.)
                    .collect::<Vec<_>>(),
            );
            let route = upload(
                &mut d,
                &(0..slots)
                    .flat_map(|slot| [((slot + 2) % experts) as f32, 0.5])
                    .collect::<Vec<_>>(),
            );
            let old = upload(&mut d, &vec![f32::NAN; slots * width + 17]);
            let new = upload(&mut d, &vec![f32::NAN; slots * width + 17]);
            let p = Params::new(&[rows, cols, top, 1, usize::from(down), batch], &[4., 25.]);
            let (reference, candidate) = if down {
                ("expert_matvec", "expert_matvec_mxfp8_native")
            } else {
                ("expert_swiglu_matvec", "expert_swiglu_mxfp8_native")
            };
            // SAFETY: native dimensions, MXFP8 blocks, valid routes, and shared/per-route input lengths.
            unsafe {
                k.run(
                    &mut d,
                    reference,
                    &[&old, &w, &scales, &factors, &x, &route],
                    p,
                    slots * width * 32,
                    128,
                )
                .unwrap();
                k.run(
                    &mut d,
                    candidate,
                    &[&new, &w, &scales, &factors, &x, &route],
                    p,
                    slots * width * 32,
                    128,
                )
                .unwrap();
            }
            assert_eq!(
                d.read(&old).unwrap(),
                d.read(&new).unwrap(),
                "{candidate} batch={batch}"
            );
            let values = read(&mut d, &new);
            assert!(values[..slots * width].iter().all(|v| v.is_finite()));
            assert!(
                values[slots * width..]
                    .iter()
                    .all(|v| v.to_bits() == f32::NAN.to_bits())
            );
        }
    }
}
