//! Benchmark-only FP32 GEMM experiments; production dispatch is unchanged.
//! Native dimensions, one warmup and four alternating timing rounds. Exact
//! outputs and guard bytes are checked. Cached synthetic weights are not an
//! end-to-end throughput prediction.
#[cfg(target_os = "macos")]
mod metal {
    use rink_core::{Accelerator, Argument, KernelSource, Result, ensure};
    use rink_inkling::kernels::{Kernels, Params, SOURCE};
    use rink_metal::{Metal, MetalBuffer, MetalKernel};
    use serde_json::json;

    const SHAPES: &[(usize, usize, usize)] = &[
        (16, 32, 32),
        (32, 32, 32),
        (32, 32, 64),
        (32, 32, 128),
        (32, 64, 32),
        (32, 64, 64),
        (64, 32, 32),
        (64, 64, 32),
        (64, 64, 64),
    ];
    fn values(d: &mut Metal, n: usize) -> Result<MetalBuffer> {
        let x: Vec<f32> = (0..n)
            .map(|i| ((i * 17 % 113) as f32 - 56.) / 113.)
            .collect();
        d.upload(bytemuck::cast_slice(&x))
    }
    // SAFETY: callers supply native dimensions and the eight GEMM buffer bindings.
    unsafe fn launch(
        d: &mut Metal,
        kernel: &MetalKernel,
        buffers: &[&MetalBuffer],
        p: Params,
        groups: usize,
    ) -> Result<()> {
        let mut args = std::array::from_fn::<_, 16, _>(|_| Argument::Skip);
        for (slot, b) in args.iter_mut().zip(buffers) {
            *slot = Argument::Buffer(*b);
        }
        args[15] = Argument::Bytes(bytemuck::bytes_of(&p));
        unsafe { d.dispatch(kernel, &args, groups * 256, 256) }
    }

    pub fn run() -> Result<()> {
        let mut d = Metal::new()?;
        d.set_profiling(true);
        let production = Kernels::compile(&mut d)?;
        let source = format!(
            "{}\n{}",
            SOURCE.metal.unwrap(),
            include_str!("prefill_tiling.metal")
        );
        let library = KernelSource {
            name: "prefill-tiling-probe",
            metal: Some(&source),
            cuda: None,
            subgroup_size: Some(32),
        };
        let mut dense = Vec::new();
        let mut expert = Vec::new();
        for &(m, n, k) in SHAPES {
            for (prefix, target) in [("dense", &mut dense), ("expert", &mut expert)] {
                let name = format!("{prefix}_{m}_{n}_{k}");
                target.push((name.clone(), d.compile(&library, &name)?));
            }
        }
        let mut results = Vec::new();
        let batch = 1024usize;
        // Dense BF16 weights and both native expert quantization formats.
        for (kind, format, rows, cols, down) in [
            ("dense", 1, 64usize, 4608usize, false),
            ("dense", 1, 384, 2304, false),
            ("dense", 1, 1024, 2304, false),
            ("dense", 1, 3072, 2304, false),
            ("dense", 1, 2304, 3072, false),
            ("expert", 0, 2304, 2304, false),
            ("expert", 0, 2304, 1152, true),
            ("expert", 1, 2304, 2304, false),
            ("expert", 1, 2304, 1152, true),
        ] {
            let is_expert = kind == "expert";
            let experts = if is_expert { 128usize } else { 1 };
            let top = if is_expert { 8usize } else { 1 };
            let count = experts * rows * cols;
            let weights: Vec<u8> = if !is_expert {
                (0..count)
                    .flat_map(|i| {
                        half::bf16::from_f32(((i % 113) as f32 - 56.) / 113.)
                            .to_bits()
                            .to_le_bytes()
                    })
                    .collect()
            } else {
                (0..if format == 0 { count / 2 } else { count })
                    .map(|i| {
                        if format == 0 {
                            (i.wrapping_mul(73) + 9) as u8
                        } else {
                            [0x01, 0x38, 0xbc, 0x10, 0xb0, 0x3c, 0x40][i % 7]
                        }
                    })
                    .collect()
            };
            let w = d.upload(&weights)?;
            drop(weights);
            let scale_codes: Vec<u8> = (0..if is_expert {
                count / if format == 0 { 16 } else { 32 }
            } else {
                4
            })
                .map(|i| {
                    if format == 0 {
                        [0x30, 0x38, 0x39, 0x40, 0x33][i % 5]
                    } else {
                        [125, 126, 127, 128, 129][i % 5]
                    }
                })
                .collect();
            let scales = d.upload(&scale_codes)?;
            drop(scale_codes);
            let factors = values(&mut d, experts)?;
            let routes: Vec<f32> = (0..batch * top)
                .flat_map(|i| {
                    [
                        ((i / top * 7 + i % top * 13) % experts) as f32,
                        1. / top as f32,
                    ]
                })
                .collect();
            let routing = d.upload(bytemuck::cast_slice(&routes))?;
            let slots = d.allocate(experts * (batch + 1) * 4)?;
            let max_tiles = (batch * top + 15 * experts) / 16;
            let tiles = d.allocate((1 + 2 * max_tiles) * 4)?;
            let x = values(&mut d, batch * cols * if down { top } else { 1 })?;
            let output_len = batch * top * rows + 64;
            // SAFETY: initialized below before any dispatch or readback.
            let y = unsafe { d.allocate_uninitialized(output_len * 4)? };
            let table = d.allocate(if is_expert && format == 0 {
                experts * 4096 * 4
            } else {
                4
            })?;
            if is_expert {
                // SAFETY: complete native weights, per-expert factors and valid distinct routes.
                unsafe {
                    if format == 0 {
                        production.run(
                            &mut d,
                            "nvfp4_decode_table",
                            &[&table, &factors],
                            Params::new(&[experts], &[]),
                            experts * 4096,
                            128,
                        )?;
                    }
                    production.run(
                        &mut d,
                        "expert_groups",
                        &[&slots, &routing],
                        Params::new(&[experts, top, batch], &[]),
                        experts * 32,
                        32,
                    )?;
                }
            }
            let buffers = [&y, &w, &scales, &factors, &x, &slots, &tiles, &table];
            let p = Params::new(&[rows, cols, top, format, usize::from(down), batch], &[]);
            let candidates = if is_expert { &expert } else { &dense };
            let mut samples = vec![Vec::new(); candidates.len() + 1];
            let mut expected = Vec::new();
            for round in 0..5 {
                for step in 0..samples.len() {
                    let index = if round % 2 == 0 {
                        step
                    } else {
                        samples.len() - 1 - step
                    };
                    let (m, n, _) = if index == 0 {
                        (if is_expert { 32 } else { 16 }, 32, 32)
                    } else {
                        SHAPES[index - 1]
                    };
                    if is_expert {
                        // SAFETY: tile buffer covers the smallest candidate's worst-case list.
                        unsafe {
                            production.run(
                                &mut d,
                                "expert_tiles",
                                &[&tiles, &slots],
                                Params::new(&[experts, batch, m], &[]),
                                32,
                                32,
                            )?;
                        }
                    }
                    if round == 0 {
                        // Poison each candidate independently so a missing store
                        // cannot inherit the reference result and pass the check.
                        // SAFETY: fills every output and guard element.
                        unsafe {
                            d.initialize_f32(&y, output_len, |out| {
                                out.fill(std::mem::MaybeUninit::new(f32::from_bits(u32::MAX)));
                            })?;
                        }
                    }
                    d.synchronize()?;
                    d.reset_stats();
                    for _ in 0..4 {
                        let tile_count = if is_expert {
                            (batch * top + (m - 1) * experts) / m
                        } else {
                            batch.div_ceil(m)
                        };
                        // SAFETY: all candidates use these validated native dimensions,
                        // complete buffers and exactly 256 threads per threadgroup.
                        unsafe {
                            if index == 0 {
                                let (name, inputs, params) = if is_expert {
                                    ("expert_matmul_m32k32", buffers.to_vec(), p)
                                } else {
                                    (
                                        "matmul_packed",
                                        vec![&y, &w, &x],
                                        Params::new(&[rows, cols, format, batch, 0], &[]),
                                    )
                                };
                                production.run(
                                    &mut d,
                                    name,
                                    &inputs,
                                    params,
                                    tile_count * rows.div_ceil(n) * 256,
                                    256,
                                )?;
                            } else {
                                launch(
                                    &mut d,
                                    &candidates[index - 1].1,
                                    &buffers,
                                    p,
                                    tile_count * rows.div_ceil(n),
                                )?;
                            }
                        }
                    }
                    d.synchronize()?;
                    ensure!(d.stats().allocations == 0, "GEMM allocated a device buffer");
                    if round > 0 {
                        samples[index].push(d.stats().gpu_seconds * 1e6 / 4.);
                    } else {
                        let actual = d.read(&y)?;
                        ensure!(
                            actual[actual.len() - 256..].iter().all(|&b| b == 0xff),
                            "output guard overwritten"
                        );
                        if index == 0 {
                            expected = actual;
                        } else {
                            ensure!(
                                actual == expected,
                                "{kind} candidate {index} differs from production"
                            );
                        }
                    }
                }
            }
            for (i, samples) in samples.iter_mut().enumerate() {
                samples.sort_by(f64::total_cmp);
                let name = if i == 0 {
                    "production"
                } else {
                    &candidates[i - 1].0
                };
                let us = (samples[1] + samples[2]) / 2.;
                eprintln!("{kind} {rows}x{cols}, format {format}: {name} {us:.1} us");
                results.push(json!({"kind":kind,"format":format,"rows":rows,"cols":cols,"batch":batch,"down":down,"kernel":name,"median_gpu_us":us,"trials_gpu_us":samples,"bit_identical":true,"device_buffer_allocations":0}));
            }
        }
        println!("{}", serde_json::to_string_pretty(&results)?);
        Ok(())
    }
}
#[cfg(target_os = "macos")]
fn main() -> rink_core::Result<()> {
    metal::run()
}
#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("This benchmark requires Metal.");
}
