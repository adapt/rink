//! Compare exact native expert specialization with the generic prefill kernel.
#[cfg(target_os = "macos")]
fn main() -> anyhow::Result<()> {
    use rink_core::{Accelerator, Argument};
    use rink_inkling::kernels::{Kernels, Params, SOURCE};
    use rink_metal::Metal;
    let mut d = Metal::new()?;
    d.set_profiling(true);
    let production = Kernels::compile(&mut d)?;
    let (experts, rows, top) = (128usize, 2304usize, 8usize);
    for (format, down) in [(0usize, false), (0, true), (1, false), (1, true)] {
        let cols = if down { 1152usize } else { 2304 };
        let name = match (format, down) {
            (0, false) => "expert_prefill_nvfp4_up",
            (0, true) => "expert_prefill_nvfp4_down",
            (1, false) => "expert_prefill_mxfp8_up",
            _ => "expert_prefill_mxfp8_down",
        };
        let candidate = d.compile(&SOURCE, name)?;
        let count = experts * rows * cols;
        let weights: Vec<u8> = (0..count / if format == 0 { 2 } else { 1 })
            .map(|i| {
                if format == 0 {
                    (i.wrapping_mul(73) + 9) as u8
                } else {
                    [1, 0x38, 0xbc, 0x10, 0xb0, 0x3c, 0x40][i % 7]
                }
            })
            .collect();
        let w = d.upload(&weights)?;
        drop(weights);
        let scales: Vec<u8> = (0..count / if format == 0 { 16 } else { 32 })
            .map(|i| {
                if format == 0 {
                    [0x30, 0x38, 0x40][i % 3]
                } else {
                    [125, 126, 127, 128, 129][i % 5]
                }
            })
            .collect();
        let scales = d.upload(&scales)?;
        let factors = d.upload(bytemuck::cast_slice(&vec![0.5f32; experts]))?;
        let table = d.allocate(if format == 0 { experts * 4096 * 4 } else { 4 })?;
        if format == 0 {
            // SAFETY: complete code/scale lookup table and factor for each expert.
            unsafe {
                production.run(
                    &mut d,
                    "nvfp4_decode_table",
                    &[&table, &factors],
                    Params::new(&[experts], &[]),
                    experts * 4096,
                    128,
                )?;
            }
        }
        for (batch, sparse) in [
            (512usize, false),
            (1024, false),
            (1024, true),
            (2048, false),
        ] {
            let vals: Vec<f32> = (0..batch * cols * if down { top } else { 1 })
                .map(|i| ((i * 23 % 173) as f32 - 86.) / 47.)
                .collect();
            let x = d.upload(bytemuck::cast_slice(&vals))?;
            let routes: Vec<f32> = (0..batch * top)
                .flat_map(|i| {
                    [
                        if sparse {
                            ((i / top * 7 + i % top * 3) % 17) as f32
                        } else {
                            ((i / top * 7 + i % top * 13) % experts) as f32
                        },
                        1. / top as f32,
                    ]
                })
                .collect();
            let routing = d.upload(bytemuck::cast_slice(&routes))?;
            let groups = d.allocate(experts * (batch + 1) * 4)?;
            let max_tiles = (batch * top + 63 * experts) / 64;
            let tiles = d.allocate((1 + 2 * max_tiles) * 4)?;
            let size = batch * top * rows + 64;
            let y = d.allocate(size * 4)?;
            // SAFETY: distinct valid routes and bounded group/tile storage.
            unsafe {
                production.run(
                    &mut d,
                    "expert_groups",
                    &[&groups, &routing],
                    Params::new(&[experts, top, batch], &[]),
                    experts * 32,
                    32,
                )?;
                production.run(
                    &mut d,
                    "expert_tiles",
                    &[&tiles, &groups],
                    Params::new(&[experts, batch, 64], &[]),
                    32,
                    32,
                )?;
            }
            d.synchronize()?;
            let p = Params::new(
                &[rows, cols, top, format, usize::from(down), batch],
                &[7., 9.],
            );
            let mut times = [Vec::new(), Vec::new()];
            let mut expected = Vec::new();
            for trial in 0..5 {
                for j in 0..2 {
                    let index = if trial % 2 == 0 { j } else { 1 - j };
                    if trial == 0 {
                        // SAFETY: initialize the complete output and guard region.
                        unsafe {
                            d.initialize_f32(&y, size, |v| {
                                v.fill(std::mem::MaybeUninit::new(f32::from_bits(u32::MAX)))
                            })?;
                        }
                    }
                    d.reset_stats();
                    for _ in 0..4 {
                        // SAFETY: native dimensions, complete buffers and identical route lists.
                        unsafe {
                            if index == 0 {
                                production.run(
                                    &mut d,
                                    "expert_matmul_prefill",
                                    &[&y, &w, &scales, &factors, &x, &groups, &tiles, &table],
                                    p,
                                    rows.div_ceil(64) * max_tiles * 256,
                                    256,
                                )?;
                            } else {
                                let mut args = std::array::from_fn::<_, 16, _>(|_| Argument::Skip);
                                for (a, b) in args
                                    .iter_mut()
                                    .zip([&y, &w, &scales, &factors, &x, &groups, &tiles, &table])
                                {
                                    *a = Argument::Buffer(b);
                                }
                                args[15] = Argument::Bytes(bytemuck::bytes_of(&p));
                                d.dispatch(
                                    &candidate,
                                    &args,
                                    rows.div_ceil(64) * max_tiles * 256,
                                    256,
                                )?;
                            }
                        }
                    }
                    d.synchronize()?;
                    anyhow::ensure!(d.stats().allocations == 0, "allocated during dispatch");
                    if trial == 0 {
                        let actual = d.read(&y)?;
                        anyhow::ensure!(
                            actual[(size - 64) * 4..].iter().all(|&v| v == 0xff),
                            "output guard overwritten"
                        );
                        if index == 0 {
                            expected = actual;
                        } else {
                            anyhow::ensure!(actual == expected, "specialized differs");
                        }
                    } else {
                        times[index].push(d.stats().gpu_seconds / 4.);
                    }
                }
            }
            for (i, t) in times.iter_mut().enumerate() {
                t.sort_by(f64::total_cmp);
                println!(
                    "{}",
                    serde_json::json!({"format":format,"batch":batch,"sparse":sparse,"specialized":i==1,"down":down,"median_gpu_seconds":(t[1]+t[2])/2.,"trials":t,"exact":true})
                );
            }
        }
    }
    Ok(())
}
#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("requires Metal");
}
