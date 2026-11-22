//! Isolated kernel experiments. Repeated weights can remain cached; use `rink bench`
//! to establish end-to-end improvements. No checkpoint is needed.
#[cfg(target_os = "macos")]
mod metal {
    use rink_core::{Accelerator, Result};
    use rink_inkling::kernels::{Kernels, Params};
    use rink_metal::{Metal, MetalBuffer};
    use serde_json::json;

    fn upload(d: &mut Metal, n: usize) -> Result<MetalBuffer> {
        let values: Vec<f32> = (0..n)
            .map(|i| ((i * 17 % 113) as f32 - 56.) / 113.)
            .collect();
        d.upload(bytemuck::cast_slice(&values))
    }

    // SAFETY: callers provide shapes and buffers matching the named kernel's ABI.
    #[allow(clippy::too_many_arguments)]
    unsafe fn time(
        d: &mut Metal,
        k: &Kernels<Metal>,
        name: &str,
        buffers: &[&MetalBuffer],
        p: Params,
        threads: usize,
        group: usize,
        repeats: usize,
    ) -> Result<f64> {
        // One warm-up trial precedes three measurements; report the median GPU
        // command-buffer time, without per-dispatch counter instrumentation.
        let mut times = Vec::new();
        for trial in 0..4 {
            d.reset_stats();
            for _ in 0..repeats {
                unsafe {
                    k.run(d, name, buffers, p, threads, group)?;
                }
            }
            d.synchronize()?;
            if trial > 0 {
                times.push(d.stats().gpu_seconds * 1e6 / repeats as f64);
            }
        }
        times.sort_by(f64::total_cmp);
        Ok(times[1])
    }

    pub fn run() -> Result<()> {
        let mut d = Metal::new()?;
        d.set_profiling(true);
        let k = Kernels::compile(&mut d)?;
        let mut results = Vec::new();
        for format in [0usize, 1] {
            for down in [false, true] {
                for batch in [128usize, 256, 512, 1024, 2048] {
                    let (experts, top) = (128usize, 8usize);
                    let (rows, cols) = if down { (2304, 1152) } else { (2304, 2304) };
                    let count = experts * rows * cols;
                    // Vary codes and scales: a single repeated pair would give the
                    // decode table unrealistically favorable cache locality.
                    let codes: Vec<u8> = (0..if format == 0 { count / 2 } else { count })
                        .map(|i| {
                            if format == 0 {
                                (i.wrapping_mul(73) + 9) as u8
                            } else {
                                [0x01, 0x38, 0xbc, 0x10, 0xb0, 0x3c, 0x40][i % 7]
                            }
                        })
                        .collect();
                    let w = d.upload(&codes)?;
                    drop(codes);
                    let scale_codes: Vec<u8> = (0..count / if format == 0 { 16 } else { 32 })
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
                    let factors = upload(&mut d, experts)?;
                    let routes: Vec<f32> = (0..batch * top)
                        .flat_map(|i| {
                            [
                                ((i / top * 7 + i % top * 13) % experts) as f32,
                                1. / top as f32,
                            ]
                        })
                        .collect();
                    let routing = d.upload(bytemuck::cast_slice(&routes))?;
                    let max_tiles = (batch * top + 7 * experts) / 8;
                    let slots = d.allocate(experts * (batch + 1) * 4)?;
                    let tiles = d.allocate((1 + 2 * max_tiles) * 4)?;
                    let x = upload(&mut d, batch * cols * if down { top } else { 1 })?;
                    let y = d.allocate(batch * top * rows * 4)?;
                    let table = d.allocate(if format == 0 { experts * 4096 * 4 } else { 4 })?;
                    if format == 0 {
                        // SAFETY: every expert has one factor and 4096 table entries.
                        unsafe {
                            k.run(
                                &mut d,
                                "nvfp4_decode_table",
                                &[&table, &factors],
                                Params::new(&[experts], &[]),
                                experts * 4096,
                                128,
                            )?;
                        }
                    }

                    // SAFETY: the routing contains distinct valid expert indices.
                    unsafe {
                        k.run(
                            &mut d,
                            "expert_groups",
                            &[&slots, &routing],
                            Params::new(&[experts, top, batch], &[]),
                            experts * 32,
                            32,
                        )?;
                    }
                    let reference = if format == 0 {
                        "expert_matmul_nvfp4"
                    } else {
                        "expert_matmul_tiled"
                    };
                    let mut expected = None;
                    for (name, tile_size, group) in [
                        (reference, 8usize, 128usize),
                        ("expert_matmul_m32k32", 32, 256),
                    ] {
                        let max_tiles = (batch * top + (tile_size - 1) * experts) / tile_size;
                        // SAFETY: the tile allocation covers the worst case eight-slot
                        // list and all candidate lists use no more space.
                        unsafe {
                            k.run(
                                &mut d,
                                "expert_tiles",
                                &[&tiles, &slots],
                                Params::new(&[experts, batch, tile_size], &[]),
                                32,
                                32,
                            )?;
                        }
                        d.synchronize()?;
                        // SAFETY: native weight blocks, expert slots, inputs, and output
                        // cover every index in this candidate's dispatch geometry.
                        let us = unsafe {
                            time(
                                &mut d,
                                &k,
                                name,
                                &[&y, &w, &scales, &factors, &x, &slots, &tiles, &table],
                                Params::new(
                                    &[rows, cols, top, format, usize::from(down), batch],
                                    &[],
                                ),
                                rows.div_ceil(32) * max_tiles * group,
                                group,
                                4,
                            )?
                        };
                        let bytes = d.read(&y)?;
                        if let Some(expected) = &expected {
                            assert!(
                                &bytes == expected,
                                "{name}: format={format} down={down} batch={batch}"
                            );
                        } else {
                            expected = Some(bytes);
                        }
                        eprintln!("{name} format={format} down={down} batch={batch}: {us:.1} us");
                        results.push(json!({"format":format,"down":down,"batch":batch,"kernel":name,"gpu_us":us}));
                    }
                }
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
