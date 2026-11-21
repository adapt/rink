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
        // Warm pipelines and data before three measurements; report the median GPU
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
        let selection = std::env::args().nth(1).unwrap_or_else(|| "all".into());
        anyhow::ensure!(
            ["all", "dense", "residual", "expert", "attention"].contains(&selection.as_str()),
            "expected all, dense, residual, expert, or attention"
        );
        if selection == "all" || selection == "dense" {
            for (rows, cols) in [
                (24usize, 2304usize),
                (64, 4608),
                (384, 2304),
                (1024, 2304),
                (3072, 2304),
                (2304, 3072),
                (200058, 2304),
            ] {
                let data: Vec<u16> = (0..rows * cols)
                    .map(|i| half::bf16::from_f32(((i % 113) as f32 - 56.) / 113.).to_bits())
                    .collect();
                let w = d.upload(bytemuck::cast_slice(&data))?;
                drop(data);
                let x = upload(&mut d, cols)?;
                let y = d.allocate(rows * 4)?;
                for (name, tile) in [("matvec", 1usize), ("matvec1", 1), ("matvec4", 4)] {
                    for group in [32usize, 128] {
                        let us = unsafe {
                            time(
                                &mut d,
                                &k,
                                name,
                                &[&y, &w, &x],
                                Params::new(&[rows, cols, 1, 1, 0], &[]),
                                rows.div_ceil(tile) * 32,
                                group,
                                if rows > 100000 { 8 } else { 128 },
                            )?
                        };
                        results.push(json!({"kind":"dense","rows":rows,"cols":cols,"kernel":name,"group":group,"gpu_us":us}));
                    }
                }
            }
        }
        if selection == "all" || selection == "residual" {
            for b in [1usize, 3, 6, 12, 24] {
                let n = 2304;
                let bank = upload(&mut d, b * n)?;
                let live = upload(&mut d, n)?;
                let q = upload(&mut d, n)?;
                let y = d.allocate(n * 4)?;
                for name in ["block_read_parallel", "block_read_vector"] {
                    let us = unsafe {
                        time(
                            &mut d,
                            &k,
                            name,
                            &[&y, &bank, &live, &q],
                            Params::new(&[n, b, 0, 0, 1], &[]),
                            256,
                            256,
                            128,
                        )?
                    };
                    results.push(json!({"kind":"residual","blocks":b,"kernel":name,"gpu_us":us}));
                }
            }
        }
        if selection == "all" || selection == "expert" {
            for format in [0usize, 1] {
                for down in [false, true] {
                    let (rows, cols, top, experts) = (
                        2304usize,
                        if down { 1152usize } else { 2304 },
                        8usize,
                        8usize,
                    );
                    let n = rows * cols * experts;
                    let codes: Vec<u8> = (0..if format == 0 { n / 2 } else { n })
                        .map(|i| ((i * 17) % 119) as u8)
                        .collect();
                    let w = d.upload(&codes)?;
                    let scales = d.upload(&vec![
                        if format == 0 { 56 } else { 127 };
                        n / if format == 0 { 16 } else { 32 }
                    ])?;
                    let factors = d.upload(bytemuck::cast_slice(&vec![0.25f32; experts]))?;
                    let x = upload(&mut d, cols * top)?;
                    let routes: Vec<f32> =
                        (0..top).flat_map(|i| [i as f32, 1. / top as f32]).collect();
                    let routing = d.upload(bytemuck::cast_slice(&routes))?;
                    let out_rows = if down { rows } else { rows / 2 };
                    let y = d.allocate(out_rows * top * 4)?;
                    for name in if down {
                        vec!["expert_matvec"]
                    } else {
                        vec!["expert_swiglu_matvec", "expert_swiglu_wide16"]
                    } {
                        let us = unsafe {
                            time(
                                &mut d,
                                &k,
                                name,
                                &[&y, &w, &scales, &factors, &x, &routing],
                                Params::new(
                                    &[rows, cols, top, format, usize::from(down), 1],
                                    &[7., 10.],
                                ),
                                out_rows * top * 32,
                                128,
                                64,
                            )?
                        };
                        results.push(json!({"kind":"expert","format":format,"down":down,"kernel":name,"gpu_us":us}));
                    }
                }
            }
        }
        if selection == "all" || selection == "attention" {
            for capacity in [32usize, 128, 512, 1024, 2048, 4096, 8192, 65536, 262144] {
                let (qh, kh, dim) = (24, 8, 128);
                let data: Vec<u16> = (0..capacity * kh * dim)
                    .map(|i| half::f16::from_f32(((i % 61) as f32 - 30.) / 128.).to_bits())
                    .collect();
                let keys = d.upload(bytemuck::cast_slice(&data))?;
                let values = d.upload(bytemuck::cast_slice(&data))?;
                drop(data);
                for batch in [1usize, 32] {
                    let q = upload(&mut d, batch * qh * dim)?;
                    let fresh = upload(&mut d, batch * kh * dim)?;
                    let rel = upload(&mut d, batch * qh * 16)?;
                    let profiles = upload(&mut d, 16 * 1024)?;
                    let sinks = upload(&mut d, batch * qh)?;
                    let gate = upload(&mut d, batch * qh * dim)?;
                    let parts = capacity.div_ceil(128);
                    let partial = d.allocate(batch * qh * parts * (dim + 2) * 4)?;
                    let y = d.allocate(batch * qh * dim * 4)?;
                    let p = Params::new(
                        &[
                            qh,
                            kh,
                            dim,
                            capacity - batch,
                            capacity,
                            capacity,
                            16,
                            1024,
                            parts,
                            batch,
                            1,
                        ],
                        &[],
                    );
                    for (name, heads) in
                        [("attention_partition", qh), ("attention_partition_gqa", kh)]
                    {
                        let us = unsafe {
                            time(
                                &mut d,
                                &k,
                                name,
                                &[
                                    &partial, &q, &keys, &values, &fresh, &fresh, &rel, &profiles,
                                ],
                                p,
                                batch * heads * parts * 128,
                                128,
                                if capacity >= 65536 { 1 } else { 8 },
                            )?
                        };
                        results.push(json!({"kind":"attention","capacity":capacity,"batch":batch,"kernel":name,"gpu_us":us}));
                    }
                    for (name, group) in
                        [("attention_finish", 32), ("attention_finish_cached", 128)]
                    {
                        let us = unsafe {
                            time(
                                &mut d,
                                &k,
                                name,
                                &[&y, &partial, &sinks, &gate],
                                p,
                                batch * qh * group,
                                group,
                                if capacity >= 65536 { 4 } else { 32 },
                            )?
                        };
                        results.push(json!({"kind":"attention","capacity":capacity,"batch":batch,"kernel":name,"gpu_us":us}));
                    }
                }
            }
        }
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({"device":d.name(),"measurements":results}))?
        );
        Ok(())
    }
}
#[cfg(target_os = "macos")]
fn main() -> anyhow::Result<()> {
    metal::run()
}
#[cfg(not(target_os = "macos"))]
fn main() -> anyhow::Result<()> {
    anyhow::bail!("kernel benchmarks require Metal on macOS")
}
