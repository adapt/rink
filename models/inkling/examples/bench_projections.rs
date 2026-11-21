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
        for (rows, cols) in [
            (24usize, 2304usize),
            (64, 4608),
            (128, 2304),
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
            for batch in [1usize, 32] {
                if rows > 100000 && batch > 1 {
                    continue;
                }
                let x = upload(&mut d, cols * batch)?;
                let y = d.allocate(rows * batch * 4)?;
                let cases = if batch == 1 {
                    vec![
                        ("matvec", rows * 32, 128),
                        ("matvec", rows * 32, 32),
                        ("matvec4", rows.div_ceil(4) * 32, 128),
                        ("matvec_packed", rows * 32, 128),
                    ]
                } else {
                    vec![
                        ("matmul", rows.div_ceil(8) * batch.div_ceil(8) * 32, 32),
                        (
                            "matmul_tiled",
                            rows.div_ceil(32) * batch.div_ceil(16) * 256,
                            256,
                        ),
                    ]
                };
                for (name, threads, group) in cases {
                    // SAFETY: dimensions and dispatches follow each projection ABI.
                    let us = unsafe {
                        time(
                            &mut d,
                            &k,
                            name,
                            &[&y, &w, &x],
                            Params::new(&[rows, cols, 1, batch, 0], &[]),
                            threads,
                            group,
                            if rows > 100000 { 8 } else { 128 },
                        )?
                    };
                    results.push(json!({"kind":"dense","rows":rows,"cols":cols,"batch":batch,"kernel":name,"group":group,"gpu_us":us}));
                }
            }
        }
        for format in [0usize, 1] {
            for down in [false, true] {
                let (experts, top, batch) = (128usize, 8usize, 32usize);
                let (rows, cols) = if down { (2304, 1152) } else { (2304, 2304) };
                let count = experts * rows * cols;
                let w = d.upload(&vec![0x21; if format == 0 { count / 2 } else { count }])?;
                let scales = d.upload(&vec![
                    if format == 0 { 0x38 } else { 127 };
                    count / if format == 0 { 16 } else { 32 }
                ])?;
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
                // SAFETY: routes select eight distinct valid experts per token;
                // group and tile allocations cover all routes and padding.
                unsafe {
                    k.run(
                        &mut d,
                        "expert_groups",
                        &[&slots, &routing],
                        Params::new(&[experts, top, batch], &[]),
                        experts * 32,
                        32,
                    )?;
                    k.run(
                        &mut d,
                        "expert_tiles",
                        &[&tiles, &slots],
                        Params::new(&[experts, batch], &[]),
                        32,
                        32,
                    )?;
                }
                d.synchronize()?;
                for (name, row_tile, group) in [
                    ("expert_matmul", 8usize, 32usize),
                    ("expert_matmul_tiled", 32, 128),
                ] {
                    // SAFETY: native weight blocks and shared/per-route inputs
                    // have the lengths required by each expert projection.
                    let us = unsafe {
                        time(
                            &mut d,
                            &k,
                            name,
                            &[&y, &w, &scales, &factors, &x, &slots, &tiles],
                            Params::new(&[rows, cols, top, format, usize::from(down), batch], &[]),
                            rows.div_ceil(row_tile) * max_tiles * group,
                            group,
                            16,
                        )?
                    };
                    results.push(json!({"kind":"expert","format":format,"down":down,"rows":rows,"cols":cols,"batch":batch,"kernel":name,"gpu_us":us}));
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
    anyhow::bail!("requires Metal on macOS")
}
