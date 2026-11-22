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
        // Four warm-up trials precede three measurements; report the median GPU
        // command-buffer time, without per-dispatch counter instrumentation.
        let mut times = Vec::new();
        for trial in 0..7 {
            d.reset_stats();
            for _ in 0..repeats {
                unsafe {
                    k.run(d, name, buffers, p, threads, group)?;
                }
            }
            d.synchronize()?;
            if trial > 3 {
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
        let experts_only = std::env::args().any(|s| s == "--experts-only");
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
            if experts_only {
                continue;
            }
            let data: Vec<u16> = (0..rows * cols)
                .map(|i| half::bf16::from_f32(((i % 113) as f32 - 56.) / 113.).to_bits())
                .collect();
            let w = d.upload(bytemuck::cast_slice(&data))?;
            drop(data);
            for batch in [1usize, 2, 8, 16, 32] {
                if rows > 100000 && batch > 1 {
                    continue;
                }
                let x = upload(&mut d, cols * batch)?;
                let y = d.allocate(rows * batch * 4)?;
                let mut cases = if batch == 1 {
                    vec![
                        ("matvec", rows * 32, 128),
                        ("matvec", rows * 32, 32),
                        ("matvec_prefetch", rows * 32, 32),
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
                        (
                            "matmul_packed",
                            rows.div_ceil(32) * batch.div_ceil(16) * 256,
                            256,
                        ),
                    ]
                };
                if batch == 1 {
                    let native = match (rows < 128, cols) {
                        (true, 2304) => Some("matvec_scalar_bf16_2304"),
                        (true, 4608) => Some("matvec_scalar_bf16_4608"),
                        (false, 2304) => Some("matvec_bf16_2304"),
                        (false, 3072) => Some("matvec_bf16_3072"),
                        _ => None,
                    };
                    if let Some(name) = native {
                        cases.push((name, rows * 32, if rows < 128 { 32 } else { 128 }));
                    }
                }
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
        for batch in [1usize, 32] {
            if experts_only {
                continue;
            }
            let width = 2304usize;
            for blocks in [1usize, 7, 13, 25] {
                let bank = upload(&mut d, blocks * batch * width)?;
                let live = upload(&mut d, batch * width)?;
                let query = upload(&mut d, width)?;
                let y = d.allocate(batch * width * 4)?;
                for (name, shards) in [
                    ("block_read_parallel", 1usize),
                    ("block_read_prefetch", 1),
                    ("block_read_prefetch", 4),
                    ("block_read_prefetch", 8),
                ] {
                    // SAFETY: the bank contains every saved block, and all
                    // activation/query buffers cover the requested dimensions.
                    let us = unsafe {
                        time(
                            &mut d,
                            &k,
                            name,
                            &[&y, &bank, &live, &query],
                            Params::new(&[width, blocks, 0, 0, batch, shards], &[]),
                            batch * shards * 256,
                            256,
                            128,
                        )?
                    };
                    results.push(json!({"kind":"residual","shards":shards,"width":width,"blocks":blocks,"batch":batch,"kernel":name,"gpu_us":us}));
                }
            }
        }
        for format in [0usize, 1] {
            for down in [false, true] {
                let (experts, top, batch) = (128usize, 8usize, 32usize);
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
                let out_rows = if down { rows } else { rows / 2 };
                let reference = if down {
                    "expert_matvec"
                } else {
                    "expert_swiglu_matvec"
                };
                let candidate = match (format, down) {
                    (0, false) => "expert_swiglu_nvfp4",
                    (0, true) => "expert_matvec_nvfp4",
                    (_, false) => "expert_swiglu_mxfp8_native",
                    (_, true) => "expert_matvec_mxfp8_native",
                };
                let mut expected = None;
                for name in [reference, candidate] {
                    let p =
                        Params::new(&[rows, cols, top, format, usize::from(down), 1], &[4., 25.]);
                    let buffers = [&y, &w, &scales, &factors, &x, &routing, &table];
                    // SAFETY: first-token routes and inputs cover the native decode geometry.
                    unsafe {
                        k.run(&mut d, name, &buffers, p, out_rows * top * 32, 128)?;
                    }
                    let bytes = d.read_prefix(&y, out_rows * top * 4)?;
                    if let Some(expected) = &expected {
                        assert_eq!(&bytes, expected, "{name}");
                    } else {
                        expected = Some(bytes);
                    }
                    // SAFETY: same checked decode buffers and dispatch as above.
                    let us = unsafe {
                        time(&mut d, &k, name, &buffers, p, out_rows * top * 32, 128, 256)?
                    };
                    results.push(json!({"kind":"expert","format":format,"down":down,"rows":rows,"cols":cols,"batch":1,"kernel":name,"gpu_us":us}));
                }
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
                let mut cases = vec![
                    ("expert_matmul", 8usize, 32usize),
                    ("expert_matmul_tiled", 32, 128),
                ];
                if format == 0 {
                    cases.push(("expert_matmul_nvfp4", 32, 128));
                }
                let mut grouped_reference = None;
                for (name, row_tile, group) in cases {
                    // SAFETY: native weight blocks and shared/per-route inputs
                    // have the lengths required by each expert projection.
                    let us = unsafe {
                        time(
                            &mut d,
                            &k,
                            name,
                            &[&y, &w, &scales, &factors, &x, &slots, &tiles, &table],
                            Params::new(&[rows, cols, top, format, usize::from(down), batch], &[]),
                            rows.div_ceil(row_tile) * max_tiles * group,
                            group,
                            16,
                        )?
                    };
                    let bytes = d.read(&y)?;
                    if let Some(reference) = &grouped_reference {
                        assert_eq!(&bytes, reference, "{name}");
                    } else {
                        grouped_reference = Some(bytes);
                    }
                    results.push(json!({"kind":"expert","format":format,"down":down,"rows":rows,"cols":cols,"batch":batch,"kernel":name,"gpu_us":us}));
                }
            }
        }
        if experts_only {
            println!("{}", serde_json::to_string_pretty(&results)?);
            return Ok(());
        }
        for fp16 in [false, true] {
            for length in [32usize, 64, 128, 256, 384, 512, 1024, 8192, 16384] {
                let (qh, kh, dim, rel_dim, extent) =
                    (24usize, 8usize, 128usize, 16usize, 1024usize);
                let cache_data: Vec<f32> = (0..length * kh * dim)
                    .map(|i| ((i * 17 % 113) as f32 - 56.) / 113.)
                    .collect();
                let cache = if fp16 {
                    let h: Vec<u16> = cache_data
                        .iter()
                        .map(|&v| half::f16::from_f32(v).to_bits())
                        .collect();
                    d.upload(bytemuck::cast_slice(&h))?
                } else {
                    d.upload(bytemuck::cast_slice(&cache_data))?
                };
                let q = upload(&mut d, qh * dim)?;
                let fresh = upload(&mut d, kh * dim)?;
                let rel = upload(&mut d, qh * rel_dim)?;
                let profiles = upload(&mut d, rel_dim * extent)?;
                let parts = length.div_ceil(128);
                let y = d.allocate(qh * parts * (dim + 2) * 4)?;
                for (name, group) in [
                    ("attention_partition", 128usize),
                    ("attention_partition_wide", 512),
                    ("attention_partition_prefetch", 256),
                ] {
                    // SAFETY: native head dimensions, cache length, and one fresh token
                    // match the partition ABI for each tested threadgroup size.
                    let us = unsafe {
                        time(
                            &mut d,
                            &k,
                            name,
                            &[&y, &q, &cache, &cache, &fresh, &fresh, &rel, &profiles],
                            Params::new(
                                &[
                                    qh,
                                    kh,
                                    dim,
                                    length - 1,
                                    length,
                                    length,
                                    rel_dim,
                                    extent,
                                    parts,
                                    1,
                                    usize::from(fp16),
                                ],
                                &[],
                            ),
                            qh * parts * group,
                            group,
                            128,
                        )?
                    };
                    results.push(json!({"kind":"attention","fp16":fp16,"length":length,"kernel":name,"group":group,"gpu_us":us}));
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
