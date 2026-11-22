//! Compare image and prefill attention schedules without a checkpoint.
//! Timings use command buffers, without per-kernel instrumentation.
//! bench_prefill_kernels [vision-sequence-length] [--vision-only] (default: 4097)
#[cfg(target_os = "macos")]
fn main() -> anyhow::Result<()> {
    use rink_core::Accelerator;
    use rink_inkling::kernels::{Kernels, Params};
    use rink_metal::{Metal, MetalBuffer};
    let mut d = Metal::new()?;
    d.set_profiling(true);
    let kernels = Kernels::compile(&mut d)?;
    let mut upload = |n, shift| -> anyhow::Result<MetalBuffer> {
        let values: Vec<f32> = (0..n)
            .map(|i| (((i * 17 + shift) % 113) as f32 - 56.) / 37.)
            .collect();
        d.upload(bytemuck::cast_slice(&values))
    };
    let n: usize = std::env::args()
        .nth(1)
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(4097);
    anyhow::ensure!(n > 0 && n <= 65537, "sequence must be in 1..=65537");
    let q = upload(n * 16 * 64, 1)?;
    let k = upload(n * 16 * 64, 13)?;
    let v = upload(n * 16 * 64, 37)?;
    let y = d.allocate(n * 16 * 64 * 4)?;
    let mut reference = Vec::new();
    for (name, group, tile) in [
        ("vision_attention", 32, 8),
        ("vision_attention_64x16", 256, 16),
        ("vision_attention_64x32", 256, 32),
    ] {
        let mut times = Vec::new();
        for trial in 0..4 {
            d.reset_stats();
            // SAFETY: 16 heads of width 64, complete Q/K/V and output buffers.
            unsafe {
                kernels.run(
                    &mut d,
                    name,
                    &[&y, &q, &k, &v],
                    Params::new(&[n, 16, 64], &[]),
                    n.div_ceil(tile) * 16 * group,
                    group,
                )?;
            }
            d.synchronize()?;
            if trial > 0 {
                times.push(d.stats().gpu_seconds);
            }
        }
        let actual = d.read(&y)?;
        if reference.is_empty() {
            reference = actual;
        } else {
            anyhow::ensure!(
                actual == reference,
                "vision attention differs from reference"
            );
        }
        times.sort_by(f64::total_cmp);
        println!(
            "{}",
            serde_json::json!({"kernel":name,"sequence":n,"median_gpu_seconds":times[1],"trials":times})
        );
    }
    drop((q, k, v, y, reference));
    if std::env::args().nth(2).as_deref() == Some("--vision-only") {
        return Ok(());
    }
    for fp16 in [false, true] {
        for length in [128usize, 1024, 8192, 10342] {
            let (qh, kh, dim, rel_dim, extent, batch) = (24, 8, 128, 16, 1024, 32);
            let values = |n, shift| -> Vec<f32> {
                (0..n)
                    .map(|i| (((i * 17 + shift) % 113) as f32 - 56.) / 113.)
                    .collect()
            };
            let cache = values(length * kh * dim, 1);
            let cache = if fp16 {
                d.upload(bytemuck::cast_slice(
                    &cache
                        .iter()
                        .map(|&v| half::f16::from_f32(v).to_bits())
                        .collect::<Vec<_>>(),
                ))?
            } else {
                d.upload(bytemuck::cast_slice(&cache))?
            };
            let q = d.upload(bytemuck::cast_slice(&values(batch * qh * dim, 13)))?;
            let fresh = d.upload(bytemuck::cast_slice(&values(batch * kh * dim, 29)))?;
            let rel = d.upload(bytemuck::cast_slice(&values(batch * qh * rel_dim, 37)))?;
            let profiles = d.upload(bytemuck::cast_slice(&values(rel_dim * extent, 41)))?;
            let parts = length.div_ceil(128);
            let y = d.allocate(batch * qh * parts * (dim + 2) * 4)?;
            let p = Params::new(
                &[
                    qh,
                    kh,
                    dim,
                    length - batch,
                    length,
                    length,
                    rel_dim,
                    extent,
                    parts,
                    batch,
                    usize::from(fp16),
                ],
                &[],
            );
            let mut reference = Vec::new();
            for (name, group) in [
                ("attention_partition", 128),
                ("attention_partition_prefetch", 256),
                ("attention_partition_wide", 512),
            ] {
                let mut times = Vec::new();
                for trial in 0..4 {
                    d.reset_stats();
                    for _ in 0..8 {
                        // SAFETY: native geometry and complete caches, fresh chunk,
                        // relative position inputs and partition output workspace.
                        unsafe {
                            kernels.run(
                                &mut d,
                                name,
                                &[&y, &q, &cache, &cache, &fresh, &fresh, &rel, &profiles],
                                p,
                                batch * qh * parts * group,
                                group,
                            )?;
                        }
                    }
                    d.synchronize()?;
                    if trial > 0 {
                        times.push(d.stats().gpu_seconds / 8.);
                    }
                }
                let actual = d.read(&y)?;
                if reference.is_empty() {
                    reference = actual;
                } else {
                    anyhow::ensure!(
                        actual == reference,
                        "prefill attention differs from reference"
                    );
                }
                times.sort_by(f64::total_cmp);
                println!(
                    "{}",
                    serde_json::json!({"kernel":name,"sequence":length,"batch":batch,"fp16":fp16,"median_gpu_seconds":times[1],"trials":times})
                );
            }
        }
    }
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn main() -> anyhow::Result<()> {
    anyhow::bail!("requires Metal on macOS")
}
