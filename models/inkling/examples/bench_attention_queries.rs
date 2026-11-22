//! Compare full-attention KV reuse at native model dimensions, without a checkpoint.
//! These are isolated kernel timings, not full-model throughput measurements.
#[cfg(target_os = "macos")]
fn main() -> anyhow::Result<()> {
    use rink_core::Accelerator;
    use rink_inkling::kernels::{Kernels, Params};
    use rink_metal::Metal;
    let mut device = Metal::new()?;
    device.set_profiling(true);
    let kernels = Kernels::compile(&mut device)?;
    let (qh, kh, dim, rel_dim, extent, batch) = (24, 8, 128, 16, 1024, 32);
    let vals = |n: usize, phase: usize| -> Vec<f32> {
        (0..n)
            .map(|i| (((i * 17 + phase) % 113) as f32 - 56.) / 113.)
            .collect()
    };
    let q = device.upload(bytemuck::cast_slice(&vals(batch * qh * dim, 13)))?;
    let fresh_k = device.upload(bytemuck::cast_slice(&vals(batch * kh * dim, 29)))?;
    let fresh_v = device.upload(bytemuck::cast_slice(&vals(batch * kh * dim, 31)))?;
    let rel = device.upload(bytemuck::cast_slice(&vals(batch * qh * rel_dim, 37)))?;
    let profiles = device.upload(bytemuck::cast_slice(&vals(rel_dim * extent, 41)))?;
    let mut records = Vec::new();
    for length in [4096usize, 16384, 65536, 131072, 262144] {
        let cache: Vec<_> = vals(length * kh * dim, 1)
            .into_iter()
            .map(|v| half::f16::from_f32(v).to_bits())
            .collect();
        let keys = device.upload(bytemuck::cast_slice(&cache))?;
        // Distinct allocations avoid a cache hit from aliasing keys and values.
        let values = device.upload(bytemuck::cast_slice(&cache))?;
        drop(cache);
        let parts = length.div_ceil(128);
        let partials = device.allocate(batch * qh * parts * (dim + 2) * 4)?;
        let geometry = Params::new(
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
                1,
            ],
            &[],
        );
        let candidates = [
            ("attention_partition_prefetch", 256, 1, qh),
            ("attention_partition_queries4", 256, 4, qh),
        ];
        let mut reference = Vec::new();
        let mut times = [Vec::new(), Vec::new()];
        for trial in 0..5 {
            for i in 0..2 {
                let index = if trial % 2 == 0 { i } else { 1 - i };
                let (name, threads, tile, heads) = candidates[index];
                device.reset_stats();
                // SAFETY: native full-attention geometry, distinct complete KV
                // buffers, complete fresh rows and a full partition workspace.
                unsafe {
                    kernels.run(
                        &mut device,
                        name,
                        &[
                            &partials, &q, &keys, &values, &fresh_k, &fresh_v, &rel, &profiles,
                        ],
                        geometry,
                        batch.div_ceil(tile) * heads * parts * threads,
                        threads,
                    )?;
                }
                device.synchronize()?;
                if trial > 0 {
                    times[index].push(device.stats().gpu_seconds);
                } else {
                    let actual = device.read(&partials)?;
                    if index == 0 {
                        reference = actual;
                    } else {
                        anyhow::ensure!(
                            actual == reference,
                            "attention partials differ at {length}"
                        );
                    }
                }
            }
        }
        for (index, (name, _, _, _)) in candidates.iter().enumerate() {
            let mut sorted = times[index].clone();
            sorted.sort_by(f64::total_cmp);
            let record = serde_json::json!({"kernel": name, "history": length, "batch": batch, "fp16": true, "trials_gpu_seconds": times[index], "median_gpu_seconds": (sorted[1] + sorted[2]) / 2., "partials_bit_exact": true});
            eprintln!("{record}");
            records.push(record);
        }
        drop(reference);
        let sinks = device.upload(bytemuck::cast_slice(&vals(batch * qh, 43)))?;
        let gate = device.upload(bytemuck::cast_slice(&vals(batch * qh * dim, 47)))?;
        let output = device.allocate(batch * qh * dim * 4)?;
        let mut reference = Vec::new();
        let mut times = [Vec::new(), Vec::new()];
        let candidates = [("attention_finish", 32), ("attention_finish_cached", 128)];
        for trial in 0..5 {
            for i in 0..2 {
                let index = if trial % 2 == 0 { i } else { 1 - i };
                let (name, threads) = candidates[index];
                device.reset_stats();
                // SAFETY: at most 2048 partitions, native dimensions and complete buffers.
                unsafe {
                    kernels.run(
                        &mut device,
                        name,
                        &[&output, &partials, &sinks, &gate],
                        geometry,
                        batch * qh * threads,
                        threads,
                    )?;
                }
                device.synchronize()?;
                if trial > 0 {
                    times[index].push(device.stats().gpu_seconds);
                } else {
                    let actual = device.read(&output)?;
                    if index == 0 {
                        reference = actual;
                    } else {
                        anyhow::ensure!(
                            actual == reference,
                            "attention finish differs at {length}"
                        );
                    }
                }
            }
        }
        for (index, (name, _)) in candidates.iter().enumerate() {
            let mut sorted = times[index].clone();
            sorted.sort_by(f64::total_cmp);
            let record = serde_json::json!({"kernel": name, "history": length, "batch": batch, "fp16": true, "trials_gpu_seconds": times[index], "median_gpu_seconds": (sorted[1] + sorted[2]) / 2., "outputs_bit_exact": true});
            eprintln!("{record}");
            records.push(record);
        }
    }
    println!("{}", serde_json::to_string_pretty(&records)?);
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn main() -> anyhow::Result<()> {
    anyhow::bail!("requires Metal on macOS")
}
