//! Exact convolution/state comparison and native prefill timings, including commit.
#[cfg(target_os = "macos")]
fn main() -> anyhow::Result<()> {
    use rink_core::{Accelerator, ensure};
    use rink_inkling::kernels::{Kernels, Params};
    use rink_metal::Metal;
    let mut d = Metal::new()?;
    d.set_profiling(true);
    let kernels = Kernels::compile(&mut d)?;
    let vals = |n: usize, phase: usize| -> Vec<f32> {
        (0..n)
            .map(|i| ((i * 17 + phase) % 113) as f32 / 113. - 0.5)
            .collect()
    };
    let mut records = Vec::new();
    for channels in [1024usize, 2304] {
        for batch in [8usize, 128, 1024] {
            for dynamic in [false, true] {
                let (taps, dim, position) = (4usize, 128usize, 4099usize);
                let x = d.upload(bytemuck::cast_slice(&vals(channels * batch, 3)))?;
                let w = d.upload(bytemuck::cast_slice(&vals(channels * taps, 5)))?;
                let dyn_w = d.upload(bytemuck::cast_slice(&vals(
                    batch * channels / dim * taps,
                    7,
                )))?;
                let history = vals(channels * taps, 11);
                let state = d.upload(bytemuck::cast_slice(&history))?;
                let y = d.allocate((channels * batch + 64) * 4)?;
                let p = Params::new(
                    &[channels, taps, position, usize::from(dynamic), dim, batch],
                    &[],
                );
                let mut reference = Vec::new();
                let mut times = [Vec::new(), Vec::new()];
                for round in 0..5 {
                    for step in 0..2 {
                        let i = if round % 2 == 0 { step } else { 1 - step };
                        d.write(&state, 0, bytemuck::cast_slice(&history))?;
                        // SAFETY: initialize the full output and guard before each check.
                        unsafe {
                            d.initialize_f32(&y, channels * batch + 64, |out| {
                                out.fill(std::mem::MaybeUninit::new(f32::from_bits(u32::MAX)))
                            })?;
                        }
                        d.reset_stats();
                        // SAFETY: independent complete tensors; old history is committed
                        // only after the parallel convolution finishes reading it.
                        unsafe {
                            kernels.run(
                                &mut d,
                                if i == 0 {
                                    "convolution"
                                } else {
                                    "convolution_prefill"
                                },
                                &[&y, &x, &w, &state, &dyn_w],
                                p,
                                channels * if i == 0 { 1 } else { batch },
                                128,
                            )?;
                            if i == 1 {
                                kernels.run(
                                    &mut d,
                                    "convolution_commit",
                                    &[&state, &x],
                                    Params::new(&[channels, taps, position, batch], &[]),
                                    channels * taps.min(batch),
                                    128,
                                )?;
                            }
                        }
                        d.synchronize()?;
                        ensure!(d.stats().allocations == 0, "new device allocation");
                        if round > 0 {
                            times[i].push(d.stats().gpu_seconds);
                        } else {
                            let mut actual = d.read(&y)?;
                            ensure!(
                                actual[channels * batch * 4..].iter().all(|&x| x == 0xff),
                                "guard changed"
                            );
                            actual.extend(d.read(&state)?);
                            if i == 0 {
                                reference = actual;
                            } else {
                                ensure!(actual == reference, "convolution or state differs");
                            }
                        }
                    }
                }
                for (i, name) in ["convolution", "convolution_prefill_and_commit"]
                    .iter()
                    .enumerate()
                {
                    let mut sorted = times[i].clone();
                    sorted.sort_by(f64::total_cmp);
                    let record = serde_json::json!({"kernel":name,"channels":channels,"batch":batch,"dynamic":dynamic,"trials_gpu_seconds":times[i],"median_gpu_seconds":(sorted[1]+sorted[2])/2.,"bit_identical":true,"device_buffer_allocations":0});
                    eprintln!("{record}");
                    records.push(record);
                }
            }
        }
    }
    println!("{}", serde_json::to_string_pretty(&records)?);
    Ok(())
}
#[cfg(not(target_os = "macos"))]
fn main() -> anyhow::Result<()> {
    anyhow::bail!("requires Metal on macOS")
}
