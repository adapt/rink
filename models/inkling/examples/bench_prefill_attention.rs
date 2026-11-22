//! Benchmark-only full-attention sharing across queries and GQA heads.
//! Uses unchanged partition buffers and compares every output byte with production.
#[cfg(target_os = "macos")]
fn main() -> anyhow::Result<()> {
    use rink_core::{Accelerator, Argument, KernelSource, ensure};
    use rink_inkling::kernels::{Params, SOURCE};
    use rink_metal::Metal;
    let mut device = Metal::new()?;
    device.set_profiling(true);
    let source = format!(
        "{}\n{}",
        SOURCE.metal.unwrap(),
        include_str!("prefill_attention.metal")
    );
    let library = KernelSource {
        name: "prefill-attention-probe",
        metal: Some(&source),
        cuda: None,
        subgroup_size: Some(32),
    };
    let mut candidates = vec![(
        "production",
        device.compile(&SOURCE, "attention_partition_queries4")?,
        4,
        1,
    )];
    for (name, queries, heads) in [
        ("attention_q4h1", 4, 1),
        ("attention_q8h1", 8, 1),
        ("attention_q16h1", 16, 1),
        ("attention_q4h3", 4, 3),
        ("attention_q8h3", 8, 3),
    ] {
        candidates.push((name, device.compile(&library, name)?, queries, heads));
    }
    let (qh, kh, dim, rel_dim, extent, fresh) =
        (24usize, 8usize, 128usize, 16usize, 1024usize, 64usize);
    let vals = |n: usize, phase: usize| -> Vec<f32> {
        (0..n)
            .map(|i| (((i * 17 + phase) % 113) as f32 - 56.) / 113.)
            .collect()
    };
    let q = device.upload(bytemuck::cast_slice(&vals(fresh * qh * dim, 13)))?;
    let fresh_k = device.upload(bytemuck::cast_slice(&vals(fresh * kh * dim, 29)))?;
    let fresh_v = device.upload(bytemuck::cast_slice(&vals(fresh * kh * dim, 31)))?;
    let rel = device.upload(bytemuck::cast_slice(&vals(fresh * qh * rel_dim, 37)))?;
    let profiles = device.upload(bytemuck::cast_slice(&vals(rel_dim * extent, 41)))?;
    let mut records = Vec::new();
    for fp16 in [true, false] {
        // Empty history, partial partitions, query offsets/tails, and long history.
        for (length, batch, offset) in [
            (64usize, 31usize, 5usize),
            (1051, 31, 5),
            (8197, 31, 5),
            (1024, 32, 0),
            (8192, 32, 0),
            (65536, 32, 0),
            (262144, 32, 0),
        ] {
            let mut cache = |phase| -> anyhow::Result<_> {
                let data = vals(length * kh * dim, phase);
                if fp16 {
                    let data: Vec<_> = data
                        .into_iter()
                        .map(|v| half::f16::from_f32(v).to_bits())
                        .collect();
                    device.upload(bytemuck::cast_slice(&data))
                } else {
                    device.upload(bytemuck::cast_slice(&data))
                }
            };
            let keys = cache(1)?;
            let values = cache(7)?;
            let parts = length.div_ceil(128);
            let output_len = batch * qh * parts * (dim + 2) + 64;
            // SAFETY: initialized below before any dispatch or readback.
            let partials = unsafe { device.allocate_uninitialized(output_len * 4)? };
            let geometry = Params::new(
                &[
                    qh,
                    kh,
                    dim,
                    length - fresh,
                    length,
                    length,
                    rel_dim,
                    extent,
                    parts,
                    batch,
                    usize::from(fp16),
                    offset,
                ],
                &[],
            );
            let mut reference = Vec::new();
            let mut times = vec![Vec::new(); candidates.len()];
            for trial in 0..5 {
                for i in 0..candidates.len() {
                    let index = if trial % 2 == 0 {
                        i
                    } else {
                        candidates.len() - 1 - i
                    };
                    let (_, kernel, tile, heads) = &candidates[index];
                    let buffers = [
                        &partials, &q, &keys, &values, &fresh_k, &fresh_v, &rel, &profiles,
                    ];
                    let mut args = std::array::from_fn::<_, 16, _>(|_| Argument::Skip);
                    for (a, b) in args.iter_mut().zip(buffers) {
                        *a = Argument::Buffer(b);
                    }
                    args[15] = Argument::Bytes(bytemuck::bytes_of(&geometry));
                    if trial == 0 {
                        // Do not let a candidate inherit output from the reference.
                        // SAFETY: fills every partial and guard element.
                        unsafe {
                            device.initialize_f32(&partials, output_len, |out| {
                                out.fill(std::mem::MaybeUninit::new(f32::from_bits(u32::MAX)));
                            })?;
                        }
                    }
                    device.reset_stats();
                    // SAFETY: native width/GQA ratio, complete cache and fresh tensors,
                    // bounded query offset, full partition output, 256 threads/group.
                    unsafe {
                        device.dispatch(
                            kernel,
                            &args,
                            batch.div_ceil(*tile) * (qh / heads) * parts * 256,
                            256,
                        )?;
                    }
                    device.synchronize()?;
                    ensure!(
                        device.stats().allocations == 0,
                        "attention allocated a device buffer"
                    );
                    if trial > 0 {
                        times[index].push(device.stats().gpu_seconds);
                    } else {
                        let actual = device.read(&partials)?;
                        ensure!(
                            actual[actual.len() - 256..].iter().all(|&b| b == 0xff),
                            "output guard overwritten"
                        );
                        if index == 0 {
                            reference = actual;
                        } else {
                            ensure!(
                                actual == reference,
                                "{} differs at length {length}, fp16 {fp16}, offset {offset}",
                                candidates[index].0
                            );
                        }
                    }
                }
            }
            for (index, (name, _, _, _)) in candidates.iter().enumerate() {
                let mut sorted = times[index].clone();
                sorted.sort_by(f64::total_cmp);
                let record = serde_json::json!({"kernel":name,"history":length,"batch":batch,"offset":offset,"fp16":fp16,"trials_gpu_seconds":times[index],"median_gpu_seconds":(sorted[1]+sorted[2])/2.,"partials_bit_exact":true,"device_buffer_allocations":0});
                eprintln!("{record}");
                records.push(record);
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
