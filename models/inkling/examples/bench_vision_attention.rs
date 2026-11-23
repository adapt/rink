//! Exact vision-attention comparison with alternating timing order.
//! bench_vision_attention [comma-separated sequence lengths, including CLS]
#[cfg(target_os = "macos")]
fn main() -> anyhow::Result<()> {
    use rink_core::{Accelerator, Argument};
    use rink_inkling::kernels::{Params, SOURCE};
    use rink_metal::Metal;
    let mut d = Metal::new()?;
    d.set_profiling(true);
    let sizes: Vec<usize> = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "129,1025,2047,2048,2049,4097,16385,41273".into())
        .split(',')
        .map(str::parse)
        .collect::<Result<_, _>>()?;
    let mut kernels = Vec::new();
    for name in [
        "vision_attention_64x16",
        "vision_attention_64x32",
        "vision_attention_64x64",
    ] {
        kernels.push(d.compile(&SOURCE, name)?);
    }
    let mut records = Vec::new();
    for n in sizes {
        anyhow::ensure!(n > 0 && n <= 65537, "sequence length must be in 1..=65537");
        let len = n * 16 * 64;
        let mut upload = |shift| {
            let values: Vec<f32> = (0..len)
                .map(|i| (((i * 17 + shift) % 113) as f32 - 56.) / 37.)
                .collect();
            d.upload(bytemuck::cast_slice(&values))
        };
        let q = upload(1)?;
        let k = upload(13)?;
        let v = upload(37)?;
        let out = d.allocate((len + 64) * 4)?;
        let baseline = usize::from(n >= 32768);
        let tiles = [if baseline == 0 { 16 } else { 32 }, 64];
        let candidates = [&kernels[baseline], &kernels[2]];
        let mut reference = Vec::new();
        let mut times = [Vec::new(), Vec::new()];
        for trial in 0..5 {
            for j in 0..2 {
                let index = if trial % 2 == 0 { j } else { 1 - j };
                if trial == 0 {
                    // SAFETY: initialize every output and guard before each comparison.
                    unsafe {
                        d.initialize_f32(&out, len + 64, |v| {
                            v.fill(std::mem::MaybeUninit::new(f32::from_bits(u32::MAX)))
                        })?;
                    }
                }
                d.reset_stats();
                let p = Params::new(&[n, 16, 64], &[]);
                let mut args = std::array::from_fn::<_, 16, _>(|_| Argument::Skip);
                for (a, b) in args.iter_mut().zip([&out, &q, &k, &v]) {
                    *a = Argument::Buffer(b);
                }
                args[15] = Argument::Bytes(bytemuck::bytes_of(&p));
                // SAFETY: complete 64-wide heads, partial query/key tiles, guarded output.
                unsafe {
                    d.dispatch(
                        candidates[index],
                        &args,
                        n.div_ceil(tiles[index]) * 16 * 256,
                        256,
                    )?;
                }
                d.synchronize()?;
                anyhow::ensure!(d.stats().allocations == 0, "dispatch allocated storage");
                if trial == 0 {
                    let actual = d.read(&out)?;
                    anyhow::ensure!(
                        actual[len * 4..].iter().all(|&v| v == 0xff),
                        "guard overwritten"
                    );
                    if index == 0 {
                        reference = actual;
                    } else {
                        anyhow::ensure!(actual == reference, "attention differs at {n}");
                    }
                } else {
                    times[index].push(d.stats().gpu_seconds);
                }
            }
        }
        for (i, t) in times.iter_mut().enumerate() {
            let mut sorted = t.clone();
            sorted.sort_by(f64::total_cmp);
            let record = serde_json::json!({"sequence":n,"query_tile":tiles[i],"median_gpu_seconds":(sorted[1]+sorted[2])/2.,"trials":t,"bit_identical":true,"device_buffer_allocations":0});
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
