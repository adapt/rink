//! Exact dense-tile comparisons at native vision dimensions and input offsets.
//! bench_vision_matrices [batch size, default 4097]
#[cfg(target_os = "macos")]
fn main() -> anyhow::Result<()> {
    use rink_core::{Accelerator, Argument};
    use rink_inkling::kernels::{Params, SOURCE};
    use rink_metal::Metal;
    let mut d = Metal::new()?;
    d.set_profiling(true);
    let kernels = [
        d.compile(&SOURCE, "matmul_packed")?,
        d.compile(&SOURCE, "matmul_prefill_m64")?,
    ];
    let batch: usize = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "4097".into())
        .parse()?;
    anyhow::ensure!((1..=65537).contains(&batch), "batch must be in 1..=65537");
    let mut records = Vec::new();
    for (rows, cols, offset) in [
        (1024usize, 1024usize, 0usize),
        (4096, 1024, 0),
        (1024, 4096, 0),
        (1024, 1200, 0),
        (4096, 4096, 1024),
        (2304, 4096, 0),
    ] {
        let weights: Vec<u16> = (0..rows * cols)
            .map(|i| half::bf16::from_f32(((i * 17 % 113) as f32 - 56.) / 37.).to_bits())
            .collect();
        let values: Vec<f32> = (0..batch * cols + offset)
            .map(|i| ((i * 23 % 173) as f32 - 86.) / 47.)
            .collect();
        let w = d.upload(bytemuck::cast_slice(&weights))?;
        let x = d.upload(bytemuck::cast_slice(&values))?;
        let size = batch * rows + 64;
        let y = d.allocate(size * 4)?;
        let mut reference = Vec::new();
        let mut times = [Vec::new(), Vec::new()];
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
                let p = Params::new(&[rows, cols, 1, batch, offset], &[]);
                let mut args = std::array::from_fn::<_, 16, _>(|_| Argument::Skip);
                for (a, b) in args.iter_mut().zip([&y, &w, &x]) {
                    *a = Argument::Buffer(b);
                }
                args[15] = Argument::Bytes(bytemuck::bytes_of(&p));
                let (m, n) = if index == 0 { (16, 32) } else { (64, 64) };
                // SAFETY: BF16 matrices, full F32 inputs including offset and guarded outputs.
                for _ in 0..4 {
                    unsafe {
                        d.dispatch(
                            &kernels[index],
                            &args,
                            rows.div_ceil(n) * batch.div_ceil(m) * 256,
                            256,
                        )?;
                    }
                }
                d.synchronize()?;
                anyhow::ensure!(d.stats().allocations == 0, "dispatch allocated storage");
                if trial == 0 {
                    let actual = d.read(&y)?;
                    anyhow::ensure!(
                        actual[(size - 64) * 4..].iter().all(|&v| v == 0xff),
                        "guard overwritten"
                    );
                    if index == 0 {
                        reference = actual;
                    } else {
                        anyhow::ensure!(actual == reference, "matrix differs at {rows}x{cols}");
                    }
                } else {
                    times[index].push(d.stats().gpu_seconds / 4.);
                }
            }
        }
        for (i, t) in times.iter_mut().enumerate() {
            let mut sorted = t.clone();
            sorted.sort_by(f64::total_cmp);
            records.push(serde_json::json!({"rows":rows,"cols":cols,"batch":batch,"input_offset":offset,"wide_tile":i==1,"median_gpu_seconds":(sorted[1]+sorted[2])/2.,"trials":t,"bit_identical":true,"device_buffer_allocations":0}));
        }
    }
    println!("{}", serde_json::to_string_pretty(&records)?);
    Ok(())
}
#[cfg(not(target_os = "macos"))]
fn main() -> anyhow::Result<()> {
    anyhow::bail!("requires Metal on macOS")
}
