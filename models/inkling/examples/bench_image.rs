//! Fixed-work image benchmark and exact continuation-logit regression check.
//! bench_image record|compare MODEL IMAGE SNAPSHOT [profile] [32|64|128]
//! Snapshots contain private image-derived data; keep them outside the repository.
#[cfg(target_os = "macos")]
fn main() -> anyhow::Result<()> {
    use anyhow::ensure;
    use rink_core::{AutoregressiveModel, greedy};
    use rink_inkling::{Image, Inkling, KvCachePrecision, PrefillProgress, image::image_prompt};
    use rink_metal::Metal;
    use std::{path::Path, time::Instant};

    let args: Vec<_> = std::env::args().collect();
    ensure!(
        (5..=7).contains(&args.len()),
        "usage: bench_image record|compare MODEL IMAGE SNAPSHOT [profile] [32|64|128]"
    );
    let record = match args[1].as_str() {
        "record" => true,
        "compare" => false,
        _ => anyhow::bail!("expected record or compare"),
    };
    let mut profile = false;
    let mut batch_limit = None;
    for option in &args[5..] {
        if option == "profile" {
            ensure!(!profile, "duplicate profile option");
            profile = true;
        } else {
            ensure!(batch_limit.is_none(), "duplicate batch limit");
            batch_limit = Some(option.parse::<usize>()?);
        }
    }
    let batch_limit = batch_limit.unwrap_or(128);
    let directory = Path::new(&args[2]);
    let image = Image::open(&args[3])?;
    let tokenizer = tokenizers::Tokenizer::from_file(directory.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("loading tokenizer: {e}"))?;
    let prompt = image_prompt("Describe this image briefly.", image.token_count(), 0.9)?;
    let encoded = tokenizer
        .encode(prompt, false)
        .map_err(|e| anyhow::anyhow!("tokenizing: {e}"))?;
    let mut device = Metal::new()?;
    device.set_profiling(true);
    device.set_kernel_profiling(profile)?;
    eprintln!(
        "Loading model; {} image tokens, {} prompt tokens",
        image.token_count(),
        encoded.len()
    );
    let mut model = Inkling::load_with_kv_cache(device, directory, 262144, KvCachePrecision::F16)?;
    model.set_prefill_batch_limit(batch_limit)?;
    model.accelerator_mut().reset_stats();
    let start = Instant::now();
    let mut encoding_start = start;
    let mut prefill_start = start;
    let mut encoding_seconds = 0.;
    let mut next = model.prefill_image_greedy_with_progress(
        encoded.get_ids(),
        &image,
        |event| match event {
            PrefillProgress::ImageEncoderLoaded => encoding_start = Instant::now(),
            PrefillProgress::ImageEncoding {
                completed_layers,
                total_layers,
            } => {
                eprintln!(
                    "vision {completed_layers}/{total_layers}: {:.2}s",
                    encoding_start.elapsed().as_secs_f64()
                );
            }
            PrefillProgress::ImageEncoded => {
                encoding_seconds = encoding_start.elapsed().as_secs_f64();
                prefill_start = Instant::now();
            }
            PrefillProgress::Prompt {
                completed_tokens,
                total_tokens,
            } if completed_tokens % 1024 == 0 => {
                eprintln!(
                    "prefill {completed_tokens}/{total_tokens}: {:.2}s",
                    prefill_start.elapsed().as_secs_f64()
                );
            }
            _ => {}
        },
    )?;
    let prefill_seconds = prefill_start.elapsed().as_secs_f64();
    let preparation_seconds = start.elapsed().as_secs_f64();
    let mut actual = Vec::new();
    actual.extend_from_slice(&(encoded.len() as u64).to_le_bytes());
    for &token in encoded.get_ids() {
        actual.extend_from_slice(&token.to_le_bytes());
    }
    let mut generated = Vec::new();
    for _ in 0..32 {
        generated.push(next);
        actual.extend_from_slice(&next.to_le_bytes());
        let logits = model.step(next)?;
        next = greedy(&logits)?;
        for v in logits {
            actual.extend_from_slice(&v.to_bits().to_le_bytes());
        }
    }
    if record {
        std::fs::write(&args[4], &actual)?;
    } else {
        let expected = std::fs::read(&args[4])?;
        ensure!(
            actual == expected,
            "prompt, generated tokens, or continuation logits differ from snapshot"
        );
    }
    let s = model.accelerator().stats();
    let kernels: Vec<_> = model.accelerator().kernel_stats().into_iter().flatten()
        .map(|(name, s)| serde_json::json!({"kernel": name, "dispatches": s.dispatches, "gpu_seconds": s.gpu_seconds})).collect();
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "device":model.device_name(), "image_width":image.width(), "image_height":image.height(),
            "image_tokens":image.token_count(), "prompt_tokens":encoded.len(), "context":262144,
            "kv_cache":"f16", "effort":0.9, "prefill_batch_limit":batch_limit,
            "profile_kernels":profile, "encoding_seconds":encoding_seconds, "prefill_seconds":prefill_seconds,
            "prefill_tokens_per_second":encoded.len() as f64 / prefill_seconds, "preparation_seconds":preparation_seconds,
            "peak_allocated_bytes":s.peak_allocated_bytes, "device_allocated_bytes":model.accelerator().allocated_memory(),
            "dispatches":s.dispatches, "gpu_seconds":s.gpu_seconds, "cpu_encoding_seconds":s.encoding_seconds,
            "generated_ids":generated, "kernels":kernels, "snapshot_matches": !record,
        }))?
    );
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn main() -> anyhow::Result<()> {
    anyhow::bail!("requires Metal on macOS")
}
