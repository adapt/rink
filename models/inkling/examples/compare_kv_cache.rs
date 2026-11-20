//! Compare cache precision on identical token histories, loading one model at a time.
//! cargo run --release -p rink-inkling --example compare_kv_cache -- MODEL_DIR
#[cfg(target_os = "macos")]
fn main() -> anyhow::Result<()> {
    use rink_core::{AutoregressiveModel, greedy};
    use rink_inkling::{Inkling, KvCachePrecision};
    use rink_metal::Metal;
    use serde_json::json;
    use std::{path::PathBuf, time::Instant};

    let directory = PathBuf::from(
        std::env::args()
            .nth(1)
            .ok_or_else(|| anyhow::anyhow!("usage: compare_kv_cache MODEL_DIR"))?,
    );
    let tokenizer = tokenizers::Tokenizer::from_file(directory.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("loading tokenizer: {e}"))?;
    let encoded = tokenizer
        .encode(
            "Explain how a computer processes information, using clear examples. ",
            false,
        )
        .map_err(|e| anyhow::anyhow!("tokenizing prompt: {e}"))?;
    anyhow::ensure!(!encoded.is_empty(), "empty prompt");
    let lengths = [128, 1024, 8192];
    let prompt: Vec<_> = encoded
        .get_ids()
        .iter()
        .copied()
        .cycle()
        .take(8192)
        .collect();
    let mut reference = Vec::new();
    let mut continuation = Vec::new();
    let mut measurements = Vec::new();
    for precision in [KvCachePrecision::F32, KvCachePrecision::F16] {
        let start = Instant::now();
        let mut model = Inkling::load_with_kv_cache(Metal::new()?, &directory, 16384, precision)?;
        let mut snapshots = Vec::new();
        let mut previous = 0;
        for length in lengths {
            let logits = model.prefill(&prompt[previous..length])?;
            snapshots.push((length, logits));
            previous = length;
            eprintln!("{precision}: prefilled {length} tokens");
        }
        for step in 0..32 {
            let token = if precision == KvCachePrecision::F32 {
                let token = greedy(&snapshots.last().unwrap().1)?;
                continuation.push(token);
                token
            } else {
                continuation[step]
            };
            snapshots.push((8193 + step, model.step(token)?));
        }
        if precision == KvCachePrecision::F32 {
            reference = snapshots;
        } else {
            for ((position, expected), (actual_position, actual)) in
                reference.iter().zip(&snapshots)
            {
                anyhow::ensure!(
                    position == actual_position && expected.len() == actual.len(),
                    "snapshot mismatch"
                );
                let n = expected.len() as f64;
                let mean_delta = expected
                    .iter()
                    .zip(actual)
                    .map(|(&a, &b)| f64::from(b) - f64::from(a))
                    .sum::<f64>()
                    / n;
                let centered_mse = expected
                    .iter()
                    .zip(actual)
                    .map(|(&a, &b)| {
                        let d = f64::from(b) - f64::from(a) - mean_delta;
                        d * d
                    })
                    .sum::<f64>()
                    / n;
                let max_error = expected
                    .iter()
                    .zip(actual)
                    .map(|(&a, &b)| (b - a).abs())
                    .fold(0f32, f32::max);
                let log_z = |logits: &[f32]| {
                    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
                    max + logits
                        .iter()
                        .map(|&x| (f64::from(x) - max).exp())
                        .sum::<f64>()
                        .ln()
                };
                let ref_z = log_z(expected);
                let actual_z = log_z(actual);
                let kl = expected
                    .iter()
                    .zip(actual)
                    .map(|(&a, &b)| {
                        let log_p = f64::from(a) - ref_z;
                        log_p.exp() * (log_p - f64::from(b) + actual_z)
                    })
                    .sum::<f64>();
                measurements.push(json!({
                    "position": position,
                    "centered_logit_rmse": centered_mse.sqrt(),
                    "max_logit_error": max_error,
                    "kl_f32_to_f16": kl,
                    "f32_argmax": greedy(expected)?, "f16_argmax": greedy(actual)?,
                }));
            }
        }
        eprintln!(
            "{precision}: finished in {:.1}s",
            start.elapsed().as_secs_f64()
        );
        // Dropping here releases the checkpoint and caches before the next load.
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "context": 16384, "prompt_lengths": lengths, "teacher_forced_tokens": continuation,
            "measurements": measurements,
        }))?
    );
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn main() -> anyhow::Result<()> {
    anyhow::bail!("this cache comparison requires Metal on macOS")
}
