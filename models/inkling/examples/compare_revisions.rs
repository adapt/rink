//! Record full logits, then compare another build on exactly the same history.
//! compare_revisions record|compare MODEL_DIR SNAPSHOT_JSON [PROMPT_TOKENS]
//! The snapshot can be large; only the comparison summary belongs in benchmarks/.
#[cfg(target_os = "macos")]
fn main() -> anyhow::Result<()> {
    use rink_core::{AutoregressiveModel, greedy};
    use rink_inkling::Inkling;
    use rink_metal::Metal;
    use serde::{Deserialize, Serialize};
    use serde_json::json;
    use std::{
        fs::File,
        io::{BufReader, BufWriter},
        path::PathBuf,
    };

    #[derive(Serialize, Deserialize)]
    struct Snapshot {
        prompt: Vec<u32>,
        lengths: Vec<usize>,
        continuation: Vec<u32>,
        logits: Vec<Vec<f32>>,
    }
    let args: Vec<_> = std::env::args().collect();
    anyhow::ensure!(
        (4..=5).contains(&args.len()),
        "usage: compare_revisions record|compare MODEL_DIR SNAPSHOT_JSON [PROMPT_TOKENS]"
    );
    let record = match args[1].as_str() {
        "record" => true,
        "compare" => false,
        _ => anyhow::bail!("expected record or compare"),
    };
    let directory = PathBuf::from(&args[2]);
    let mut reference: Snapshot = if record {
        let length = args
            .get(4)
            .map(|s| s.parse::<usize>())
            .transpose()?
            .unwrap_or(1024);
        anyhow::ensure!(
            (128..=8192).contains(&length),
            "prompt length must be in 128..=8192"
        );
        let tokenizer = tokenizers::Tokenizer::from_file(directory.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let encoded = tokenizer
            .encode(
                "Explain how a computer processes information, using clear examples. ",
                false,
            )
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        anyhow::ensure!(!encoded.is_empty(), "empty prompt");
        let mut lengths = vec![128, length];
        lengths.dedup();
        Snapshot {
            prompt: encoded
                .get_ids()
                .iter()
                .copied()
                .cycle()
                .take(length)
                .collect(),
            lengths,
            continuation: Vec::new(),
            logits: Vec::new(),
        }
    } else {
        serde_json::from_reader(BufReader::new(File::open(&args[3])?))?
    };
    anyhow::ensure!(
        !reference.lengths.is_empty()
            && reference.lengths[0] > 0
            && reference.lengths.windows(2).all(|v| v[0] < v[1])
            && reference.lengths.last() == Some(&reference.prompt.len())
            && reference.prompt.len() <= 8192,
        "invalid snapshot prompt lengths"
    );
    if !record {
        anyhow::ensure!(
            reference.continuation.len() == 32
                && reference.logits.len() == reference.lengths.len() + 32,
            "invalid snapshot continuation or logit count"
        );
    }
    let mut model = Inkling::load(Metal::new()?, &directory, 16384)?;
    let mut snapshots = Vec::new();
    let mut previous = 0;
    for &length in &reference.lengths {
        snapshots.push((length, model.prefill(&reference.prompt[previous..length])?));
        previous = length;
    }
    for step in 0..32 {
        let token = if record {
            let t = greedy(&snapshots.last().unwrap().1)?;
            reference.continuation.push(t);
            t
        } else {
            reference.continuation[step]
        };
        snapshots.push((reference.prompt.len() + step + 1, model.step(token)?));
    }
    if record {
        reference.logits = snapshots.into_iter().map(|(_, logits)| logits).collect();
        serde_json::to_writer(BufWriter::new(File::create(&args[3])?), &reference)?;
        eprintln!("Recorded {} snapshots", reference.logits.len());
        return Ok(());
    }
    anyhow::ensure!(
        reference.logits.len() == snapshots.len(),
        "snapshot count mismatch"
    );
    let mut measurements = Vec::new();
    for ((position, actual), expected) in snapshots.iter().zip(&reference.logits) {
        anyhow::ensure!(expected.len() == actual.len(), "logit dimensions differ");
        let n = expected.len() as f64;
        let mean = expected
            .iter()
            .zip(actual)
            .map(|(&a, &b)| f64::from(b) - f64::from(a))
            .sum::<f64>()
            / n;
        let rmse = (expected
            .iter()
            .zip(actual)
            .map(|(&a, &b)| (f64::from(b) - f64::from(a) - mean).powi(2))
            .sum::<f64>()
            / n)
            .sqrt();
        let max_error = expected
            .iter()
            .zip(actual)
            .map(|(&a, &b)| (b - a).abs())
            .fold(0f32, f32::max);
        let log_z = |v: &[f32]| {
            let m = v.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
            m + v
                .iter()
                .map(|&x| (f64::from(x) - m).exp())
                .sum::<f64>()
                .ln()
        };
        let a_z = log_z(expected);
        let b_z = log_z(actual);
        let kl = expected
            .iter()
            .zip(actual)
            .map(|(&a, &b)| {
                let p = f64::from(a) - a_z;
                p.exp() * (p - f64::from(b) + b_z)
            })
            .sum::<f64>();
        measurements.push(json!({"position":position,"centered_logit_rmse":rmse,"max_logit_error":max_error,"kl_reference_to_current":kl,"reference_argmax":greedy(expected)?,"current_argmax":greedy(actual)?}));
    }
    println!(
        "{}",
        serde_json::to_string_pretty(
            &json!({"context":16384,"kv_cache":"f16","prompt_lengths":reference.lengths,"teacher_forced_tokens":reference.continuation,"measurements":measurements})
        )?
    );
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn main() -> anyhow::Result<()> {
    anyhow::bail!("this comparison requires Metal on macOS")
}
