//! Check full-vocabulary logits against another build on identical token histories.
//! compare_revisions record MODEL_DIR SNAPSHOT_JSON [f16|f32] [PROMPT_TOKENS]
//! compare_revisions compare MODEL_DIR SNAPSHOT_JSON [BATCH_LIMIT]
//! Record with the baseline executable, then compare with the candidate executable.
//! Snapshots store float bits, not decimal floats, and can exceed 100 MB.
#[cfg(target_os = "macos")]
mod comparison {
    use anyhow::{Result, ensure};
    use rink_core::{AutoregressiveModel, greedy};
    use rink_inkling::{Inkling, KvCachePrecision};
    use rink_metal::Metal;
    use serde::{Deserialize, Serialize};
    use serde_json::json;
    use std::{
        fs::File,
        io::{BufReader, BufWriter},
        path::Path,
    };

    #[derive(Serialize, Deserialize)]
    struct Case {
        name: String,
        prompt: Vec<u32>,
        checkpoints: Vec<usize>,
        continuation: Vec<u32>,
        logit_bits: Vec<Vec<u32>>,
    }

    #[derive(Serialize, Deserialize)]
    struct Snapshot {
        version: u32,
        context: usize,
        kv_cache: String,
        cases: Vec<Case>,
    }

    fn new_snapshot(
        directory: &Path,
        precision: KvCachePrecision,
        length: usize,
    ) -> Result<Snapshot> {
        ensure!(
            (1025..=8192).contains(&length),
            "prompt length must be in 1025..=8192"
        );
        let tokenizer = tokenizers::Tokenizer::from_file(directory.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("loading tokenizer: {e}"))?;
        let mut cases = Vec::new();
        for (name, text, repeat) in [
            (
                "chunk-and-ring-boundaries",
                "Explain how a computer processes information, using clear examples. ",
                true,
            ),
            (
                "code",
                "Write a Rust function that merges two sorted arrays of integers. Preserve duplicates. Explain the time and space complexity, and include examples with empty arrays and negative numbers. Then explain how the implementation would change if the input arrays could be modified in place.",
                false,
            ),
            (
                "conversation",
                "<|message_user|>Remember the number 57. Reply with OK.<|end_message|><|message_assistant|>OK.<|end_message|><|message_user|>What number did I ask you to remember? Explain how you kept track of it. Then give three examples of situations where remembering an exact number matters.<|end_message|><|message_assistant|>",
                false,
            ),
        ] {
            let encoded = tokenizer
                .encode(text, false)
                .map_err(|e| anyhow::anyhow!("tokenizing prompt: {e}"))?;
            ensure!(!encoded.is_empty(), "empty prompt");
            let prompt: Vec<_> = encoded
                .get_ids()
                .iter()
                .copied()
                .cycle()
                .take(if repeat { length } else { encoded.len() })
                .collect();
            let mut checkpoints: Vec<_> = [1, 7, 8, 31, 32, 33, 127, 128, 129, 1023, 1024, 1025]
                .into_iter()
                .filter(|&n| n < prompt.len())
                .collect();
            checkpoints.push(prompt.len());
            cases.push(Case {
                name: name.into(),
                prompt,
                checkpoints,
                continuation: Vec::new(),
                logit_bits: Vec::new(),
            });
        }
        Ok(Snapshot {
            version: 1,
            context: 16384,
            kv_cache: precision.to_string(),
            cases,
        })
    }

    pub fn run() -> Result<()> {
        let args: Vec<_> = std::env::args().collect();
        ensure!(
            (4..=6).contains(&args.len()),
            "usage: compare_revisions record|compare MODEL_DIR SNAPSHOT_JSON [f16|f32] [PROMPT_TOKENS]"
        );
        let record = match args[1].as_str() {
            "record" => true,
            "compare" => false,
            _ => anyhow::bail!("expected record or compare"),
        };
        ensure!(
            record || args.len() <= 5,
            "compare reads precision and prompts from the snapshot; optional argument is BATCH_LIMIT"
        );
        let directory = Path::new(&args[2]);
        let mut snapshot = if record {
            let precision = args
                .get(4)
                .map(|s| s.parse())
                .transpose()?
                .unwrap_or_default();
            let length = args.get(5).map(|s| s.parse()).transpose()?.unwrap_or(1025);
            new_snapshot(directory, precision, length)?
        } else {
            serde_json::from_reader::<_, Snapshot>(BufReader::new(File::open(&args[3])?))?
        };
        ensure!(
            snapshot.version == 1 && snapshot.context == 16384 && !snapshot.cases.is_empty(),
            "invalid snapshot header"
        );
        for case in &snapshot.cases {
            ensure!(
                !case.checkpoints.is_empty()
                    && case.checkpoints[0] > 0
                    && case.checkpoints.windows(2).all(|v| v[0] < v[1])
                    && case.checkpoints.last() == Some(&case.prompt.len())
                    && case.prompt.len() <= 8192,
                "invalid checkpoints for {}",
                case.name
            );
            if !record {
                ensure!(
                    case.continuation.len() == 32
                        && case.logit_bits.len() == case.checkpoints.len() + 32,
                    "invalid continuation or logit count for {}",
                    case.name
                );
            }
        }
        let mut model = Inkling::load_with_kv_cache(
            Metal::new()?,
            directory,
            snapshot.context,
            snapshot.kv_cache.parse()?,
        )?;
        if !record && let Some(limit) = args.get(4) {
            model.set_prefill_batch_limit(limit.parse()?)?;
        }
        let mut measurements = Vec::new();
        let mut compared_logits = 0usize;
        let mut mismatched_logits = 0usize;
        for case in &mut snapshot.cases {
            model.reset()?;
            let mut previous = 0;
            let mut actual = Vec::new();
            for &length in &case.checkpoints {
                actual.push((length, model.prefill(&case.prompt[previous..length])?));
                previous = length;
            }
            for step in 0..32 {
                let token = if record {
                    let token = greedy(&actual.last().unwrap().1)?;
                    case.continuation.push(token);
                    token
                } else {
                    case.continuation[step]
                };
                actual.push((case.prompt.len() + step + 1, model.step(token)?));
            }
            if record {
                case.logit_bits = actual
                    .into_iter()
                    .map(|(_, logits)| logits.into_iter().map(f32::to_bits).collect())
                    .collect();
            } else {
                for ((position, logits), expected) in actual.iter().zip(&case.logit_bits) {
                    ensure!(logits.len() == expected.len(), "logit dimensions differ");
                    let mut mismatches = 0;
                    let mut max_error = 0f32;
                    for (&value, &bits) in logits.iter().zip(expected) {
                        let reference = f32::from_bits(bits);
                        ensure!(reference.is_finite(), "non-finite reference logit");
                        mismatches += usize::from(value.to_bits() != bits);
                        max_error = max_error.max((value - reference).abs());
                    }
                    compared_logits += logits.len();
                    mismatched_logits += mismatches;
                    measurements.push(json!({"case":case.name,"position":position,
                        "mismatched_logits":mismatches,"max_absolute_error":max_error}));
                }
            }
            eprintln!(
                "{}: {} prompt tokens, 32 continuation tokens",
                case.name,
                case.prompt.len()
            );
        }
        if record {
            serde_json::to_writer(BufWriter::new(File::create(&args[3])?), &snapshot)?;
        } else {
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "context":snapshot.context,"kv_cache":snapshot.kv_cache,
                    "prefill_batch":model.prefill_batch_limit(),
                    "compared_logits":compared_logits,"mismatched_logits":mismatched_logits,
                    "bit_identical":mismatched_logits == 0,"measurements":measurements,
                }))?
            );
            ensure!(
                mismatched_logits == 0,
                "{mismatched_logits} logits differ from the reference bits"
            );
        }
        Ok(())
    }
}

#[cfg(target_os = "macos")]
fn main() -> anyhow::Result<()> {
    comparison::run()
}

#[cfg(not(target_os = "macos"))]
fn main() -> anyhow::Result<()> {
    anyhow::bail!("this comparison requires Metal on macOS")
}
