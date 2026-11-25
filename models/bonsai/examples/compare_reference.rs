//! Compare every saved logit from an independent MLX oracle with the native engine.
#[cfg(target_os = "macos")]
fn main() -> anyhow::Result<()> {
    use rink_core::{AutoregressiveModel, greedy};
    use serde::Deserialize;
    use std::collections::BTreeMap;
    #[derive(Deserialize)]
    struct Reference {
        tokens: Vec<u32>,
        logits: BTreeMap<usize, Vec<f32>>,
    }
    let args: Vec<_> = std::env::args().collect();
    anyhow::ensure!(
        (3..=4).contains(&args.len()),
        "usage: compare_reference CHECKPOINT REFERENCE.json [PREFILL_BATCH]"
    );
    let reference: Reference = serde_json::from_slice(&std::fs::read(&args[2])?)?;
    let mut model =
        rink_bonsai::Bonsai::load(rink_metal::Metal::new()?, &args[1], reference.tokens.len())?;
    let batch = args.get(3).map(|s| s.parse::<usize>()).transpose()?;
    if let Some(batch) = batch {
        model.set_prefill_batch_limit(batch)?;
    }
    let mut results = Vec::new();
    let mut all_match = true;
    for (i, &token) in reference.tokens.iter().enumerate() {
        if batch.is_some() && !reference.logits.contains_key(&(i + 1)) {
            continue;
        }
        let actual = if batch.is_some() {
            model.prefill(&reference.tokens[model.position()..i + 1])?
        } else {
            model.step(token)?
        };
        if let Some(expected) = reference.logits.get(&(i + 1)) {
            anyhow::ensure!(actual.len() == expected.len(), "vocabulary mismatch");
            let mut squared = 0.0;
            let mut norm = 0.0;
            let mut maximum = 0.0f32;
            for (&a, &b) in actual.iter().zip(expected) {
                maximum = maximum.max((a - b).abs());
                squared += (a as f64 - b as f64).powi(2);
                norm += (b as f64).powi(2);
            }
            let error = (squared / norm.max(1e-30)).sqrt();
            let agrees = greedy(&actual)? == greedy(expected)?;
            all_match &= agrees;
            let row = serde_json::json!({"position":i+1,"logits":actual.len(),"max_absolute_error":maximum,"relative_rms_error":error,"greedy_matches":agrees,"actual_token":greedy(&actual)?,"reference_token":greedy(expected)?});
            eprintln!("{row}");
            results.push(row);
            anyhow::ensure!(
                error < 0.01,
                "relative RMS logit error exceeds 1% at position {}",
                i + 1
            );
        }
    }
    println!(
        "{}",
        serde_json::to_string_pretty(
            &serde_json::json!({"prefill_batch":batch,"all_greedy_match":all_match,"checks":results})
        )?
    );
    anyhow::ensure!(all_match, "greedy predictions differ from the reference");
    Ok(())
}
#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("requires macOS with Metal");
}
