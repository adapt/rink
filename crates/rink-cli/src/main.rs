use anyhow::{Result, ensure};
use clap::{Parser, Subcommand};
use rink_inkling::KvCachePrecision;
use std::path::PathBuf;

#[cfg(target_os = "macos")]
mod bench;
#[cfg(target_os = "macos")]
mod chat;
#[cfg(target_os = "macos")]
mod progress;
#[cfg(any(target_os = "macos", test))]
mod session;

#[derive(Parser)]
#[command(
    version,
    about = "Rink: native Rust inference with model-owned accelerator kernels"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Benchmark fixed prompt/decode lengths; emit JSON with Metal counters and timings.
    Bench {
        #[arg(long, default_value = "checkpoints/inkling-2-mini")]
        model: PathBuf,
        #[arg(long, value_delimiter = ',', default_value = "32,128")]
        prompt_tokens: Vec<usize>,
        #[arg(long, default_value_t = 32)]
        decode_tokens: usize,
        #[arg(long, default_value_t = 3)]
        repetitions: usize,
        #[arg(long, default_value_t = 4096)]
        context: usize,
        /// KV-cache storage precision: f16 or f32; attention accumulates in f32.
        #[arg(long, default_value = "f16")]
        kv_cache: KvCachePrecision,
        /// Force single-token prefill and CPU sampling for comparison.
        #[arg(long)]
        reference: bool,
        /// Diagnostic per-kernel GPU timings; adds compute passes and affects throughput.
        #[arg(long)]
        profile_kernels: bool,
        /// Report prefill timing windows at existing chunk boundaries (in tokens).
        #[arg(long)]
        prefill_window: Option<usize>,
    },
    /// Validate the model configuration and show its physical/virtual layer layout.
    Inspect {
        #[arg(long)]
        model: PathBuf,
    },
    /// Chat interactively on Metal, keeping the model and conversation in memory.
    Chat {
        #[arg(long, default_value = "checkpoints/inkling-2-mini")]
        model: PathBuf,
        /// Maximum generated tokens per reply or /continue, including thinking.
        #[arg(long, default_value_t = 16384)]
        max_tokens: usize,
        /// Maximum cached sequence length across the entire conversation.
        #[arg(long, default_value_t = 262144)]
        context: usize,
        /// KV-cache storage precision: f16 or f32; attention accumulates in f32.
        #[arg(long, default_value = "f16")]
        kv_cache: KvCachePrecision,
        /// Chat reasoning effort, between 0 and 0.99.
        #[arg(long, default_value_t = 0.9)]
        effort: f32,
        /// Display the model's thinking as well as its answer.
        #[arg(long)]
        show_thinking: bool,
    },
    /// Generate text on Metal from a local, original Hugging Face checkpoint.
    Generate {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        prompt: String,
        /// Attach one local PNG or JPEG image to the prompt.
        #[arg(long, conflicts_with = "raw")]
        image: Option<PathBuf>,
        #[arg(long, default_value_t = 128)]
        max_tokens: usize,
        /// Maximum cached sequence length (prompt plus generation).
        #[arg(long, default_value_t = 2048)]
        context: usize,
        /// KV-cache storage precision: f16 or f32; attention accumulates in f32.
        #[arg(long, default_value = "f16")]
        kv_cache: KvCachePrecision,
        /// Pass an already rendered prompt directly to the tokenizer.
        #[arg(long)]
        raw: bool,
        /// Chat reasoning effort, between 0 and 0.99.
        #[arg(long, default_value_t = 0.9)]
        effort: f32,
    },
}
fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Bench {
            model,
            prompt_tokens,
            decode_tokens,
            repetitions,
            context,
            kv_cache,
            reference,
            profile_kernels,
            prefill_window,
        } => {
            ensure!(
                repetitions > 0 && decode_tokens > 0,
                "repetitions and decode tokens must be positive"
            );
            ensure!(prefill_window != Some(0), "prefill window must be positive");
            ensure!(
                !prompt_tokens.is_empty()
                    && prompt_tokens
                        .iter()
                        .all(|&n| n > 0
                            && n.checked_add(decode_tokens).is_some_and(|n| n <= context)),
                "prompt plus decode tokens must fit --context"
            );
            #[cfg(target_os = "macos")]
            {
                bench::run(bench::Options {
                    directory: model,
                    lengths: prompt_tokens,
                    decode: decode_tokens,
                    repetitions,
                    context,
                    kv_cache,
                    reference,
                    profile_kernels,
                    prefill_window,
                })
            }
            #[cfg(not(target_os = "macos"))]
            {
                let _ = (model, kv_cache, reference, profile_kernels, prefill_window);
                anyhow::bail!("benchmark requires macOS with Metal")
            }
        }
        Command::Inspect { model } => {
            let c = rink_inkling::Config::load(model.join("config.json"))?;
            let t = &c.text_config;
            println!(
                "Inkling 2 Mini: {} physical layers, {} layer visits / independent caches",
                t.num_hidden_layers,
                t.schedule().len()
            );
            println!(
                "Width {}, {} of {} experts, vocabulary {} ({} padded)",
                t.hidden_size,
                t.num_experts_per_tok,
                t.n_routed_experts,
                t.unpadded_vocab_size,
                t.vocab_size
            );
            println!(
                "Loop {}..{} repeated {} times; model context limit {}",
                t.loop_config.core_start,
                t.loop_config.core_end,
                t.loop_config.num_passes,
                t.model_max_length
            );
            Ok(())
        }
        Command::Generate {
            model,
            prompt,
            image,
            max_tokens,
            context,
            kv_cache,
            raw,
            effort,
        } => {
            ensure!(max_tokens > 0, "--max-tokens must be positive");
            ensure!(
                effort.is_finite() && (0.0..=0.99).contains(&effort),
                "--effort must be in 0..=0.99"
            );
            generate(
                model, prompt, image, max_tokens, context, raw, effort, kv_cache,
            )
        }
        Command::Chat {
            model,
            max_tokens,
            context,
            kv_cache,
            effort,
            show_thinking,
        } => {
            ensure!(max_tokens > 0, "--max-tokens must be positive");
            ensure!(
                max_tokens < context,
                "--context must leave room for a prompt plus --max-tokens"
            );
            ensure!(
                effort.is_finite() && (0.0..=0.99).contains(&effort),
                "--effort must be in 0..=0.99"
            );
            #[cfg(target_os = "macos")]
            {
                chat::run(model, max_tokens, context, effort, show_thinking, kv_cache)
            }
            #[cfg(not(target_os = "macos"))]
            {
                let _ = (model, show_thinking, kv_cache);
                anyhow::bail!("chat requires macOS with Metal; CUDA is not implemented yet")
            }
        }
    }
}

#[cfg(not(target_os = "macos"))]
#[allow(clippy::too_many_arguments)]
fn generate(
    _: PathBuf,
    _: String,
    _: Option<PathBuf>,
    _: usize,
    _: usize,
    _: bool,
    _: f32,
    _: KvCachePrecision,
) -> Result<()> {
    anyhow::bail!("this build supports generation on macOS with Metal; CUDA is not implemented yet")
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn generate(
    directory: PathBuf,
    prompt: String,
    image: Option<PathBuf>,
    max_tokens: usize,
    context: usize,
    raw: bool,
    effort: f32,
    kv_cache: KvCachePrecision,
) -> Result<()> {
    use anyhow::Context;
    use rink_core::{Accelerator, AutoregressiveModel};
    use std::{io::Write, time::Instant};
    let tokenizer = tokenizers::Tokenizer::from_file(directory.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("loading tokenizer: {e}"))?;
    let image = image.map(rink_inkling::Image::open).transpose()?;
    let rendered = if let Some(image) = &image {
        rink_inkling::image::image_prompt(&prompt, image.token_count(), effort)?
    } else if raw {
        prompt
    } else {
        rink_inkling::chat_prompt(&prompt, effort)?
    };
    let encoded = tokenizer
        .encode(rendered, false)
        .map_err(|e| anyhow::anyhow!("tokenizing prompt: {e}"))?;
    let tokens = encoded.get_ids();
    ensure!(
        !tokens.is_empty(),
        "prompt must encode to at least one token"
    );
    let total = tokens
        .len()
        .checked_add(max_tokens - 1)
        .context("token budget overflow")?;
    ensure!(
        total <= context,
        "prompt ({} tokens) and generation require {total} cache positions, exceeding --context {context}",
        tokens.len()
    );
    let device = rink_metal::Metal::new()?;
    eprintln!(
        "Loading on {} (recommended GPU working set {:.1} GiB)",
        device.name(),
        device.recommended_memory() as f64 / (1u64 << 30) as f64
    );
    let mut model =
        rink_inkling::Inkling::load_with_kv_cache(device, &directory, context, kv_cache)?;
    let mut prefill = progress::PrefillDisplay::new();
    let mut next = if let Some(image) = &image {
        model.prefill_image_greedy_with_progress(tokens, image, |event| prefill.observe(event))
    } else {
        model.prefill_greedy_with_progress(tokens, |event| prefill.observe(event))
    }
    .context("prefilling prompt")?;
    eprintln!("Generating…");
    let start = Instant::now();
    // Keep protocol tokens visible so thinking and final-answer boundaries are explicit.
    let mut stream = tokenizer.decode_stream(false);
    let mut output = std::io::stdout().lock();
    let mut generated = 0;
    let mut decoded = 0;
    for i in 0..max_tokens {
        if next == model.config().eos_token_id {
            break;
        }
        if let Some(text) = stream
            .step(next)
            .map_err(|e| anyhow::anyhow!("decoding token: {e}"))?
        {
            write!(output, "{text}")?;
            output.flush()?;
        }
        generated += 1;
        if i + 1 < max_tokens {
            next = model.step_greedy(next)?;
            decoded += 1;
        }
    }
    writeln!(output)?;
    let seconds = start.elapsed().as_secs_f64();
    eprintln!(
        "Generation: {generated} tokens in {seconds:.2}s ({:.1} decode tok/s)",
        progress::rate(decoded, seconds)
    );
    Ok(())
}
