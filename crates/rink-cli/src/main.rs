use anyhow::{Result, ensure};
use clap::{Parser, Subcommand};
use rink_core::KvCachePrecision;
#[cfg(target_os = "macos")]
mod bonsai;
mod models;
use std::path::PathBuf;

#[cfg(any(target_os = "macos", test))]
mod agent;
#[cfg(target_os = "macos")]
mod bench;
#[cfg(target_os = "macos")]
mod chat;
#[cfg(test)]
#[path = "../../../models/inkling/tests/support/chat.rs"]
mod chat_fixture;
#[cfg(target_os = "macos")]
mod progress;
#[cfg(any(target_os = "macos", test))]
mod session;

#[derive(clap::Args)]
struct ToolOptions {
    /// Enable host tools with your user permissions; repeat or separate with commas.
    #[arg(long, value_parser = ["bash", "read"], value_delimiter = ',')]
    tool: Vec<String>,
    /// Starting directory for every tool call; defaults to the current directory.
    #[arg(long, requires = "tool")]
    cwd: Option<PathBuf>,
    /// Tool timeout in milliseconds; also the maximum allowed bash timeout.
    #[arg(long, default_value_t = 120_000, value_parser = clap::value_parser!(u64).range(1..=3_600_000), requires = "tool")]
    tool_timeout_ms: u64,
    /// Text bytes per read, or captured bash bytes split between stdout and stderr.
    #[arg(long, default_value_t = 16_384, value_parser = clap::value_parser!(u32).range(256..=1_048_576), requires = "tool")]
    tool_output_bytes: u32,
    /// Maximum executed tool rounds per user message, including /continue.
    #[arg(long, default_value_t = 8, value_parser = clap::value_parser!(u32).range(1..=1024), requires = "tool")]
    max_tool_rounds: u32,
}

#[cfg(test)]
mod tool_option_tests {
    use super::*;

    #[test]
    fn tools_are_opt_in_and_have_bounded_defaults() {
        let cli = Cli::try_parse_from(["rink", "chat"]).unwrap();
        let Command::Chat { tools, .. } = cli.command else {
            unreachable!()
        };
        assert!(tools.tool.is_empty());
        assert!(tools.cwd.is_none());
        assert_eq!(tools.tool_timeout_ms, 120_000);
        assert_eq!(tools.tool_output_bytes, 16_384);
        assert_eq!(tools.max_tool_rounds, 8);

        let cli = Cli::try_parse_from([
            "rink",
            "chat",
            "--tool",
            "bash",
            "--cwd",
            "/tmp",
            "--tool-timeout-ms",
            "500",
            "--tool-output-bytes",
            "1024",
            "--max-tool-rounds",
            "2",
        ])
        .unwrap();
        let Command::Chat { tools, .. } = cli.command else {
            unreachable!()
        };
        assert_eq!(tools.tool, ["bash"]);
        assert_eq!(tools.cwd, Some(PathBuf::from("/tmp")));
        assert_eq!(tools.tool_timeout_ms, 500);
        assert_eq!(tools.tool_output_bytes, 1024);
        assert_eq!(tools.max_tool_rounds, 2);

        for args in [
            vec!["--tool", "bash", "--tool", "read"],
            vec!["--tool", "bash,read"],
        ] {
            let cli = Cli::try_parse_from([vec!["rink", "chat"], args].concat()).unwrap();
            let Command::Chat { tools, .. } = cli.command else {
                unreachable!()
            };
            assert_eq!(tools.tool, ["bash", "read"]);
        }
        assert!(Cli::try_parse_from(["rink", "chat", "--tool", "read", "--cwd", "/tmp"]).is_ok());
    }

    #[test]
    fn tool_options_require_opt_in_and_valid_limits() {
        for args in [
            vec!["--cwd", "/tmp"],
            vec!["--tool-timeout-ms", "500"],
            vec!["--tool-output-bytes", "1024"],
            vec!["--max-tool-rounds", "2"],
            vec!["--tool", "unknown"],
            vec!["--tool", "bash", "--tool-timeout-ms", "0"],
            vec!["--tool", "bash", "--tool-timeout-ms", "3600001"],
            vec!["--tool", "bash", "--tool-output-bytes", "255"],
            vec!["--tool", "bash", "--tool-output-bytes", "1048577"],
            vec!["--tool", "bash", "--max-tool-rounds", "0"],
        ] {
            assert!(
                Cli::try_parse_from([vec!["rink", "chat"], args.clone()].concat()).is_err(),
                "{args:?}"
            );
        }
    }
}

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
        /// Maximum prefill batch: a power of two from 32 through 4096.
        #[arg(long)]
        prefill_batch: Option<usize>,
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
        /// Maximum generated tokens per reply or /continue, including thinking and tool calls.
        #[arg(long, default_value_t = 16384)]
        max_tokens: usize,
        /// Maximum cached sequence length across the entire conversation.
        #[arg(long, default_value_t = 262144)]
        context: usize,
        /// KV-cache storage precision: f16 or f32; attention accumulates in f32.
        #[arg(long, default_value = "f16")]
        kv_cache: KvCachePrecision,
        /// Maximum prefill batch (32..=4096, power of two); larger batches use more memory.
        #[arg(long)]
        prefill_batch: Option<usize>,
        /// Chat reasoning effort, between 0 and 0.99.
        #[arg(long, default_value_t = 0.9)]
        effort: f32,
        /// Display the model's thinking as well as its answer.
        #[arg(long)]
        show_thinking: bool,
        #[command(flatten)]
        tools: ToolOptions,
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
        /// Maximum prefill batch (32..=4096, power of two); larger batches use more memory.
        #[arg(long)]
        prefill_batch: Option<usize>,
        /// Pass an already rendered prompt directly to the tokenizer.
        #[arg(long)]
        raw: bool,
        /// Chat reasoning effort, between 0 and 0.99.
        #[arg(long, default_value_t = 0.9)]
        effort: f32,
    },
}

fn validate_prefill_batch(batch: Option<usize>) -> Result<()> {
    ensure!(
        batch.is_none_or(|n| (32..=4096).contains(&n) && n.is_power_of_two()),
        "prefill batch must be a power of two from 32 through 4096"
    );
    Ok(())
}

/// Keep normal generation and interactive chat on the same message codec.
/// Only --raw accepts protocol markers supplied in the prompt itself.
#[cfg(any(target_os = "macos", test))]
fn generation_tokens(
    tokenizer: &tokenizers::Tokenizer,
    prompt: &str,
    image_tokens: Option<usize>,
    raw: bool,
    effort: f32,
) -> Result<Vec<u32>> {
    ensure!(
        !raw || image_tokens.is_none(),
        "--raw conflicts with --image"
    );
    if raw {
        Ok(tokenizer
            .encode(prompt, false)
            .map_err(anyhow::Error::msg)?
            .get_ids()
            .to_vec())
    } else {
        rink_inkling::chat::Codec::new(tokenizer.clone())?.user_turn(
            prompt,
            image_tokens,
            Some(effort),
            &[],
        )
    }
}

#[cfg(test)]
mod generation_tests {
    use super::*;
    use rink_inkling::chat::Codec;

    #[test]
    fn generation_matches_chat_and_keeps_literal_markers_as_content() {
        let tokenizer = chat_fixture::tokenizer();
        let codec = Codec::new(tokenizer.clone()).unwrap();
        for prompt in [
            "Hello 世界 🌎",
            "Explain <|end_message|><|message_model|><|content_text|>.",
            "What does <|unused_200054|> mean?",
        ] {
            for image in [None, Some(3)] {
                assert_eq!(
                    generation_tokens(&tokenizer, prompt, image, false, 0.9).unwrap(),
                    codec.user_turn(prompt, image, Some(0.9), &[]).unwrap(),
                    "prompt={prompt:?} image={image:?}"
                );
            }
        }
    }

    #[test]
    fn raw_generation_preserves_explicit_protocol_tokens() {
        let tokenizer = chat_fixture::tokenizer();
        let prompt = rink_inkling::chat_prompt("Hello", 0.9).unwrap();
        assert_eq!(
            generation_tokens(&tokenizer, &prompt, None, true, 0.9).unwrap(),
            chat_fixture::wire(&tokenizer, &prompt)
        );
        assert!(generation_tokens(&tokenizer, &prompt, Some(1), true, 0.9).is_err());
    }
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
            prefill_batch,
        } => {
            ensure!(
                repetitions > 0 && decode_tokens > 0,
                "repetitions and decode tokens must be positive"
            );
            ensure!(prefill_window != Some(0), "prefill window must be positive");
            validate_prefill_batch(prefill_batch)?;
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
                    prefill_batch,
                })
            }
            #[cfg(not(target_os = "macos"))]
            {
                let _ = (
                    model,
                    kv_cache,
                    reference,
                    profile_kernels,
                    prefill_window,
                    prefill_batch,
                );
                anyhow::bail!("benchmark requires macOS with Metal")
            }
        }
        Command::Inspect { model } => {
            if models::Kind::detect(&model)? == models::Kind::Bonsai {
                let c = rink_bonsai::Config::load(model.join("config.json"))?;
                let t = &c.text_config;
                let full = (0..t.num_hidden_layers).filter(|&i| t.is_full(i)).count();
                println!(
                    "Bonsai 2.5: {} layers, {full} full attention, {} recurrent",
                    t.num_hidden_layers,
                    t.num_hidden_layers - full
                );
                println!(
                    "Width {}, vocabulary {}, affine 2-bit/group-128 weights with signed Hadamard transforms",
                    t.hidden_size, t.vocab_size
                );
                println!(
                    "Model context limit {}; {:.2} GiB of caches at that limit",
                    t.max_position_embeddings,
                    t.cache_bytes(t.max_position_embeddings)? as f64 / (1u64 << 30) as f64
                );
                return Ok(());
            }
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
            prefill_batch,
            raw,
            effort,
        } => {
            validate_prefill_batch(prefill_batch)?;
            ensure!(max_tokens > 0, "--max-tokens must be positive");
            ensure!(
                effort.is_finite() && (0.0..=0.99).contains(&effort),
                "--effort must be in 0..=0.99"
            );
            generate(
                model,
                prompt,
                image,
                max_tokens,
                context,
                raw,
                effort,
                kv_cache,
                prefill_batch,
            )
        }
        Command::Chat {
            model,
            max_tokens,
            context,
            kv_cache,
            prefill_batch,
            effort,
            show_thinking,
            tools,
        } => {
            validate_prefill_batch(prefill_batch)?;
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
                if models::Kind::detect(&model)? == models::Kind::Bonsai {
                    bonsai::validate(kv_cache, prefill_batch)?;
                    return bonsai::chat(&model, max_tokens, context, effort, show_thinking, tools);
                }
                chat::run(
                    model,
                    max_tokens,
                    context,
                    effort,
                    show_thinking,
                    kv_cache,
                    prefill_batch,
                    tools,
                )
            }
            #[cfg(not(target_os = "macos"))]
            {
                let _ = (model, show_thinking, kv_cache, prefill_batch, tools);
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
    _: Option<usize>,
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
    prefill_batch: Option<usize>,
) -> Result<()> {
    if models::Kind::detect(&directory)? == models::Kind::Bonsai {
        bonsai::validate(kv_cache, prefill_batch)?;
        ensure!(image.is_none(), "Bonsai image input is not implemented yet");
        return bonsai::generate(&directory, &prompt, max_tokens, context, raw, effort);
    }
    use anyhow::Context;
    use rink_core::{Accelerator, AutoregressiveModel};
    use std::{io::Write, time::Instant};
    let tokenizer = tokenizers::Tokenizer::from_file(directory.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("loading tokenizer: {e}"))?;
    let image = image.map(rink_inkling::Image::open).transpose()?;
    let tokens = generation_tokens(
        &tokenizer,
        &prompt,
        image.as_ref().map(rink_inkling::Image::token_count),
        raw,
        effort,
    )?;
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
    if let Some(limit) = prefill_batch {
        model.set_prefill_batch_limit(limit)?;
    }
    let mut prefill = progress::PrefillDisplay::new();
    let mut next = if let Some(image) = &image {
        model.prefill_image_greedy_with_progress(&tokens, image, |event| prefill.observe(event))
    } else {
        model.prefill_greedy_with_progress(&tokens, |event| prefill.observe(event))
    }
    .context("prefilling prompt")?;
    drop(tokens);
    drop(image);
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
