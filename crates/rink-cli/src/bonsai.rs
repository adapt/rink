//! Bonsai's text UI uses the shared conversation session and its own chat codec.
use crate::{ToolOptions, session::Session};
use anyhow::{Context, Result, ensure};
use rink_bonsai::{Bonsai, chat::Codec};
use rink_core::{Accelerator, AutoregressiveModel, KvCachePrecision};
use rink_metal::Metal;
use std::{
    io::{BufRead, Write},
    path::Path,
    time::Instant,
};

pub fn validate(kv: KvCachePrecision, batch: Option<usize>) -> Result<()> {
    ensure!(
        kv == KvCachePrecision::F16,
        "Bonsai uses FP16 activations and KV storage; use --kv-cache f16"
    );
    ensure!(
        batch.is_none_or(|n| (1..=4096).contains(&n) && n.is_power_of_two()),
        "prefill batch must be a power of two from 1 through 4096"
    );
    Ok(())
}
fn load(directory: &Path, context: usize, batch: Option<usize>) -> Result<Bonsai<Metal>> {
    let device = Metal::new()?;
    let config = rink_bonsai::Config::load(directory.join("config.json"))?;
    let cache = config.text_config.cache_bytes(context)?;
    eprintln!(
        "Loading Bonsai 2.5 on {}; {:.2} GiB for KV and recurrent caches…",
        device.name(),
        cache as f64 / (1u64 << 30) as f64
    );
    Bonsai::load_with_prefill_batch(
        device,
        directory,
        context,
        batch.unwrap_or(rink_bonsai::DEFAULT_PREFILL_BATCH),
    )
}
fn codec(directory: &Path) -> Result<Codec> {
    let codec = Codec::new(
        tokenizers::Tokenizer::from_file(directory.join("tokenizer.json"))
            .map_err(anyhow::Error::msg)?,
    )?;
    let config = rink_bonsai::Config::load(directory.join("config.json"))?;
    ensure!(
        codec.eos()[1] == config.text_config.eos_token_id
            && codec
                .eos()
                .iter()
                .all(|&t| (t as usize) < config.text_config.vocab_size),
        "Bonsai tokenizer stop tokens do not match the model configuration"
    );
    Ok(codec)
}
fn prefill(model: &mut Bonsai<Metal>, tokens: &[u32]) -> Result<u32> {
    eprintln!("Prefilling {} tokens…", tokens.len());
    let start = Instant::now();
    let mut status = crate::progress::StatusLine::new();
    let next = model.prefill_greedy_with_progress(tokens, |completed| {
        status.update(
            &format!(
                "Prefill: {completed}/{} tokens; {:.1}s",
                tokens.len(),
                start.elapsed().as_secs_f64()
            ),
            false,
        );
    })?;
    status.clear();
    let seconds = start.elapsed().as_secs_f64();
    eprintln!(
        "Prefill: {} tokens in {seconds:.2}s ({:.1} tok/s)",
        tokens.len(),
        tokens.len() as f64 / seconds
    );
    Ok(next)
}

pub fn generate(
    directory: &Path,
    prompt: &str,
    max_tokens: usize,
    context: usize,
    raw: bool,
    effort: f32,
    batch: Option<usize>,
) -> Result<()> {
    let codec = codec(directory)?;
    let tokens = if raw {
        let mut tokenizer = codec.tokenizer().clone();
        tokenizer.set_encode_special_tokens(false);
        tokenizer
            .encode(prompt, false)
            .map_err(anyhow::Error::msg)?
            .get_ids()
            .to_vec()
    } else {
        codec.user_turn(prompt, true, effort)?
    };
    ensure!(
        !tokens.is_empty(),
        "prompt must encode to at least one token"
    );
    ensure!(
        tokens
            .len()
            .checked_add(max_tokens - 1)
            .context("token budget overflow")?
            <= context,
        "prompt and generation exceed --context"
    );
    let mut model = load(directory, context, batch)?;
    let mut next = prefill(&mut model, &tokens)?;
    let mut output = std::io::stdout().lock();
    let mut stream = codec.tokenizer().decode_stream(false);
    let start = Instant::now();
    let mut generated = 0;
    let mut decoded = 0;
    for i in 0..max_tokens {
        if codec.eos().contains(&next) {
            break;
        }
        if let Some(text) = stream.step(next).map_err(anyhow::Error::msg)? {
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
        decoded as f64 / seconds
    );
    Ok(())
}

pub fn chat(
    directory: &Path,
    max_tokens: usize,
    context: usize,
    effort: f32,
    show_thinking: bool,
    tools: ToolOptions,
    batch: Option<usize>,
) -> Result<()> {
    ensure!(
        tools.tool.is_empty(),
        "Bonsai tool-call parsing is not implemented; --tool is currently supported by Inkling"
    );
    let codec = codec(directory)?;
    let mut session = Session::with_eos(load(directory, context, batch)?, &codec.eos());
    let mut input = std::io::stdin().lock();
    let mut output = std::io::stdout().lock();
    let mut line = String::new();
    writeln!(
        output,
        "Rink chat — Bonsai 2.5. Enter a message and press Return."
    )?;
    writeln!(
        output,
        "/reset clears the conversation; /continue resumes a reply; /quit exits. Text input only."
    )?;
    let effort_name = if effort == 0.0 {
        "off"
    } else if effort < 1.0 / 3.0 {
        "low"
    } else if effort < 2.0 / 3.0 {
        "medium"
    } else {
        "xhigh"
    };
    writeln!(
        output,
        "Context: {context} tokens; reply limit: {max_tokens}; reasoning: {effort_name}; KV cache: f16."
    )?;
    let mut stream = codec.tokenizer().decode_stream(true);
    let mut thinking = effort > 0.0;
    let mut failed = false;
    loop {
        write!(output, "\nYou: ")?;
        output.flush()?;
        line.clear();
        if input.read_line(&mut line)? == 0 {
            writeln!(output)?;
            break;
        }
        let message = line.trim();
        match message {
            "" => continue,
            "/quit" | "/exit" => break,
            "/help" => {
                writeln!(
                    output,
                    "Enter one message per line. /reset clears history, /continue resumes a reply, /quit exits. Bonsai currently accepts text only. --show-thinking displays reasoning."
                )?;
                continue;
            }
            "/reset" => {
                session.reset()?;
                failed = false;
                stream = codec.tokenizer().decode_stream(true);
                writeln!(output, "Conversation cleared. The model is still loaded.")?;
                continue;
            }
            "/continue" => {
                if failed {
                    writeln!(output, "Use /reset after a conversation error.")?;
                    continue;
                }
                if let Err(e) = session.reply_budget(max_tokens) {
                    writeln!(output, "{e}")?;
                    continue;
                }
            }
            command if command.starts_with('/') => {
                writeln!(output, "Unknown or unsupported command. Use /help.")?;
                continue;
            }
            _ => {
                if failed || session.pending() {
                    writeln!(
                        output,
                        "The last reply is unfinished. Use /continue or /reset."
                    )?;
                    continue;
                }
                let tokens = codec.user_turn(message, session.position() == 0, effort)?;
                if let Err(e) = session.check_prompt(&tokens, max_tokens) {
                    writeln!(output, "{e}")?;
                    continue;
                }
                session.start_with(&tokens, max_tokens, prefill)?;
                stream = codec.tokenizer().decode_stream(true);
                thinking = effort > 0.0;
            }
        }
        let start = Instant::now();
        let before = session.position();
        let mut label = true;
        if thinking && !show_thinking {
            eprintln!("Thinking…");
        }
        for _ in 0..session.reply_budget(max_tokens)? {
            let token = match session.next_token() {
                Ok(Some(t)) => t,
                Ok(None) => break,
                Err(e) => {
                    writeln!(output, "\nGeneration failed: {e:#}. Use /reset.")?;
                    failed = true;
                    break;
                }
            };
            if token == codec.thinking_start() || token == codec.thinking_end() {
                thinking = token == codec.thinking_start();
                stream = codec.tokenizer().decode_stream(true);
                label = true;
                continue;
            }
            if (show_thinking || !thinking)
                && let Some(text) = stream.step(token).map_err(anyhow::Error::msg)?
            {
                if label {
                    write!(
                        output,
                        "\n{}: ",
                        if thinking { "Thinking" } else { "Assistant" }
                    )?;
                    label = false;
                }
                write!(output, "{text}")?;
                output.flush()?;
            }
        }
        writeln!(output)?;
        let n = session.position() - before;
        let seconds = start.elapsed().as_secs_f64();
        eprintln!(
            "Generation: {n} tokens in {seconds:.2}s ({:.1} tok/s); context {}/{context}",
            n as f64 / seconds,
            session.position()
        );
        if !failed && session.pending() {
            writeln!(output, "[Reply paused; use /continue or /reset.]")?;
        }
    }
    Ok(())
}
