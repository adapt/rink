use crate::{
    progress::{PrefillDisplay, StatusLine, rate},
    session::Session,
};
use anyhow::{Context, Result};
use rink_core::Accelerator;
use std::{
    io::{BufRead, Write},
    path::PathBuf,
    time::Instant,
};

#[derive(Clone, Copy)]
enum Channel {
    Hidden,
    Thinking,
    Text,
}

impl Channel {
    fn label(self, show_thinking: bool) -> Option<&'static str> {
        match self {
            Self::Thinking if show_thinking => Some("Thinking"),
            Self::Text => Some("Assistant"),
            _ => None,
        }
    }
}

pub fn run(
    directory: PathBuf,
    max_tokens: usize,
    context: usize,
    effort: f32,
    show_thinking: bool,
    kv_cache: rink_inkling::KvCachePrecision,
    prefill_batch: Option<usize>,
) -> Result<()> {
    let tokenizer = tokenizers::Tokenizer::from_file(directory.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("loading tokenizer: {e}"))?;
    let special = |name: &str| {
        tokenizer
            .token_to_id(name)
            .with_context(|| format!("tokenizer is missing {name}"))
    };
    let thinking_token = special("<|content_thinking|>")?;
    let text_token = special("<|content_text|>")?;
    let model_token = special("<|message_model|>")?;
    let end_token = special("<|end_message|>")?;
    let eos_token = special("<|content_model_end_sampling|>")?;
    let config = rink_inkling::Config::load(directory.join("config.json"))?;
    anyhow::ensure!(
        config.eos_token_id == eos_token,
        "tokenizer EOS does not match the model configuration"
    );
    let device = rink_metal::Metal::new()?;
    eprintln!("Loading Inkling 2 Mini on {}…", device.name());
    let mut model =
        rink_inkling::Inkling::load_with_kv_cache(device, &directory, context, kv_cache)?;
    if let Some(limit) = prefill_batch {
        model.set_prefill_batch_limit(limit)?;
    }
    let prefill_batch = model.prefill_batch_limit();
    let mut session = Session::new(model, eos_token);
    let mut input = std::io::stdin().lock();
    let mut output = std::io::stdout().lock();
    writeln!(output, "Rink chat — enter a message and press Return.")?;
    writeln!(
        output,
        "/image PATH attaches a picture; /reset clears the conversation; /continue resumes a reply; /quit exits."
    )?;
    writeln!(
        output,
        "Context: {context} tokens; reply limit: {max_tokens}; effort: {effort}; KV cache: {kv_cache}; prefill batch: {prefill_batch}."
    )?;
    let mut line = String::new();
    let mut stream = tokenizer.decode_stream(true);
    let mut channel = Channel::Hidden;
    let mut pending_image: Option<rink_inkling::Image> = None;
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
                    "Enter one message per line. /image PATH attaches a PNG or JPEG to your next message; /image clear removes it. /reset clears history, /continue finishes a truncated reply, /quit exits. Thinking is shown with --show-thinking."
                )?;
                continue;
            }
            "/reset" => {
                session.reset()?;
                pending_image = None;
                channel = Channel::Hidden;
                stream = tokenizer.decode_stream(true);
                writeln!(output, "Conversation cleared. The model is still loaded.")?;
                continue;
            }
            "/continue" => {
                if let Err(error) = session.reply_budget(max_tokens) {
                    writeln!(output, "{error}")?;
                    continue;
                }
            }
            "/image clear" => {
                pending_image = None;
                writeln!(output, "Pending image cleared.")?;
                continue;
            }
            "/image" => {
                writeln!(
                    output,
                    "Use /image PATH to attach a PNG or JPEG, then enter your question. /image clear removes the attachment."
                )?;
                continue;
            }
            command if command.starts_with("/image ") => {
                if session.pending() {
                    writeln!(
                        output,
                        "The last reply is unfinished. Use /continue or /reset."
                    )?;
                    continue;
                }
                if config.vision_config.is_none()
                    || tokenizer.token_to_id(rink_inkling::image::IMAGE_TOKEN)
                        != Some(config.image_token_id)
                {
                    writeln!(
                        output,
                        "This checkpoint or tokenizer does not support image input."
                    )?;
                    continue;
                }
                let path = command[7..].trim();
                let path = path
                    .strip_prefix('"')
                    .and_then(|p| p.strip_suffix('"'))
                    .or_else(|| path.strip_prefix('\'').and_then(|p| p.strip_suffix('\'')))
                    .unwrap_or(path);
                match rink_inkling::Image::open(path) {
                    Ok(image) => {
                        writeln!(
                            output,
                            "Attached {} × {} image ({} context tokens). Enter your question next.",
                            image.width(),
                            image.height(),
                            image.token_count()
                        )?;
                        pending_image = Some(image);
                    }
                    Err(error) => writeln!(output, "Cannot attach image: {error:#}")?,
                }
                continue;
            }
            command if command.starts_with('/') => {
                writeln!(output, "Unknown command. Use /help for available commands.")?;
                continue;
            }
            _ => {
                if session.pending() {
                    writeln!(
                        output,
                        "The last reply is unfinished. Use /continue or /reset."
                    )?;
                    continue;
                }
                let rendered = if let Some(image) = &pending_image {
                    if session.position() == 0 {
                        rink_inkling::image::image_prompt(message, image.token_count(), effort)?
                    } else {
                        rink_inkling::image::image_user_message(message, image.token_count())
                    }
                } else if session.position() == 0 {
                    rink_inkling::chat_prompt(message, effort)?
                } else {
                    rink_inkling::chat_user_message(message)
                };
                let encoded = tokenizer
                    .encode(rendered, false)
                    .map_err(|e| anyhow::anyhow!("tokenizing message: {e}"))?;
                if let Err(error) = session.check_prompt(encoded.get_ids(), max_tokens) {
                    writeln!(output, "{error}")?;
                    continue;
                }
                let mut prefill = PrefillDisplay::new();
                if let Some(image) = &pending_image {
                    session.start_with(encoded.get_ids(), max_tokens, |model, tokens| {
                        model.prefill_image_greedy_with_progress(tokens, image, |event| {
                            prefill.observe(event);
                        })
                    })?;
                } else {
                    session.start_with(encoded.get_ids(), max_tokens, |model, tokens| {
                        model.prefill_greedy_with_progress(tokens, |event| prefill.observe(event))
                    })?;
                }
                pending_image = None;
                stream = tokenizer.decode_stream(true);
                channel = Channel::Hidden;
            }
        }

        // Keep both the decoder and the channel across /continue, including a
        // partially decoded UTF-8 character at the token limit.
        let mut label = channel.label(show_thinking);
        let start = Instant::now();
        let before = session.position();
        let mut status = StatusLine::new();
        status.update("Generating…", true);
        for _ in 0..session.reply_budget(max_tokens)? {
            let Some(token) = session.next_token()? else {
                break;
            };
            let boundary = if token == thinking_token {
                Some(Channel::Thinking)
            } else if token == text_token {
                Some(Channel::Text)
            } else if token == model_token || token == end_token {
                Some(Channel::Hidden)
            } else {
                None
            };
            if let Some(next_channel) = boundary {
                channel = next_channel;
                label = channel.label(show_thinking);
                stream = tokenizer.decode_stream(true);
                if matches!(channel, Channel::Thinking) && !show_thinking {
                    status.update("Thinking…", true);
                }
            } else if channel.label(show_thinking).is_some() {
                if let Some(text) = stream
                    .step(token)
                    .map_err(|e| anyhow::anyhow!("decoding token: {e}"))?
                {
                    status.clear();
                    if let Some(label) = label.take() {
                        write!(output, "\n{label}: ")?;
                    }
                    write!(output, "{text}")?;
                    output.flush()?;
                }
            }
            if matches!(channel, Channel::Thinking) && !show_thinking {
                status.update(
                    &format!(
                        "Thinking… {} tokens generated; {:.1}s",
                        session.position() - before,
                        start.elapsed().as_secs_f64()
                    ),
                    false,
                );
            }
        }
        status.clear();
        writeln!(output)?;
        if session.pending() {
            if session.position() == context {
                writeln!(
                    output,
                    "[Context full; use /reset to start a new conversation.]"
                )?;
            } else {
                writeln!(
                    output,
                    "[Reply paused at the token limit; use /continue or /reset.]"
                )?;
            }
        }
        output.flush()?;
        let generated = session.position() - before;
        let seconds = start.elapsed().as_secs_f64();
        eprintln!(
            "Generation: {generated} tokens in {seconds:.2}s ({:.1} tok/s); context {}/{context}",
            rate(generated, seconds),
            session.position()
        );
    }
    Ok(())
}
