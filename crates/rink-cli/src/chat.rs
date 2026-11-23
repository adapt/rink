use crate::{
    ToolOptions,
    agent::{ToolEvent, ToolLoop},
    progress::{PrefillDisplay, StatusLine, rate},
    session::Session,
};
use anyhow::Result;
use rink_core::Accelerator;
use rink_inkling::chat::{Channel as NativeChannel, Codec, Event};
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
    Tool,
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

#[allow(clippy::too_many_arguments)]
pub fn run(
    directory: PathBuf,
    max_tokens: usize,
    context: usize,
    effort: f32,
    show_thinking: bool,
    kv_cache: rink_inkling::KvCachePrecision,
    prefill_batch: Option<usize>,
    tools: ToolOptions,
) -> Result<()> {
    let tokenizer = tokenizers::Tokenizer::from_file(directory.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("loading tokenizer: {e}"))?;
    let codec = Codec::new(tokenizer)?;
    let eos_token = codec.eos();
    let mut registry = rink_tools::Registry::default();
    if !tools.tool.is_empty() {
        let cwd = tools
            .cwd
            .unwrap_or(std::env::current_dir()?)
            .canonicalize()?;
        let timeout = std::time::Duration::from_millis(tools.tool_timeout_ms);
        if tools.tool.iter().any(|name| name == "bash") {
            registry.register(rink_tools::Bash::new(
                &cwd,
                timeout,
                tools.tool_output_bytes as usize,
            )?)?;
        }
        if tools.tool.iter().any(|name| name == "read") {
            registry.register(rink_tools::Read::new(
                &cwd,
                timeout,
                tools.tool_output_bytes as usize,
            )?)?;
        }
        eprintln!(
            "Tools {} enabled with direct host access in {}. Ctrl-C cancels a running tool.",
            tools.tool.join(", "),
            cwd.display()
        );
    }
    let definitions = registry.definitions();
    let config = rink_inkling::Config::load(directory.join("config.json"))?;
    let images_supported =
        config.vision_config.is_some() && codec.image_token() == config.image_token_id;
    let mut tool_loop = ToolLoop::new(registry, tools.max_tool_rounds as usize, images_supported);
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
    let mut stream = codec.tokenizer().decode_stream(true);
    let mut parser = codec.parser();
    let mut needs_reset = false;
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
                parser = codec.parser();
                tool_loop.reset();
                needs_reset = false;
                pending_image = None;
                channel = Channel::Hidden;
                stream = codec.tokenizer().decode_stream(true);
                writeln!(output, "Conversation cleared. The model is still loaded.")?;
                continue;
            }
            "/continue" => {
                if needs_reset {
                    writeln!(output, "Use /reset after a conversation error.")?;
                    continue;
                }
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
                if session.pending() || needs_reset {
                    writeln!(
                        output,
                        "The last reply is unfinished. Use /continue or /reset."
                    )?;
                    continue;
                }
                if config.vision_config.is_none() || codec.image_token() != config.image_token_id {
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
                if session.pending() || needs_reset {
                    writeln!(
                        output,
                        "The last reply is unfinished. Use /continue or /reset."
                    )?;
                    continue;
                }
                let encoded = codec.user_turn(
                    message,
                    pending_image.as_ref().map(rink_inkling::Image::token_count),
                    (session.position() == 0).then_some(effort),
                    &definitions,
                )?;
                if let Err(error) = session.check_prompt(&encoded, max_tokens) {
                    writeln!(output, "{error}")?;
                    continue;
                }
                let mut prefill = PrefillDisplay::new();
                if let Some(image) = &pending_image {
                    session.start_with(&encoded, max_tokens, |model, tokens| {
                        model.prefill_image_greedy_with_progress(tokens, image, |event| {
                            prefill.observe(event);
                        })
                    })?;
                } else {
                    session.start_with(&encoded, max_tokens, |model, tokens| {
                        model.prefill_greedy_with_progress(tokens, |event| prefill.observe(event))
                    })?;
                }
                pending_image = None;
                parser = codec.parser();
                tool_loop.reset();
                stream = codec.tokenizer().decode_stream(true);
                channel = Channel::Hidden;
            }
        }

        // Keep the UTF-8 decoder and native parser across /continue. Tool calls
        // remain buffered until EOS, including when a pause splits their JSON.
        let mut remaining = max_tokens;
        let mut tool_pause = false;
        loop {
            if remaining == 0 {
                break;
            }
            let mut label = channel.label(show_thinking);
            let start = Instant::now();
            let before = session.position();
            let mut status = StatusLine::new();
            let mut complete = false;
            status.update("Generating…", true);
            for _ in 0..session.reply_budget(remaining)? {
                let Some(token) = session.next_token()? else {
                    complete = true;
                    break;
                };
                let event = match parser.push(token, codec.tokenizer()) {
                    Ok(event) => event,
                    Err(error) => {
                        status.clear();
                        writeln!(output, "\nInvalid model response: {error:#}. Use /reset.")?;
                        needs_reset = true;
                        break;
                    }
                };
                match event {
                    Event::Boundary(next) => {
                        channel = match next {
                            Some(NativeChannel::Thinking) => Channel::Thinking,
                            Some(NativeChannel::Text) => Channel::Text,
                            Some(NativeChannel::Tool) => Channel::Tool,
                            None => Channel::Hidden,
                        };
                        label = channel.label(show_thinking);
                        stream = codec.tokenizer().decode_stream(true);
                        if matches!(channel, Channel::Tool) {
                            status.update("Preparing tool call…", true);
                        }
                    }
                    Event::Token(_, token) if channel.label(show_thinking).is_some() => {
                        if let Some(text) = stream.step(token).map_err(anyhow::Error::msg)? {
                            status.clear();
                            if let Some(label) = label.take() {
                                write!(output, "\n{label}: ")?;
                            }
                            write!(output, "{text}")?;
                            output.flush()?;
                        }
                    }
                    _ => {}
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
            let generated = session.position() - before;
            remaining = remaining.saturating_sub(generated);
            let seconds = start.elapsed().as_secs_f64();
            eprintln!(
                "Generation: {generated} tokens in {seconds:.2}s ({:.1} tok/s); context {}/{context}",
                rate(generated, seconds),
                session.position()
            );
            if needs_reset || !complete {
                break;
            }
            let calls = match parser.finish() {
                Ok(calls) => calls,
                Err(error) => {
                    writeln!(output, "Invalid model response: {error:#}. Use /reset.")?;
                    needs_reset = true;
                    break;
                }
            };
            if calls.is_empty() {
                break;
            }
            let interrupt = rink_tools::Interrupt::install()?;
            let result = tool_loop.run(
                &codec,
                calls,
                session.remaining_capacity().saturating_sub(1),
                interrupt.flag(),
                |event| display_tool(&mut output, event),
            );
            drop(interrupt);
            let result = match result {
                Ok(result) => result,
                Err(error) => {
                    writeln!(output, "Cannot continue tool round: {error:#}. Use /reset.")?;
                    needs_reset = true;
                    break;
                }
            };
            let mut prefill = PrefillDisplay::new();
            let pause = result.pause;
            session.start_append(result.token_budget(), 1, |model| {
                result.prefill(|tokens, image| match image {
                    Some(image) => {
                        model.prefill_image_greedy_with_progress(tokens, image, |event| {
                            prefill.observe(event)
                        })
                    }
                    None => {
                        model.prefill_greedy_with_progress(tokens, |event| prefill.observe(event))
                    }
                })
            })?;
            parser = codec.parser();
            stream = codec.tokenizer().decode_stream(true);
            channel = Channel::Hidden;
            if let Some(reason) = pause {
                writeln!(output, "[{reason}]")?;
                tool_pause = true;
                break;
            }
        }
        if !needs_reset && session.pending() && !tool_pause {
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
    }
    Ok(())
}

fn terminal_text(text: &str) -> String {
    let mut escaped = String::with_capacity(text.len());
    for c in text.chars() {
        if c.is_control() && c != '\n' && c != '\t' {
            escaped.extend(c.escape_default());
        } else {
            escaped.push(c);
        }
    }
    escaped
}

fn display_tool(output: &mut impl Write, event: ToolEvent<'_>) -> Result<()> {
    match event {
        ToolEvent::Starting(call) => {
            let arguments = match &call.arguments {
                Ok(args) => args
                    .get("command")
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_owned)
                    .unwrap_or_else(|| args.to_string()),
                Err(_) => "[invalid arguments]".into(),
            };
            writeln!(
                output,
                "\nTool {}: {}",
                call.name,
                terminal_text(&arguments)
            )?;
        }
        ToolEvent::Notice(text) => writeln!(output, "{}", terminal_text(text))?,
        ToolEvent::Finished(name, result) => {
            for part in &result.content {
                let text = match part {
                    rink_tools::ContentPart::Text(text) => text,
                    rink_tools::ContentPart::Image(image) => {
                        writeln!(
                            output,
                            "[Image queued: {}]",
                            terminal_text(&image.path.to_string_lossy())
                        )?;
                        continue;
                    }
                };
                let value: serde_json::Value = serde_json::from_str(text).unwrap_or_default();
                if name == "bash" && value.get("stdout").is_some() {
                    for key in ["stdout", "stderr"] {
                        if let Some(text) = value[key].as_str().filter(|s| !s.is_empty()) {
                            writeln!(output, "{}", terminal_text(text))?;
                        }
                    }
                    writeln!(
                        output,
                        "[bash: exit {}; signal {}; timeout {}; cancelled {}; output truncated {}]",
                        value["exit_code"],
                        value["signal"],
                        value["timed_out"],
                        value["cancelled"],
                        value["stdout_truncated"] == true || value["stderr_truncated"] == true
                    )?;
                } else {
                    writeln!(output, "{}", terminal_text(text))?;
                }
            }
        }
    }
    output.flush()?;
    Ok(())
}
