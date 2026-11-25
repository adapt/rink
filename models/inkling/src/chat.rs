//! Inkling's native message framing and incremental response parser.
//! The released chat template defines tool declarations, named invocations and
//! tool replies: https://huggingface.co/thinkingmachines/Inkling-2-Mini/blob/main/chat_template.jinja
use anyhow::{Context, Result, bail, ensure};
pub use rink_core::chat::Invocation;
use rink_core::chat::ToolCodec;
use serde::Deserialize;
use serde_json::Value;
use std::collections::HashSet;
use tokenizers::Tokenizer;

#[derive(Clone)]
struct Special {
    user: u32,
    model: u32,
    system: u32,
    tool: u32,
    text: u32,
    thinking: u32,
    xml: u32,
    invoke_json: u32,
    image: u32,
    patch: u32,
    end: u32,
    eos: u32,
    all: HashSet<u32>,
}

pub struct Codec {
    tokenizer: Tokenizer,
    special: Special,
}

impl Codec {
    pub fn new(mut tokenizer: Tokenizer) -> Result<Self> {
        let id = |s: &str| {
            tokenizer
                .token_to_id(s)
                .with_context(|| format!("tokenizer is missing {s}"))
        };
        let special = Special {
            user: id("<|message_user|>")?,
            model: id("<|message_model|>")?,
            system: id("<|message_system|>")?,
            tool: id("<|message_tool|>")?,
            text: id("<|content_text|>")?,
            thinking: id("<|content_thinking|>")?,
            xml: id("<|content_xml|>")?,
            invoke_json: id("<|content_invoke_tool_json|>")?,
            image: id("<|content_image|>")?,
            patch: id(crate::image::IMAGE_TOKEN)?,
            end: id("<|end_message|>")?,
            eos: id("<|content_model_end_sampling|>")?,
            all: tokenizer
                .get_added_tokens_decoder()
                .into_iter()
                .filter_map(|(id, t)| t.special.then_some(id))
                .collect(),
        };
        // In tokenizers this flag makes special-token spellings ordinary input.
        // Framing is inserted as IDs; user text and tool output cannot inject it.
        tokenizer.set_encode_special_tokens(true);
        tokenizer.with_padding(None);
        tokenizer
            .with_truncation(None)
            .map_err(anyhow::Error::msg)?;
        Ok(Self { tokenizer, special })
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }
    pub fn eos(&self) -> u32 {
        self.special.eos
    }
    pub fn image_token(&self) -> u32 {
        self.special.patch
    }
    pub fn parser(&self) -> ResponseParser {
        ResponseParser {
            special: self.special.clone(),
            state: State::Idle,
            calls: Vec::new(),
        }
    }

    fn ordinary(&self, out: &mut Vec<u32>, text: &str) -> Result<()> {
        let encoded = self
            .tokenizer
            .encode(text, false)
            .map_err(anyhow::Error::msg)?;
        ensure!(
            !encoded
                .get_ids()
                .iter()
                .any(|id| self.special.all.contains(id)),
            "ordinary text produced a protocol token"
        );
        out.extend_from_slice(encoded.get_ids());
        Ok(())
    }

    fn message(
        &self,
        out: &mut Vec<u32>,
        author: u32,
        name: &str,
        kind: u32,
        text: &str,
    ) -> Result<()> {
        out.push(author);
        self.ordinary(out, name)?;
        out.push(kind);
        self.ordinary(out, text)?;
        out.push(self.special.end);
        Ok(())
    }

    /// Effort and tool declarations appear only at the start (and after reset).
    pub fn user_turn(
        &self,
        prompt: &str,
        image_tokens: Option<usize>,
        effort: Option<f32>,
        tools: &[Value],
    ) -> Result<Vec<u32>> {
        let mut out = Vec::new();
        if let Some(effort) = effort {
            ensure!(
                effort.is_finite() && (0.0..=0.99).contains(&effort),
                "reasoning effort must be in 0..=0.99"
            );
            if !tools.is_empty() {
                self.message(
                    &mut out,
                    self.special.system,
                    "tool_declare",
                    self.special.xml,
                    &serde_json::to_string(tools)?,
                )?;
            }
            self.message(
                &mut out,
                self.special.system,
                "",
                self.special.text,
                &format!("Thinking effort level: {effort}"),
            )?;
        }
        if let Some(tokens) = image_tokens {
            ensure!(
                tokens > 0
                    && tokens
                        <= (crate::image::MAX_IMAGE_DIMENSION as usize)
                            .div_ceil(40)
                            .pow(2),
                "invalid image token count"
            );
            out.extend([self.special.user, self.special.image]);
            out.extend(std::iter::repeat_n(self.special.patch, tokens));
            out.push(self.special.end);
        }
        self.message(&mut out, self.special.user, "", self.special.text, prompt)?;
        Ok(out)
    }

    pub fn tool_result(&self, name: &str, content: &str) -> Result<Vec<u32>> {
        ensure!(valid_name(name), "invalid tool name");
        let mut out = Vec::new();
        self.message(
            &mut out,
            self.special.tool,
            name,
            self.special.text,
            content,
        )?;
        Ok(out)
    }

    /// The release template permits image parts on named tool messages. The
    /// processor expands the image marker to one placeholder per visual tile.
    pub fn tool_image(&self, name: &str, image_tokens: usize) -> Result<Vec<u32>> {
        ensure!(valid_name(name), "invalid tool name");
        ensure!(
            image_tokens > 0
                && image_tokens
                    <= (crate::image::MAX_IMAGE_DIMENSION as usize)
                        .div_ceil(40)
                        .pow(2),
            "invalid image token count"
        );
        let mut out = vec![self.special.tool];
        self.ordinary(&mut out, name)?;
        out.push(self.special.image);
        out.extend(std::iter::repeat_n(self.special.patch, image_tokens));
        out.push(self.special.end);
        Ok(out)
    }
}

fn valid_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 128
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"_.-".contains(&b))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Channel {
    Thinking,
    Text,
    Tool,
}

#[derive(Debug)]
pub enum Event {
    Boundary(Option<Channel>),
    Token(Channel, u32),
}

impl ToolCodec for Codec {
    fn tool_result(&self, name: &str, content: &str) -> Result<Vec<u32>> {
        Codec::tool_result(self, name, content)
    }
    fn tool_image(&self, name: &str, image_tokens: usize) -> Result<Vec<u32>> {
        Codec::tool_image(self, name, image_tokens)
    }
}

enum State {
    Idle,
    Header(Vec<u32>),
    Body {
        channel: Channel,
        name: String,
        tokens: Vec<u32>,
    },
    Failed,
}

pub struct ResponseParser {
    special: Special,
    state: State,
    calls: Vec<Invocation>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Payload {
    name: String,
    args: Value,
}

impl ResponseParser {
    /// Feed generated tokens only. Keep this parser across token-limit pauses.
    /// EOS is consumed by the session; call finish only when sampling has ended.
    pub fn push(&mut self, token: u32, tokenizer: &Tokenizer) -> Result<Event> {
        let previous = std::mem::replace(&mut self.state, State::Failed);
        let decode = |tokens: &[u32]| tokenizer.decode(tokens, false).map_err(anyhow::Error::msg);
        let (next, event) = match previous {
            State::Idle => {
                ensure!(token == self.special.model, "expected an assistant message");
                (State::Header(Vec::new()), Event::Boundary(None))
            }
            State::Header(mut tokens) => {
                let channel = if token == self.special.text {
                    Some(Channel::Text)
                } else if token == self.special.thinking {
                    Some(Channel::Thinking)
                } else if token == self.special.invoke_json {
                    Some(Channel::Tool)
                } else {
                    None
                };
                if let Some(channel) = channel {
                    let name = decode(&tokens)?;
                    ensure!(
                        if channel == Channel::Tool {
                            valid_name(&name)
                        } else {
                            name.is_empty()
                        },
                        "invalid assistant message recipient"
                    );
                    (
                        State::Body {
                            channel,
                            name,
                            tokens: Vec::new(),
                        },
                        Event::Boundary(Some(channel)),
                    )
                } else {
                    ensure!(
                        !self.special.all.contains(&token) && tokens.len() < 128,
                        "invalid or oversized assistant header"
                    );
                    tokens.push(token);
                    (State::Header(tokens), Event::Boundary(None))
                }
            }
            State::Body {
                channel,
                name,
                mut tokens,
            } => {
                if token == self.special.end {
                    if channel == Channel::Tool {
                        ensure!(self.calls.len() < 32, "too many tool calls in one response");
                        let text = decode(&tokens)?;
                        let arguments = (|| -> Result<Value> {
                            ensure!(text.len() <= 65_536, "tool arguments exceed 65536 bytes");
                            let payload: Payload =
                                serde_json::from_str(&text).context("invalid tool-call JSON")?;
                            ensure!(
                                payload.name == name,
                                "tool name differs between header and payload"
                            );
                            ensure!(payload.args.is_object(), "tool arguments must be an object");
                            Ok(payload.args)
                        })()
                        .map_err(|e| format!("{e:#}"));
                        self.calls.push(Invocation { name, arguments });
                    }
                    (State::Idle, Event::Boundary(None))
                } else {
                    ensure!(
                        !self.special.all.contains(&token),
                        "unexpected protocol token inside assistant content"
                    );
                    if channel == Channel::Tool {
                        ensure!(
                            tokens.len() < 65_536,
                            "tool arguments exceed the token limit"
                        );
                        tokens.push(token);
                    }
                    (
                        State::Body {
                            channel,
                            name,
                            tokens,
                        },
                        Event::Token(channel, token),
                    )
                }
            }
            State::Failed => bail!("response parser must be reset after a protocol error"),
        };
        self.state = next;
        Ok(event)
    }

    /// No call is released for execution before the complete response reaches EOS.
    pub fn finish(&mut self) -> Result<Vec<Invocation>> {
        if !matches!(self.state, State::Idle) {
            self.state = State::Failed;
            self.calls.clear();
            bail!("sampling ended inside an incomplete message");
        }
        Ok(std::mem::take(&mut self.calls))
    }
}
