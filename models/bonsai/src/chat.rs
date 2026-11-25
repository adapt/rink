//! Bonsai's native chat template and XML-style function calls.
//! Framing belongs to this model; host tool execution is shared by the CLI.
use anyhow::{Context, Result, bail, ensure};
use rink_core::chat::{Invocation, ToolCodec};
use serde_json::{Map, Value, json};
use std::collections::{HashMap, HashSet};
use tokenizers::{AddedToken, Tokenizer};

pub struct Codec {
    tokenizer: Tokenizer,
    start: u32,
    end: u32,
    end_text: u32,
    think: u32,
    end_think: u32,
    call: u32,
    end_call: u32,
    response: u32,
    end_response: u32,
    special: HashSet<u32>,
}
impl Codec {
    pub fn new(mut tokenizer: Tokenizer) -> Result<Self> {
        let id = |name: &str| {
            tokenizer
                .token_to_id(name)
                .with_context(|| format!("Bonsai tokenizer is missing {name}"))
        };
        let (start, end, end_text, think, end_think, call, end_call, response, end_response) = (
            id("<|im_start|>")?,
            id("<|im_end|>")?,
            id("<|endoftext|>")?,
            id("<think>")?,
            id("</think>")?,
            id("<tool_call>")?,
            id("</tool_call>")?,
            id("<tool_response>")?,
            id("</tool_response>")?,
        );
        // The release marks these added tokens as non-special. Promote them in
        // this tokenizer so ordinary user/tool text cannot insert control IDs.
        tokenizer.add_special_tokens(
            &[
                "<think>",
                "</think>",
                "<tool_call>",
                "</tool_call>",
                "<tool_response>",
                "</tool_response>",
            ]
            .map(|s| AddedToken::from(s, true)),
        );
        let special = tokenizer
            .get_added_tokens_decoder()
            .into_iter()
            .filter_map(|(id, t)| t.special.then_some(id))
            .collect();
        tokenizer.set_encode_special_tokens(true);
        tokenizer.with_padding(None);
        tokenizer
            .with_truncation(None)
            .map_err(anyhow::Error::msg)?;
        Ok(Self {
            tokenizer,
            start,
            end,
            end_text,
            think,
            end_think,
            call,
            end_call,
            response,
            end_response,
            special,
        })
    }
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }
    pub fn eos(&self) -> [u32; 2] {
        [self.end, self.end_text]
    }
    pub fn thinking_start(&self) -> u32 {
        self.think
    }
    pub fn thinking_end(&self) -> u32 {
        self.end_think
    }
    fn text(&self, out: &mut Vec<u32>, text: &str) -> Result<()> {
        let e = self
            .tokenizer
            .encode(text, false)
            .map_err(anyhow::Error::msg)?;
        ensure!(
            !e.get_ids().iter().any(|id| self.special.contains(id)),
            "ordinary text produced a protocol token"
        );
        out.extend_from_slice(e.get_ids());
        Ok(())
    }
    fn message(&self, out: &mut Vec<u32>, role: &str, text: &str) -> Result<()> {
        out.push(self.start);
        self.text(out, &format!("{role}\n"))?;
        self.text(out, text.trim())?;
        out.push(self.end);
        self.text(out, "\n")
    }
    fn assistant(&self, out: &mut Vec<u32>, effort: f32) -> Result<()> {
        out.push(self.start);
        self.text(out, "assistant\n")?;
        out.push(self.think);
        if effort == 0.0 {
            self.text(out, "\n\n")?;
            out.push(self.end_think);
            self.text(out, "\n\n")?;
        } else {
            self.text(out, "\n")?;
        }
        Ok(())
    }
    /// Effort 0 disables thinking; positive effort selects low (<1/3), medium
    /// (<2/3), or xhigh. Unlike Inkling, this model has categorical effort levels.
    pub fn user_turn(&self, prompt: &str, first: bool, effort: f32) -> Result<Vec<u32>> {
        self.user_turn_with_tools(prompt, first, effort, &[])
    }
    pub fn user_turn_with_tools(
        &self,
        prompt: &str,
        first: bool,
        effort: f32,
        tools: &[Value],
    ) -> Result<Vec<u32>> {
        ensure!(
            effort.is_finite() && (0.0..=0.99).contains(&effort),
            "reasoning effort must be in 0..=0.99"
        );
        let mut out = Vec::new();
        if first {
            let instructions = match effort {
                e if e >= 2.0 / 3.0 => {
                    "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer."
                }
                e if e > 0.0 && e < 1.0 / 3.0 => {
                    "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration."
                }
                _ => "",
            };
            if tools.is_empty() {
                if !instructions.is_empty() {
                    self.message(&mut out, "system", instructions)?;
                }
            } else {
                out.push(self.start);
                self.text(&mut out, "system\n")?;
                if !instructions.is_empty() {
                    self.text(&mut out, instructions)?;
                    self.text(&mut out, "\n\n")?;
                }
                self.text(
                    &mut out,
                    "# Tools\n\nYou have access to the following functions:\n\n<tools>",
                )?;
                for tool in tools {
                    let mut function = tool.clone();
                    function
                        .as_object_mut()
                        .context("tool definition must be an object")?
                        .remove("type");
                    self.text(&mut out, "\n")?;
                    self.text(
                        &mut out,
                        &template_json(&json!({"type":"function", "function":function}))?,
                    )?;
                }
                self.text(&mut out, "\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n")?;
                out.push(self.call);
                self.text(&mut out, "\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n")?;
                out.push(self.end_call);
                self.text(&mut out, "\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within ")?;
                out.extend([self.call, self.end_call]);
                self.text(&mut out, " XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>")?;
                out.push(self.end);
                self.text(&mut out, "\n")?;
            }
        } else {
            self.text(&mut out, "\n")?;
        }
        self.message(&mut out, "user", prompt)?;
        self.assistant(&mut out, effort)?;
        Ok(out)
    }
    pub fn tool_codec(&self, effort: f32) -> ToolFormat<'_> {
        ToolFormat {
            codec: self,
            effort,
        }
    }
    pub fn parser(&self, effort: f32, tools: &[Value]) -> ResponseParser {
        ResponseParser {
            think: self.think,
            end_think: self.end_think,
            call: self.call,
            end_call: self.end_call,
            special: self.special.clone(),
            state: if effort > 0.0 {
                State::Thinking
            } else {
                State::Text
            },
            calls: Vec::new(),
            schemas: tools
                .iter()
                .filter_map(|t| Some((t["name"].as_str()?.to_owned(), t["parameters"].clone())))
                .collect(),
        }
    }
}

// Match the template's JSON spacing without changing strings (including shell
// syntax, escaped quotes and Unicode). serde_json already sorts object keys.
fn template_json(value: &Value) -> Result<String> {
    let compact = serde_json::to_string(value)?;
    let mut out = String::with_capacity(compact.len());
    let (mut string, mut escaped) = (false, false);
    for c in compact.chars() {
        out.push(c);
        if string {
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                string = false;
            }
        } else if c == '"' {
            string = true;
        } else if matches!(c, ',' | ':') {
            out.push(' ');
        }
    }
    Ok(out)
}

pub struct ToolFormat<'a> {
    codec: &'a Codec,
    effort: f32,
}
impl ToolCodec for ToolFormat<'_> {
    fn tool_round_start(&self) -> Result<Vec<u32>> {
        let c = self.codec;
        let mut out = Vec::new();
        c.text(&mut out, "\n")?;
        out.push(c.start);
        c.text(&mut out, "user")?;
        Ok(out)
    }
    fn tool_result(&self, _name: &str, content: &str) -> Result<Vec<u32>> {
        let c = self.codec;
        let mut out = Vec::new();
        c.text(&mut out, "\n")?;
        out.push(c.response);
        c.text(&mut out, "\n")?;
        c.text(&mut out, content.trim())?;
        c.text(&mut out, "\n")?;
        out.push(c.end_response);
        Ok(out)
    }
    fn tool_round_end(&self) -> Result<Vec<u32>> {
        let c = self.codec;
        let mut out = vec![c.end];
        c.text(&mut out, "\n")?;
        c.assistant(&mut out, self.effort)?;
        Ok(out)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Channel {
    Thinking,
    Text,
    Tool,
}
pub enum Event {
    Boundary(Channel),
    Token(Channel, u32),
}
enum State {
    Thinking,
    Text,
    Tool(Vec<u32>),
    AfterTool(Vec<u32>),
    Failed,
    Finished,
}
pub struct ResponseParser {
    think: u32,
    end_think: u32,
    call: u32,
    end_call: u32,
    special: HashSet<u32>,
    schemas: HashMap<String, Value>,
    state: State,
    calls: Vec<Invocation>,
}
impl ResponseParser {
    /// Keep across /continue. Tool calls are released only after the session
    /// consumes EOS; an incomplete or invalid response executes nothing.
    pub fn push(&mut self, token: u32, tokenizer: &Tokenizer) -> Result<Event> {
        let previous = std::mem::replace(&mut self.state, State::Failed);
        let (state, event) = match previous {
            State::Thinking if token == self.end_think => {
                (State::Text, Event::Boundary(Channel::Text))
            }
            State::Thinking => {
                ensure!(
                    !self.special.contains(&token) || token == self.call || token == self.end_call,
                    "unexpected protocol token in reasoning"
                );
                (State::Thinking, Event::Token(Channel::Thinking, token))
            }
            State::Text if token == self.think => {
                (State::Thinking, Event::Boundary(Channel::Thinking))
            }
            State::Text if token == self.call => {
                (State::Tool(Vec::new()), Event::Boundary(Channel::Tool))
            }
            State::Text => {
                ensure!(
                    !self.special.contains(&token),
                    "unexpected protocol token in assistant text"
                );
                (State::Text, Event::Token(Channel::Text, token))
            }
            State::Tool(tokens) if token == self.end_call => {
                ensure!(self.calls.len() < 32, "too many tool calls in one response");
                let text = tokenizer
                    .decode(&tokens, false)
                    .map_err(anyhow::Error::msg)?;
                ensure!(text.len() <= 65_536, "tool call exceeds 65536 bytes");
                let (name, body) = function_header(&text)?;
                let arguments =
                    arguments(body, self.schemas.get(&name)).map_err(|e| format!("{e:#}"));
                self.calls.push(Invocation { name, arguments });
                (State::AfterTool(Vec::new()), Event::Boundary(Channel::Tool))
            }
            State::Tool(mut tokens) => {
                ensure!(
                    !self.special.contains(&token),
                    "unexpected protocol token in tool arguments"
                );
                ensure!(tokens.len() < 65_536, "tool call exceeds token limit");
                tokens.push(token);
                (State::Tool(tokens), Event::Token(Channel::Tool, token))
            }
            State::AfterTool(tokens) if token == self.call => {
                ensure!(
                    tokenizer
                        .decode(&tokens, false)
                        .map_err(anyhow::Error::msg)?
                        .trim()
                        .is_empty(),
                    "unexpected content after tool call"
                );
                (State::Tool(Vec::new()), Event::Boundary(Channel::Tool))
            }
            State::AfterTool(mut tokens) => {
                ensure!(
                    !self.special.contains(&token) && tokens.len() < 1024,
                    "unexpected content after tool call"
                );
                tokens.push(token);
                (State::AfterTool(tokens), Event::Token(Channel::Tool, token))
            }
            State::Failed | State::Finished => bail!("response parser must be reset"),
        };
        self.state = state;
        Ok(event)
    }
    pub fn finish(&mut self, tokenizer: &Tokenizer) -> Result<Vec<Invocation>> {
        let previous = std::mem::replace(&mut self.state, State::Failed);
        match previous {
            State::Text => {}
            State::AfterTool(tokens) => ensure!(
                tokenizer
                    .decode(&tokens, false)
                    .map_err(anyhow::Error::msg)?
                    .trim()
                    .is_empty(),
                "unexpected content after tool call"
            ),
            _ => {
                self.calls.clear();
                bail!("sampling ended inside an incomplete response");
            }
        }
        self.state = State::Finished;
        Ok(std::mem::take(&mut self.calls))
    }
}
fn valid_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 128
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"_.-".contains(&b))
}
fn function_header(text: &str) -> Result<(String, &str)> {
    let text = text
        .trim()
        .strip_prefix("<function=")
        .context("expected <function=name>")?;
    let (name, body) = text.split_once('>').context("incomplete function header")?;
    ensure!(valid_name(name), "invalid tool name");
    Ok((name.to_owned(), body))
}
fn arguments(mut body: &str, schema: Option<&Value>) -> Result<Value> {
    let mut args = Map::new();
    loop {
        body = body.trim_start();
        if let Some(rest) = body.strip_prefix("</function>") {
            ensure!(rest.trim().is_empty(), "unexpected content after function");
            break;
        }
        let rest = body
            .strip_prefix("<parameter=")
            .context("expected parameter or closing function tag")?;
        let (name, rest) = rest
            .split_once('>')
            .context("incomplete parameter header")?;
        ensure!(
            valid_name(name) && !args.contains_key(name),
            "invalid or duplicate parameter name"
        );
        let (raw, rest) = rest
            .split_once("</parameter>")
            .context("unclosed parameter")?;
        // Strip the template's delimiter newlines, not whitespace in a command.
        let raw = raw.strip_prefix('\n').unwrap_or(raw);
        let raw = raw.strip_suffix('\n').unwrap_or(raw);
        let property = schema.and_then(|s| s["properties"].get(name));
        ensure!(
            schema.is_none_or(|s| s["additionalProperties"] != false) || property.is_some(),
            "unknown parameter {name}"
        );
        let value = match property
            .and_then(|p| p["type"].as_str())
            .unwrap_or("string")
        {
            "string" => Value::String(raw.into()),
            kind => {
                let v: Value = serde_json::from_str(raw)
                    .with_context(|| format!("invalid {kind} parameter {name}"))?;
                ensure!(
                    match kind {
                        "integer" => v.is_i64() || v.is_u64(),
                        "number" => v.is_number(),
                        "boolean" => v.is_boolean(),
                        "array" => v.is_array(),
                        "object" => v.is_object(),
                        "null" => v.is_null(),
                        _ => false,
                    },
                    "wrong type for parameter {name}"
                );
                v
            }
        };
        args.insert(name.into(), value);
        body = rest;
    }
    if let Some(required) = schema.and_then(|s| s["required"].as_array()) {
        for name in required {
            ensure!(
                name.as_str().is_some_and(|n| args.contains_key(n)),
                "missing required parameter {name}"
            );
        }
    }
    Ok(Value::Object(args))
}
