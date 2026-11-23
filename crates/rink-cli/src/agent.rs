//! Tool rounds append new suffixes; the session owns the exact KV prefix.
use anyhow::{Result, ensure};
use rink_inkling::{
    Image,
    chat::{Codec, Invocation},
};
use rink_tools::{ContentPart, ImageFile, Registry, ToolResult};
use std::{
    io::{BufReader, Seek},
    sync::atomic::{AtomicBool, Ordering},
};

const OMITTED: &str = "[Tool output omitted: context limit.]";
const TRUNCATED: &str = "\n[Tool output truncated to fit remaining context.]";
const IMAGE_FULL: &str = "[Image omitted: context limit.]";
const IMAGE_INVALID: &str = "[Image could not be decoded; the file may be corrupt or unsupported.]";

pub enum ToolEvent<'a> {
    Starting(&'a Invocation),
    Finished(&'a str, &'a ToolResult),
    Notice(&'a str),
}

pub struct ToolLoop {
    registry: Registry,
    rounds: usize,
    max_rounds: usize,
    images_supported: bool,
}

struct Part {
    tokens: Vec<u32>,
    // Text is retained only while fitting this result to the context budget.
    text: Option<String>,
    image: Option<(ImageFile, Vec<u32>)>,
}
impl Part {
    fn text(codec: &Codec, name: &str, text: String) -> Result<Self> {
        Ok(Self {
            tokens: codec.tool_result(name, &text)?,
            text: Some(text),
            image: None,
        })
    }
    fn reservation(&self) -> usize {
        self.tokens
            .len()
            .max(self.image.as_ref().map_or(0, |(_, error)| error.len()))
    }
}

pub struct CompletedRound {
    parts: Vec<Part>,
    pub pause: Option<&'static str>,
}
impl CompletedRound {
    pub fn token_budget(&self) -> usize {
        self.parts.iter().map(Part::reservation).sum()
    }

    /// Consume snapshots in order. Only one decoded image is live at a time.
    /// A corrupt image becomes a text error, with space reserved before prefill.
    pub fn prefill(
        self,
        mut append: impl FnMut(&[u32], Option<&Image>) -> Result<u32>,
    ) -> Result<u32> {
        let mut next = None;
        for part in self.parts {
            if let Some((mut source, fallback)) = part.image {
                let image = (|| {
                    source.file.rewind()?;
                    Image::read(BufReader::new(&mut source.file))
                })();
                match image {
                    Ok(image) => next = Some(append(&part.tokens, Some(&image))?),
                    Err(error) => {
                        eprintln!("Cannot decode image {:?}: {error:#}", source.path);
                        next = Some(append(&fallback, None)?);
                    }
                }
            } else {
                next = Some(append(&part.tokens, None)?);
            }
        }
        next.ok_or_else(|| anyhow::anyhow!("empty tool result"))
    }

    #[cfg(test)]
    fn tokens(&self) -> Vec<u32> {
        self.parts
            .iter()
            .flat_map(|p| p.tokens.iter().copied())
            .collect()
    }
}

impl ToolLoop {
    pub fn new(registry: Registry, max_rounds: usize, images_supported: bool) -> Self {
        Self {
            registry,
            rounds: 0,
            max_rounds,
            images_supported,
        }
    }

    pub fn reset(&mut self) {
        self.rounds = 0;
    }

    pub fn run(
        &mut self,
        codec: &Codec,
        calls: Vec<Invocation>,
        token_budget: usize,
        cancelled: &AtomicBool,
        mut observe: impl FnMut(ToolEvent<'_>) -> Result<()>,
    ) -> Result<CompletedRound> {
        ensure!(!calls.is_empty(), "empty tool round");
        let minima = calls
            .iter()
            .map(|call| Ok(codec.tool_result(&call.name, OMITTED)?.len()))
            .collect::<Result<Vec<_>>>()?;
        let mut minimum: usize = minima.iter().sum();
        ensure!(
            minimum <= token_budget,
            "context is too full to record tool results; no commands were run"
        );
        let limited = self.rounds >= self.max_rounds;
        self.rounds += usize::from(!limited);
        let mut parts = Vec::new();
        let mut remaining = token_budget;
        let mut interrupted = false;
        for (call, reserved) in calls.into_iter().zip(minima) {
            minimum -= reserved;
            observe(ToolEvent::Starting(&call))?;
            let result = if limited {
                ToolResult::error(
                    "Tool round limit reached. Give your answer using the results already available; do not request more tools.",
                )
            } else if cancelled.load(Ordering::Relaxed) || interrupted {
                interrupted = true;
                ToolResult::error("Cancelled by the user; command was not started.")
            } else {
                match &call.arguments {
                    Ok(args) => self.registry.execute(&call.name, args, cancelled),
                    Err(error) => ToolResult::error(error),
                }
            };
            interrupted |= result.cancelled;
            observe(ToolEvent::Finished(&call.name, &result))?;
            let (mut reply, notice) = fit_result(
                codec,
                &call.name,
                result.content,
                remaining - minimum,
                self.images_supported,
            )?;
            if let Some(notice) = notice {
                observe(ToolEvent::Notice(notice))?;
            }
            remaining -= reply.iter().map(Part::reservation).sum::<usize>();
            // Keep text-only rounds as one prefill, as before. Image parts keep
            // their own boundaries so each can reuse the single-image encoder.
            for part in reply.drain(..) {
                if part.image.is_none() && parts.last().is_some_and(|p: &Part| p.image.is_none()) {
                    parts.last_mut().unwrap().tokens.extend(part.tokens);
                } else {
                    parts.push(part);
                }
            }
        }
        Ok(CompletedRound {
            parts,
            pause: if interrupted {
                Some("Tool execution cancelled. Use /continue to let the model read the result.")
            } else if limited {
                Some(
                    "Tool round limit reached. Use /continue to let the model answer without further execution.",
                )
            } else {
                None
            },
        })
    }
}

fn fit_result(
    codec: &Codec,
    name: &str,
    content: Vec<ContentPart>,
    budget: usize,
    images_supported: bool,
) -> Result<(Vec<Part>, Option<&'static str>)> {
    let mut parts = Vec::new();
    for content in content {
        let part = match content {
            ContentPart::Text(text) => Part::text(codec, name, text)?,
            ContentPart::Image(mut image) => {
                let tokens = (|| {
                    ensure!(
                        images_supported,
                        "this checkpoint does not support image input"
                    );
                    image.file.rewind()?;
                    Image::token_count_from_reader(BufReader::new(&mut image.file))
                })();
                match tokens {
                    Ok(count) => Part {
                        tokens: codec.tool_image(name, count)?,
                        text: None,
                        image: Some((image, codec.tool_result(name, IMAGE_INVALID)?)),
                    },
                    Err(error) => Part::text(
                        codec,
                        name,
                        format!("Cannot read image {:?}: {error:#}", image.path),
                    )?,
                }
            }
        };
        parts.push(part);
    }
    if parts.is_empty() {
        parts.push(Part::text(codec, name, String::new())?);
    }
    let mut notice = None;
    while parts.iter().map(Part::reservation).sum::<usize>() > budget {
        if let Some(part) = parts
            .iter_mut()
            .filter(|p| p.text.as_deref().is_some_and(|t| t != OMITTED))
            .max_by_key(|p| p.text.as_ref().unwrap().len())
        {
            let text = part.text.as_mut().unwrap();
            if text.len() <= TRUNCATED.len() * 2 {
                *text = OMITTED.into();
            } else {
                let body = text.strip_suffix(TRUNCATED).unwrap_or(text);
                let mut end = body.len() / 2;
                while !body.is_char_boundary(end) {
                    end -= 1;
                }
                text.truncate(end);
                text.push_str(TRUNCATED);
            }
            part.tokens = codec.tool_result(name, text)?;
        } else if parts.iter().any(|p| p.image.is_some()) {
            // Never truncate an image's placeholders, or insert only part of a
            // multi-image result. Drop snapshots and return an explicit error.
            notice = Some(IMAGE_FULL);
            parts = vec![Part::text(codec, name, IMAGE_FULL.into())?];
        } else {
            // Several content parts can need more framing than one result.
            parts = vec![Part::text(codec, name, OMITTED.into())?];
            ensure!(
                parts[0].tokens.len() <= budget,
                "tool result cannot fit the context"
            );
        }
    }
    for part in &mut parts {
        part.text = None;
    }
    Ok((parts, notice))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{chat_fixture, session::Session};
    use rink_core::AutoregressiveModel;
    use rink_tools::Tool;
    use serde_json::{Value, json};
    use std::{
        cell::RefCell,
        collections::VecDeque,
        rc::Rc,
        sync::{Arc, atomic::AtomicUsize},
    };

    struct Probe {
        count: Arc<AtomicUsize>,
        content: String,
    }
    impl Tool for Probe {
        fn name(&self) -> &'static str {
            "probe"
        }
        fn definition(&self) -> Value {
            json!({"name":"probe","parameters":{"type":"object"}})
        }
        fn execute(&self, _: &Value, _: &AtomicBool) -> Result<ToolResult> {
            self.count.fetch_add(1, Ordering::Relaxed);
            Ok(ToolResult {
                content: vec![ContentPart::Text(self.content.clone())],
                cancelled: false,
            })
        }
    }
    fn tools(content: &str, rounds: usize) -> (ToolLoop, Arc<AtomicUsize>) {
        let count = Arc::new(AtomicUsize::new(0));
        let mut registry = Registry::default();
        registry
            .register(Probe {
                count: count.clone(),
                content: content.into(),
            })
            .unwrap();
        (ToolLoop::new(registry, rounds, true), count)
    }
    fn call() -> Vec<Invocation> {
        vec![Invocation {
            name: "probe".into(),
            arguments: Ok(json!({})),
        }]
    }

    struct Script {
        seen: Rc<RefCell<Vec<u32>>>,
        inputs: Rc<RefCell<Vec<Vec<u32>>>>,
        turns: VecDeque<VecDeque<u32>>,
        current: VecDeque<u32>,
    }
    impl AutoregressiveModel for Script {
        fn context_capacity(&self) -> usize {
            16384
        }
        fn position(&self) -> usize {
            self.seen.borrow().len()
        }
        fn reset(&mut self) -> Result<()> {
            self.seen.borrow_mut().clear();
            Ok(())
        }
        fn step(&mut self, _: u32) -> Result<Vec<f32>> {
            unreachable!("use greedy path")
        }
        fn step_greedy(&mut self, token: u32) -> Result<u32> {
            self.seen.borrow_mut().push(token);
            Ok(self.current.pop_front().unwrap_or(0))
        }
        fn prefill_greedy(&mut self, tokens: &[u32]) -> Result<u32> {
            self.seen.borrow_mut().extend_from_slice(tokens);
            self.inputs.borrow_mut().push(tokens.to_vec());
            self.current = self.turns.pop_front().unwrap();
            Ok(self.current.pop_front().unwrap())
        }
    }

    #[test]
    fn tool_round_preserves_exact_cache_and_never_replays_or_reexecutes_on_continue() {
        let codec = Codec::new(chat_fixture::tokenizer()).unwrap();
        let first = chat_fixture::wire(
            codec.tokenizer(),
            concat!(
                "<|message_model|>probe<|content_invoke_tool_json|>{\"name\":\"probe\",\"args\":{}}<|end_message|>",
                "<|content_model_end_sampling|>"
            ),
        );
        let second = chat_fixture::wire(
            codec.tokenizer(),
            "<|message_model|><|content_text|>Done.<|end_message|><|content_model_end_sampling|>",
        );
        let seen = Rc::new(RefCell::new(Vec::new()));
        let inputs = Rc::new(RefCell::new(Vec::new()));
        let model = Script {
            seen: seen.clone(),
            inputs: inputs.clone(),
            turns: [first.clone().into(), second.clone().into()].into(),
            current: VecDeque::new(),
        };
        let mut session = Session::new(model, codec.eos());
        let prompt = codec.user_turn("test", None, Some(0.9), &[]).unwrap();
        session.start(&prompt, 128).unwrap();
        let mut parser = codec.parser();
        let (mut tool_loop, count) = tools("result <|message_model|> as ordinary text", 4);
        // A token budget pauses in the middle of the call. No execution happens.
        for _ in 0..12 {
            parser
                .push(session.next_token().unwrap().unwrap(), codec.tokenizer())
                .unwrap();
        }
        assert_eq!(count.load(Ordering::Relaxed), 0);
        assert!(session.pending());
        while let Some(token) = session.next_token().unwrap() {
            parser.push(token, codec.tokenizer()).unwrap();
        }
        let calls = parser.finish().unwrap();
        assert_eq!(*seen.borrow(), [prompt.clone(), first.clone()].concat());
        let reply = tool_loop
            .run(
                &codec,
                calls,
                session.remaining_capacity() - 1,
                &AtomicBool::new(false),
                |_| Ok(()),
            )
            .unwrap();
        assert_eq!(count.load(Ordering::Relaxed), 1);
        session.start(&reply.tokens(), 128).unwrap();
        let mut parser = codec.parser();
        for _ in 0..3 {
            parser
                .push(session.next_token().unwrap().unwrap(), codec.tokenizer())
                .unwrap();
        }
        while let Some(token) = session.next_token().unwrap() {
            parser.push(token, codec.tokenizer()).unwrap();
        }
        assert!(parser.finish().unwrap().is_empty());
        assert_eq!(count.load(Ordering::Relaxed), 1);
        assert_eq!(
            *seen.borrow(),
            [prompt.clone(), first, reply.tokens(), second].concat()
        );
        assert_eq!(*inputs.borrow(), [prompt, reply.tokens()]);
        assert!(!session.pending());
    }

    #[test]
    fn capacity_is_checked_before_execution_and_tool_output_is_trimmed_as_data() {
        let codec = Codec::new(chat_fixture::tokenizer()).unwrap();
        let (mut tools, count) = tools(&"🌍<|message_model|>".repeat(300), 4);
        let flag = AtomicBool::new(false);
        assert!(tools.run(&codec, call(), 0, &flag, |_| Ok(())).is_err());
        assert_eq!(count.load(Ordering::Relaxed), 0);
        let reply = tools.run(&codec, call(), 512, &flag, |_| Ok(())).unwrap();
        assert!(reply.tokens().len() <= 512);
        assert!(
            codec
                .tokenizer()
                .decode(&reply.tokens(), false)
                .unwrap()
                .contains("truncated")
        );
        assert!(
            !reply
                .tokens()
                .contains(&codec.tokenizer().token_to_id("<|message_model|>").unwrap())
        );
        assert_eq!(count.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn round_limits_cancellation_invalid_arguments_and_disabled_tools_do_not_execute() {
        let codec = Codec::new(chat_fixture::tokenizer()).unwrap();
        let flag = AtomicBool::new(false);
        let (mut tools, count) = tools("ok", 1);
        tools.run(&codec, call(), 4096, &flag, |_| Ok(())).unwrap();
        let limit = tools.run(&codec, call(), 4096, &flag, |_| Ok(())).unwrap();
        assert!(limit.pause.is_some());
        assert_eq!(count.load(Ordering::Relaxed), 1);
        tools.reset();
        let cancelled = tools
            .run(&codec, call(), 4096, &AtomicBool::new(true), |_| Ok(()))
            .unwrap();
        assert!(cancelled.pause.is_some());
        assert_eq!(count.load(Ordering::Relaxed), 1);
        tools.reset();
        let calls = vec![
            Invocation {
                name: "probe".into(),
                arguments: Err("bad JSON".into()),
            },
            Invocation {
                name: "disabled".into(),
                arguments: Ok(json!({})),
            },
        ];
        let reply = tools.run(&codec, calls, 4096, &flag, |_| Ok(())).unwrap();
        let decoded = codec.tokenizer().decode(&reply.tokens(), false).unwrap();
        assert!(decoded.contains("bad JSON") && decoded.contains("not enabled"));
        assert_eq!(count.load(Ordering::Relaxed), 1);
    }

    fn reader() -> rink_tools::Read {
        rink_tools::Read::new(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../models/inkling/tests/fixtures/vision"),
            std::time::Duration::from_secs(2),
            4096,
        )
        .unwrap()
    }

    fn image_content(name: &str) -> Vec<ContentPart> {
        reader()
            .execute(&json!({"path":name}), &AtomicBool::new(false))
            .unwrap()
            .content
    }

    #[test]
    fn multiple_image_results_preserve_order_cache_and_continuation_without_reinsertion() {
        let codec = Codec::new(chat_fixture::tokenizer()).unwrap();
        let mut registry = Registry::default();
        registry.register(reader()).unwrap();
        let mut tools = ToolLoop::new(registry, 8, true);
        let calls = ["input.png", "oriented.jpg"]
            .map(|path| Invocation {
                name: "read".into(),
                arguments: Ok(json!({"path":path})),
            })
            .into();
        let reply = tools
            .run(&codec, calls, 4096, &AtomicBool::new(false), |_| Ok(()))
            .unwrap();
        let reserved = reply.token_budget();
        let tokens = reply.tokens();
        let answer = chat_fixture::wire(
            codec.tokenizer(),
            "<|message_model|><|content_text|>Both images read.<|end_message|><|content_model_end_sampling|>",
        );
        // Each prefill produces a next-token prediction. Only the last part's
        // prediction may become a sampled token in the conversation.
        let mut turns = VecDeque::from([VecDeque::from([codec.eos()])]);
        for _ in 1..reply.parts.len() {
            turns.push_back(VecDeque::from([999]));
        }
        turns.push_back(answer.clone().into());
        let seen = Rc::new(RefCell::new(Vec::new()));
        let inputs = Rc::new(RefCell::new(Vec::new()));
        let mut session = Session::new(
            Script {
                seen: seen.clone(),
                inputs,
                turns,
                current: VecDeque::new(),
            },
            codec.eos(),
        );
        session.start(&[42], 1).unwrap();
        assert_eq!(session.next_token().unwrap(), None);
        let mut images = Vec::new();
        session
            .start_append(reserved, 1, |model| {
                reply.prefill(|part, image| {
                    if let Some(image) = image {
                        assert_eq!(
                            part.iter().filter(|&&t| t == codec.image_token()).count(),
                            image.token_count()
                        );
                        images.push((image.width(), image.height()));
                    } else {
                        assert!(!part.contains(&codec.image_token()));
                    }
                    model.prefill_greedy(part)
                })
            })
            .unwrap();
        assert_eq!(images, [(241, 81), (7, 11)]);
        assert_eq!(
            *seen.borrow(),
            [vec![42, codec.eos()], tokens.clone()].concat()
        );
        // Pause a reply and then continue. There is no image data to reinsert.
        let mut parser = codec.parser();
        for _ in 0..3 {
            parser
                .push(session.next_token().unwrap().unwrap(), codec.tokenizer())
                .unwrap();
        }
        assert!(session.pending());
        while let Some(token) = session.next_token().unwrap() {
            parser.push(token, codec.tokenizer()).unwrap();
        }
        assert!(parser.finish().unwrap().is_empty());
        assert_eq!(
            *seen.borrow(),
            [vec![42, codec.eos()], tokens, answer].concat()
        );
        session.reset().unwrap();
        assert_eq!(session.position(), 0);
    }

    #[test]
    fn image_context_rejection_and_decode_failures_append_only_valid_text_results() {
        use std::io::{SeekFrom, Write};
        let codec = Codec::new(chat_fixture::tokenizer()).unwrap();
        let minimum = codec.tool_result("read", OMITTED).unwrap().len();
        let (parts, notice) =
            fit_result(&codec, "read", image_content("input.png"), minimum, true).unwrap();
        assert_eq!(notice, Some(IMAGE_FULL));
        assert!(
            parts
                .iter()
                .all(|p| p.image.is_none() && !p.tokens.contains(&codec.image_token()))
        );
        let text = codec.tokenizer().decode(&parts[0].tokens, false).unwrap();
        assert!(text.contains("Image omitted"));

        let (parts, _) =
            fit_result(&codec, "read", image_content("input.png"), 4096, false).unwrap();
        assert!(parts.iter().all(|p| p.image.is_none()));
        assert!(
            codec
                .tokenizer()
                .decode(
                    &parts
                        .iter()
                        .flat_map(|p| p.tokens.iter().copied())
                        .collect::<Vec<_>>(),
                    false
                )
                .unwrap()
                .contains("does not support image")
        );

        let mut content = image_content("input.png");
        let ContentPart::Image(source) = &mut content[1] else {
            unreachable!()
        };
        // Destroy the compressed pixel stream, leaving a valid PNG header.
        source.file.seek(SeekFrom::Start(41)).unwrap();
        source.file.write_all(&[0; 16]).unwrap();
        let (parts, _) = fit_result(&codec, "read", content, 4096, true).unwrap();
        assert!(
            parts.iter().any(|p| p.image.is_some()),
            "header must pass preflight"
        );
        let reply = CompletedRound { parts, pause: None };
        let reserved = reply.token_budget();
        let mut actual = Vec::new();
        reply
            .prefill(|tokens, image| {
                assert!(image.is_none());
                assert!(!tokens.contains(&codec.image_token()));
                actual.extend_from_slice(tokens);
                Ok(7)
            })
            .unwrap();
        assert!(actual.len() <= reserved);
        assert!(
            codec
                .tokenizer()
                .decode(&actual, false)
                .unwrap()
                .contains(IMAGE_INVALID)
        );
    }
}
