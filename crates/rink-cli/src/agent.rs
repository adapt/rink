//! Tool rounds operate on new suffixes only; the session owns the exact KV prefix.
use anyhow::{Result, ensure};
use rink_inkling::chat::{Codec, Invocation};
use rink_tools::{Registry, ToolResult};
use std::sync::atomic::{AtomicBool, Ordering};

const OMITTED: &str = "[Tool output omitted: context limit.]";
const TRUNCATED: &str = "\n[Tool output truncated to fit remaining context.]";

pub enum ToolEvent<'a> {
    Starting(&'a Invocation),
    Finished(&'a str, &'a ToolResult),
}

pub struct ToolLoop {
    pub registry: Registry,
    rounds: usize,
    max_rounds: usize,
}

pub struct CompletedRound {
    pub tokens: Vec<u32>,
    pub pause: Option<&'static str>,
}

impl ToolLoop {
    pub fn new(registry: Registry, max_rounds: usize) -> Self {
        Self {
            registry,
            rounds: 0,
            max_rounds,
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
        // Check before any command can change the host. Reserve enough space to
        // record one result for every call, even if output must be discarded.
        let minimum = calls
            .iter()
            .map(|call| codec.tool_result(&call.name, OMITTED))
            .collect::<Result<Vec<_>>>()?
            .iter()
            .map(Vec::len)
            .sum::<usize>();
        ensure!(
            minimum <= token_budget,
            "context is too full to record tool results; no commands were run"
        );
        let limited = self.rounds >= self.max_rounds;
        self.rounds += usize::from(!limited);
        let mut replies = Vec::new();
        let mut interrupted = false;
        for call in calls {
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
            replies.push((call.name, result.content));
        }
        let tokens = fit_results(codec, &mut replies, token_budget)?;
        Ok(CompletedRound {
            tokens,
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

fn fit_results(codec: &Codec, replies: &mut [(String, String)], budget: usize) -> Result<Vec<u32>> {
    loop {
        let parts = replies
            .iter()
            .map(|(name, text)| codec.tool_result(name, text))
            .collect::<Result<Vec<_>>>()?;
        if parts.iter().map(Vec::len).sum::<usize>() <= budget {
            return Ok(parts.into_iter().flatten().collect());
        }
        let (_, text) = replies
            .iter_mut()
            .filter(|(_, text)| text != OMITTED)
            .max_by_key(|(_, text)| text.len())
            .ok_or_else(|| anyhow::anyhow!("tool results cannot fit the context"))?;
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
    }
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
                content: self.content.clone(),
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
        (ToolLoop::new(registry, rounds), count)
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
        session.start(&reply.tokens, 128).unwrap();
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
            [prompt.clone(), first, reply.tokens.clone(), second].concat()
        );
        assert_eq!(*inputs.borrow(), [prompt, reply.tokens]);
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
        assert!(reply.tokens.len() <= 512);
        assert!(
            codec
                .tokenizer()
                .decode(&reply.tokens, false)
                .unwrap()
                .contains("truncated")
        );
        assert!(
            !reply
                .tokens
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
        let decoded = codec.tokenizer().decode(&reply.tokens, false).unwrap();
        assert!(decoded.contains("bad JSON") && decoded.contains("not enabled"));
        assert_eq!(count.load(Ordering::Relaxed), 1);
    }
}
