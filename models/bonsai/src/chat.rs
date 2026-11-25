//! Text framing from the released chat template. User strings cannot inject
//! protocol tokens; framing and reasoning delimiters are inserted as token IDs.
use anyhow::{Context, Result, ensure};
use std::collections::HashSet;
use tokenizers::Tokenizer;

pub struct Codec {
    tokenizer: Tokenizer,
    start: u32,
    end: u32,
    end_text: u32,
    think: u32,
    end_think: u32,
    special: HashSet<u32>,
}
impl Codec {
    pub fn new(mut tokenizer: Tokenizer) -> Result<Self> {
        let id = |name: &str| {
            tokenizer
                .token_to_id(name)
                .with_context(|| format!("Bonsai tokenizer is missing {name}"))
        };
        let (start, end, end_text, think, end_think) = (
            id("<|im_start|>")?,
            id("<|im_end|>")?,
            id("<|endoftext|>")?,
            id("<think>")?,
            id("</think>")?,
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
    /// Effort 0 disables thinking; positive effort selects low (<1/3), medium
    /// (<2/3), or xhigh. Unlike Inkling, this model has categorical effort levels.
    pub fn user_turn(&self, prompt: &str, first: bool, effort: f32) -> Result<Vec<u32>> {
        ensure!(
            effort.is_finite() && (0.0..=0.99).contains(&effort),
            "reasoning effort must be in 0..=0.99"
        );
        let mut out = Vec::new();
        if first {
            let instructions = match effort {
                e if e >= 2.0 / 3.0 => Some(
                    "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.",
                ),
                e if e > 0.0 && e < 1.0 / 3.0 => Some(
                    "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.",
                ),
                _ => None,
            };
            if let Some(text) = instructions {
                self.message(&mut out, "system", text)?;
            }
        } else {
            self.text(&mut out, "\n")?;
        }
        self.message(&mut out, "user", prompt)?;
        out.push(self.start);
        self.text(&mut out, "assistant\n")?;
        out.push(self.think);
        if effort == 0.0 {
            self.text(&mut out, "\n\n")?;
            out.push(self.end_think);
            self.text(&mut out, "\n\n")?;
        } else {
            self.text(&mut out, "\n")?;
        }
        Ok(out)
    }
}
