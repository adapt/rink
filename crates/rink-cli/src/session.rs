//! Incremental conversation state. Every generated token, including EOS, enters
//! the cache exactly once so a new user turn can follow without replaying history.
use anyhow::{Context, Result, ensure};
use rink_core::AutoregressiveModel;

enum State {
    Ready,
    Generating(u32),
    Failed,
}

pub struct Session<M> {
    model: M,
    eos: u32,
    state: State,
}

impl<M: AutoregressiveModel> Session<M> {
    pub fn new(model: M, eos: u32) -> Self {
        Self {
            model,
            eos,
            state: State::Ready,
        }
    }

    pub fn position(&self) -> usize {
        self.model.position()
    }

    pub fn remaining_capacity(&self) -> usize {
        self.model
            .context_capacity()
            .saturating_sub(self.position())
    }

    pub fn pending(&self) -> bool {
        matches!(self.state, State::Generating(_))
    }

    /// Reject oversized requests before changing any conversation state.
    pub fn check_prompt(&self, tokens: &[u32], max_tokens: usize) -> Result<()> {
        self.check_append(tokens.len(), max_tokens)
    }

    fn check_append(&self, token_count: usize, max_tokens: usize) -> Result<()> {
        ensure!(
            matches!(self.state, State::Ready),
            "use /continue or /reset before a new message"
        );
        ensure!(token_count > 0, "message must encode to at least one token");
        ensure!(max_tokens > 0, "reply budget must be positive");
        let total = self
            .position()
            .checked_add(token_count)
            .and_then(|n| n.checked_add(max_tokens))
            .context("token budget overflow")?;
        ensure!(
            total <= self.model.context_capacity(),
            "conversation plus message and reply budget needs {total} tokens, but --context is {}; use /reset or restart with a larger --context",
            self.model.context_capacity()
        );
        Ok(())
    }

    #[cfg(test)]
    pub fn start(&mut self, tokens: &[u32], max_tokens: usize) -> Result<()> {
        self.start_with(tokens, max_tokens, |model, tokens| {
            model.prefill_greedy(tokens)
        })
    }

    /// A model-specific prefill (for example, image embeddings) uses the same
    /// budget, continuation and failure rules as a text turn.
    pub fn start_with(
        &mut self,
        tokens: &[u32],
        max_tokens: usize,
        prefill: impl FnOnce(&mut M, &[u32]) -> Result<u32>,
    ) -> Result<()> {
        self.start_append(tokens.len(), max_tokens, |model| prefill(model, tokens))
    }

    /// Append multiple text/image parts as one round. Reserve the worst-case
    /// token count first; generation resumes only after all parts are cached.
    pub fn start_append(
        &mut self,
        token_count: usize,
        max_tokens: usize,
        prefill: impl FnOnce(&mut M) -> Result<u32>,
    ) -> Result<()> {
        self.check_append(token_count, max_tokens)?;
        let before = self.position();
        self.state = State::Failed;
        let next = prefill(&mut self.model)?;
        ensure!(
            self.position() > before && self.position() - before <= token_count,
            "prefill did not respect its reserved token count"
        );
        self.state = State::Generating(next);
        Ok(())
    }

    /// A continuation can use the last few positions even if a full chunk won't fit.
    pub fn reply_budget(&self, requested: usize) -> Result<usize> {
        ensure!(self.pending(), "there is no unfinished reply to continue");
        let remaining = self
            .model
            .context_capacity()
            .saturating_sub(self.position());
        ensure!(
            remaining > 0,
            "context is full; use /reset or restart with a larger --context"
        );
        ensure!(requested > 0, "reply budget must be positive");
        Ok(requested.min(remaining))
    }

    /// `None` means EOS. A token limit leaves logits intact for /continue.
    pub fn next_token(&mut self) -> Result<Option<u32>> {
        let State::Generating(token) = self.state else {
            anyhow::bail!("no active reply; a failed session must be reset");
        };
        self.state = State::Failed;
        let next = self.model.step_greedy(token)?;
        if token == self.eos {
            self.state = State::Ready;
            Ok(None)
        } else {
            self.state = State::Generating(next);
            Ok(Some(token))
        }
    }

    pub fn reset(&mut self) -> Result<()> {
        self.state = State::Failed;
        self.model.reset()?;
        self.state = State::Ready;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Model {
        tokens: Vec<u32>,
        capacity: usize,
        fail_at: Option<usize>,
    }

    impl AutoregressiveModel for Model {
        fn context_capacity(&self) -> usize {
            self.capacity
        }
        fn position(&self) -> usize {
            self.tokens.len()
        }
        fn reset(&mut self) -> Result<()> {
            self.tokens.clear();
            self.fail_at = None;
            Ok(())
        }
        fn step(&mut self, token: u32) -> Result<Vec<f32>> {
            ensure!(self.fail_at != Some(self.position()), "injected failure");
            ensure!(self.position() < self.capacity, "cache full");
            self.tokens.push(token);
            // Prompt token 1 predicts 2, then 3, then EOS 0.
            let mut logits = vec![-10.; 4];
            logits[((token + 1) % 4) as usize] = 10.;
            Ok(logits)
        }
    }

    fn session(capacity: usize) -> Session<Model> {
        Session::new(
            Model {
                tokens: vec![],
                capacity,
                fail_at: None,
            },
            0,
        )
    }

    #[test]
    fn keeps_exact_history_across_turns_continuation_and_reset() {
        let mut s = session(16);
        s.start(&[1], 3).unwrap();
        assert_eq!(s.next_token().unwrap(), Some(2));
        assert!(s.pending());
        assert!(s.start(&[1], 3).is_err());
        assert_eq!(s.reply_budget(3).unwrap(), 3);
        assert_eq!(s.next_token().unwrap(), Some(3));
        assert_eq!(s.next_token().unwrap(), None);
        assert!(!s.pending());
        assert_eq!(s.model.tokens, [1, 2, 3, 0]);

        s.start(&[1], 3).unwrap();
        assert_eq!(s.next_token().unwrap(), Some(2));
        assert_eq!(s.next_token().unwrap(), Some(3));
        assert_eq!(s.next_token().unwrap(), None);
        assert_eq!(s.model.tokens, [1, 2, 3, 0, 1, 2, 3, 0]);

        s.reset().unwrap();
        assert_eq!(s.position(), 0);
        assert!(!s.pending());
        s.start(&[1], 3).unwrap();
        assert_eq!(s.next_token().unwrap(), Some(2));
    }

    #[test]
    fn bounds_include_eos_and_rejections_preserve_the_cache() {
        let mut s = session(4);
        assert!(s.start(&[1, 1], 3).is_err());
        assert!(s.start(&[], 3).is_err());
        assert!(s.start(&[1], usize::MAX).is_err());
        assert_eq!(s.position(), 0);
        s.start(&[1], 2).unwrap();
        s.next_token().unwrap();
        s.next_token().unwrap();
        assert_eq!(s.reply_budget(100).unwrap(), 1);
        assert_eq!(s.next_token().unwrap(), None);
        assert_eq!(s.position(), 4);
        assert!(s.start(&[1], 1).is_err());
        assert_eq!(s.model.tokens, [1, 2, 3, 0]);
    }

    #[test]
    fn failures_require_reset_and_reset_discards_a_partial_reply() {
        let mut s = session(8);
        s.start(&[1], 3).unwrap();
        s.model.fail_at = Some(1);
        assert!(s.next_token().is_err());
        assert!(s.next_token().is_err());
        assert!(s.start(&[1], 3).is_err());
        s.reset().unwrap();
        s.start(&[1], 3).unwrap();
        s.next_token().unwrap();
        s.reset().unwrap();
        s.start(&[1], 3).unwrap();
        assert_eq!(s.model.tokens, [1]);

        s.reset().unwrap();
        s.model.fail_at = Some(1);
        assert!(s.start(&[1, 1], 3).is_err());
        assert!(s.start(&[1], 3).is_err());
        s.reset().unwrap();
        s.start(&[1], 3).unwrap();
    }

    #[test]
    fn custom_prefill_checks_budget_and_requires_reset_after_failure() {
        let mut s = session(8);
        assert!(
            s.start_with(&[1; 6], 3, |_, _| panic!("must reject before prefill"))
                .is_err()
        );
        s.start_with(&[1], 3, |model, tokens| model.prefill_greedy(tokens))
            .unwrap();
        assert_eq!(s.next_token().unwrap(), Some(2));
        s.reset().unwrap();
        assert!(
            s.start_with(&[1], 3, |_, _| anyhow::bail!("injected image failure"))
                .is_err()
        );
        assert!(s.start(&[1], 3).is_err());
        s.reset().unwrap();
        s.start(&[1], 3).unwrap();
    }
}
