use rink_inkling::PrefillProgress;
use std::{
    io::{IsTerminal, Write},
    time::{Duration, Instant},
};

/// Status goes to stderr so redirected model output stays usable. Terminals
/// reuse one line; logs receive occasional complete lines without ANSI escapes.
pub struct StatusLine {
    terminal: bool,
    live: bool,
    last_update: Option<Instant>,
}

impl StatusLine {
    pub fn new() -> Self {
        Self {
            terminal: std::io::stderr().is_terminal(),
            live: false,
            last_update: None,
        }
    }

    pub fn update(&mut self, message: &str, force: bool) {
        if !force
            && self
                .last_update
                .is_some_and(|t| t.elapsed() < Duration::from_secs(1))
        {
            return;
        }
        let mut stderr = std::io::stderr().lock();
        if self.terminal {
            let _ = write!(stderr, "\r\x1b[2K{message}");
            self.live = true;
        } else {
            let _ = writeln!(stderr, "{message}");
        }
        let _ = stderr.flush();
        self.last_update = Some(Instant::now());
    }

    pub fn clear(&mut self) {
        if self.live {
            let mut stderr = std::io::stderr().lock();
            let _ = write!(stderr, "\r\x1b[2K");
            let _ = stderr.flush();
            self.live = false;
        }
    }

    fn finish(&mut self, message: &str) {
        self.clear();
        let _ = writeln!(std::io::stderr().lock(), "{message}");
    }
}

impl Drop for StatusLine {
    fn drop(&mut self) {
        // An error must not leave a live status line or report false success.
        self.clear();
    }
}

pub struct PrefillDisplay {
    line: StatusLine,
    started: Instant,
    prompt_tokens: usize,
}

impl PrefillDisplay {
    pub fn new() -> Self {
        Self {
            line: StatusLine::new(),
            started: Instant::now(),
            prompt_tokens: 0,
        }
    }

    pub fn observe(&mut self, event: PrefillProgress) {
        match event {
            PrefillProgress::LoadingImageEncoder => {
                self.started = Instant::now();
                self.line.update("Loading image encoder…", true);
            }
            PrefillProgress::ImageEncoderLoaded => {
                self.line.finish(&format!(
                    "Image encoder load: {:.2}s",
                    self.started.elapsed().as_secs_f64()
                ));
            }
            PrefillProgress::ImageEncoding {
                completed_layers,
                total_layers,
            } => {
                if completed_layers == 0 {
                    self.started = Instant::now();
                }
                self.line.update(
                    &format!(
                        "Encoding image: {completed_layers}/{total_layers} layers; {:.1}s",
                        self.started.elapsed().as_secs_f64()
                    ),
                    completed_layers == 0,
                );
            }
            PrefillProgress::ImageEncoded => {
                self.line.finish(&format!(
                    "Image encoding: {:.2}s",
                    self.started.elapsed().as_secs_f64()
                ));
            }
            PrefillProgress::Prompt {
                completed_tokens,
                total_tokens,
            } => {
                if completed_tokens == 0 {
                    self.started = Instant::now();
                    self.prompt_tokens = total_tokens;
                }
                let seconds = self.started.elapsed().as_secs_f64();
                self.line.update(
                    &format!(
                        "Prompt prefill: {completed_tokens}/{total_tokens} tokens; {seconds:.1}s ({:.1} tok/s)",
                        rate(completed_tokens, seconds)
                    ),
                    completed_tokens == 0,
                );
            }
            PrefillProgress::Complete => {
                let seconds = self.started.elapsed().as_secs_f64();
                self.line.finish(&format!(
                    "Prompt prefill: {0}/{0} tokens in {seconds:.2}s ({1:.1} tok/s)",
                    self.prompt_tokens,
                    rate(self.prompt_tokens, seconds)
                ));
            }
        }
    }
}

pub fn rate(tokens: usize, seconds: f64) -> f64 {
    if seconds > 0. {
        tokens as f64 / seconds
    } else {
        0.
    }
}
