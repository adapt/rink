//! Inkling 2 Mini's native checkpoint, image encoder, looped decoder, and accelerator kernels.
pub mod checkpoint;
pub mod config;
pub mod image;
pub mod kernels;
pub mod model;
mod progress;
pub mod vision;
pub use config::Config;
pub use image::Image;
pub use model::{Inkling, KvCachePrecision};
pub use progress::PrefillProgress;

/// Single user-turn subset of the released chat template (no generation suffix).
pub fn chat_prompt(prompt: &str, effort: f32) -> rink_core::Result<String> {
    rink_core::ensure!(
        effort.is_finite() && (0.0..=0.99).contains(&effort),
        "reasoning effort must be in 0..=0.99"
    );
    Ok(format!(
        "<|message_system|><|content_text|>Thinking effort level: {effort}<|end_message|>{}",
        chat_user_message(prompt)
    ))
}

/// Append a user turn after a complete assistant reply, including its EOS token.
/// The thinking-effort system message is only needed at the start of a conversation.
pub fn chat_user_message(prompt: &str) -> String {
    format!("<|message_user|><|content_text|>{prompt}<|end_message|>")
}
