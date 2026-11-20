//! Inkling 2 Mini's native checkpoint, looped text decoder, and accelerator kernels.
pub mod checkpoint;
pub mod config;
pub mod kernels;
pub mod model;
pub use config::Config;
pub use model::Inkling;

/// Single user-turn subset of the released chat template (no generation suffix).
pub fn chat_prompt(prompt: &str, effort: f32) -> rink_core::Result<String> {
    rink_core::ensure!(
        effort.is_finite() && (0.0..=0.99).contains(&effort),
        "reasoning effort must be in 0..=0.99"
    );
    Ok(format!(
        "<|message_system|><|content_text|>Thinking effort level: {effort}<|end_message|><|message_user|><|content_text|>{prompt}<|end_message|>"
    ))
}
