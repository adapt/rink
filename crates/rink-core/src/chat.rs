//! Model-independent tool calls. Each model owns its wire format and parser.
use anyhow::{Result, bail};
use serde_json::Value;

#[derive(Debug)]
pub struct Invocation {
    pub name: String,
    /// Invalid payloads become tool errors and must never reach an executor.
    pub arguments: std::result::Result<Value, String>,
}

/// Encode result bodies and any framing shared by a complete tool round.
/// All framing is reserved before execution, including the assistant prefix.
pub trait ToolCodec {
    fn tool_result(&self, name: &str, content: &str) -> Result<Vec<u32>>;
    fn tool_image(&self, _name: &str, _image_tokens: usize) -> Result<Vec<u32>> {
        bail!("this model does not support image tool results")
    }
    fn tool_round_start(&self) -> Result<Vec<u32>> {
        Ok(Vec::new())
    }
    fn tool_round_end(&self) -> Result<Vec<u32>> {
        Ok(Vec::new())
    }
}
