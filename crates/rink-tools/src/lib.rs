//! Host-side tools. Model-specific message framing belongs to the model crate.
use anyhow::{Result, ensure};
use serde_json::{Value, json};
use std::{collections::BTreeMap, fs::File, path::PathBuf, sync::atomic::AtomicBool};

mod read;
pub use read::Read;

#[cfg(unix)]
mod bash;
#[cfg(unix)]
mod interrupt;
#[cfg(unix)]
pub use bash::Bash;
#[cfg(unix)]
pub use interrupt::Interrupt;

/// An owned snapshot, independent of later writes to the source file. Encoded
/// images stay on disk until the model consumes them; closing the file removes it.
pub struct ImageFile {
    pub path: PathBuf,
    pub file: File,
}

pub enum ContentPart {
    Text(String),
    Image(ImageFile),
}

pub struct ToolResult {
    pub content: Vec<ContentPart>,
    pub cancelled: bool,
}

impl ToolResult {
    pub fn text(content: impl Into<String>) -> Self {
        Self {
            content: vec![ContentPart::Text(content.into())],
            cancelled: false,
        }
    }

    pub fn as_text(&self) -> Option<&str> {
        match self.content.as_slice() {
            [ContentPart::Text(text)] => Some(text),
            _ => None,
        }
    }

    pub fn error(message: impl AsRef<str>) -> Self {
        Self::text(json!({"error": message.as_ref()}).to_string())
    }
}

pub trait Tool {
    fn name(&self) -> &'static str;
    /// A function declaration with name, description, type and JSON-schema parameters.
    fn definition(&self) -> Value;
    fn execute(&self, args: &Value, cancelled: &AtomicBool) -> Result<ToolResult>;
}

#[derive(Default)]
pub struct Registry {
    tools: BTreeMap<String, Box<dyn Tool>>,
}

impl Registry {
    pub fn register(&mut self, tool: impl Tool + 'static) -> Result<()> {
        ensure!(!self.tools.contains_key(tool.name()), "duplicate tool");
        self.tools.insert(tool.name().into(), Box::new(tool));
        Ok(())
    }

    pub fn definitions(&self) -> Vec<Value> {
        self.tools.values().map(|tool| tool.definition()).collect()
    }

    pub fn execute(&self, name: &str, args: &Value, cancelled: &AtomicBool) -> ToolResult {
        match self.tools.get(name) {
            Some(tool) => tool
                .execute(args, cancelled)
                .unwrap_or_else(|error| ToolResult::error(format!("{error:#}"))),
            None => ToolResult::error(format!("tool {name:?} is not enabled")),
        }
    }
}
