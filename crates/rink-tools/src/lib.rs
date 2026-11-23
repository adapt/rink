//! Host-side tools. Model-specific message framing belongs to the model crate.
use anyhow::{Result, ensure};
use serde_json::{Value, json};
use std::{collections::BTreeMap, sync::atomic::AtomicBool};

#[cfg(unix)]
mod bash;
#[cfg(unix)]
mod interrupt;
#[cfg(unix)]
pub use bash::Bash;
#[cfg(unix)]
pub use interrupt::Interrupt;

pub struct ToolResult {
    pub content: String,
    pub cancelled: bool,
}

impl ToolResult {
    pub fn error(message: impl AsRef<str>) -> Self {
        Self {
            content: json!({"error": message.as_ref()}).to_string(),
            cancelled: false,
        }
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
