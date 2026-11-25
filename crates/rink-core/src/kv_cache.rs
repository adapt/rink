use crate::{Result, bail};

/// Storage precision for keys and values; attention arithmetic stays float32.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum KvCachePrecision {
    #[default]
    F16,
    F32,
}

impl KvCachePrecision {
    pub fn bytes(self) -> usize {
        match self {
            Self::F16 => 2,
            Self::F32 => 4,
        }
    }
}

impl std::fmt::Display for KvCachePrecision {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::F16 => "f16",
            Self::F32 => "f32",
        })
    }
}

impl std::str::FromStr for KvCachePrecision {
    type Err = anyhow::Error;
    fn from_str(value: &str) -> Result<Self> {
        match value {
            "f16" => Ok(Self::F16),
            "f32" => Ok(Self::F32),
            _ => bail!("KV cache precision must be f16 or f32"),
        }
    }
}
