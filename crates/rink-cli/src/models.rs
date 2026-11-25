//! Application model registry. Detection uses checkpoint metadata, never its path.
use anyhow::{Context, Result, bail};
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Inkling,
    Bonsai,
}
impl Kind {
    pub fn detect(directory: &Path) -> Result<Self> {
        let c: serde_json::Value =
            serde_json::from_slice(&std::fs::read(directory.join("config.json"))?)?;
        match c
            .get("model_type")
            .and_then(|v| v.as_str())
            .context("config.json is missing model_type")?
        {
            "inkling2_loop" => Ok(Self::Inkling),
            "prism_hadamard_qwen35" => Ok(Self::Bonsai),
            other => bail!(
                "unsupported model_type {other:?}; supported models: Inkling 2 Mini, Bonsai 2.5"
            ),
        }
    }
}

#[cfg(target_os = "macos")]
pub mod loaded {
    use super::*;
    use rink_core::{Accelerator, AutoregressiveModel, KvCachePrecision};
    use rink_metal::Metal;

    pub enum Model {
        Inkling(Box<rink_inkling::Inkling<Metal>>),
        Bonsai(Box<rink_bonsai::Bonsai<Metal>>),
    }
    impl Model {
        pub fn load(
            device: Metal,
            directory: &Path,
            context: usize,
            kv: KvCachePrecision,
            batch: Option<usize>,
        ) -> Result<Self> {
            match Kind::detect(directory)? {
                Kind::Inkling => {
                    let mut m =
                        rink_inkling::Inkling::load_with_kv_cache(device, directory, context, kv)?;
                    if let Some(n) = batch {
                        m.set_prefill_batch_limit(n)?;
                    }
                    Ok(Self::Inkling(Box::new(m)))
                }
                Kind::Bonsai => {
                    crate::bonsai::validate(kv, batch)?;
                    Ok(Self::Bonsai(Box::new(
                        rink_bonsai::Bonsai::load_with_prefill_batch(
                            device,
                            directory,
                            context,
                            batch.unwrap_or(rink_bonsai::DEFAULT_PREFILL_BATCH),
                        )?,
                    )))
                }
            }
        }
        pub fn accelerator(&self) -> &Metal {
            match self {
                Self::Inkling(m) => m.accelerator(),
                Self::Bonsai(m) => m.accelerator(),
            }
        }
        pub fn accelerator_mut(&mut self) -> &mut Metal {
            match self {
                Self::Inkling(m) => m.accelerator_mut(),
                Self::Bonsai(m) => m.accelerator_mut(),
            }
        }
        pub fn device_name(&self) -> &str {
            self.accelerator().name()
        }
        pub fn prefill_batch_limit(&self) -> usize {
            match self {
                Self::Inkling(m) => m.prefill_batch_limit(),
                Self::Bonsai(m) => m.prefill_batch_limit(),
            }
        }
        pub fn prefill_progress(
            &mut self,
            tokens: &[u32],
            mut progress: impl FnMut(usize),
        ) -> Result<u32> {
            match self {
                Self::Inkling(m) => m.prefill_greedy_with_progress(tokens, |event| match event {
                    rink_inkling::PrefillProgress::Prompt {
                        completed_tokens, ..
                    } => progress(completed_tokens),
                    rink_inkling::PrefillProgress::Complete => progress(tokens.len()),
                    _ => {}
                }),
                Self::Bonsai(m) => m.prefill_greedy_with_progress(tokens, progress),
            }
        }
    }
    impl AutoregressiveModel for Model {
        fn context_capacity(&self) -> usize {
            match self {
                Self::Inkling(m) => m.context_capacity(),
                Self::Bonsai(m) => m.context_capacity(),
            }
        }
        fn position(&self) -> usize {
            match self {
                Self::Inkling(m) => m.position(),
                Self::Bonsai(m) => m.position(),
            }
        }
        fn reset(&mut self) -> Result<()> {
            match self {
                Self::Inkling(m) => m.reset(),
                Self::Bonsai(m) => m.reset(),
            }
        }
        fn step(&mut self, token: u32) -> Result<Vec<f32>> {
            match self {
                Self::Inkling(m) => m.step(token),
                Self::Bonsai(m) => m.step(token),
            }
        }
        fn step_greedy(&mut self, token: u32) -> Result<u32> {
            match self {
                Self::Inkling(m) => m.step_greedy(token),
                Self::Bonsai(m) => m.step_greedy(token),
            }
        }
        fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
            match self {
                Self::Inkling(m) => m.prefill(tokens),
                Self::Bonsai(m) => m.prefill(tokens),
            }
        }
        fn prefill_greedy(&mut self, tokens: &[u32]) -> Result<u32> {
            match self {
                Self::Inkling(m) => m.prefill_greedy(tokens),
                Self::Bonsai(m) => m.prefill_greedy(tokens),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn routes_both_architectures_from_metadata() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models");
        assert_eq!(
            Kind::detect(&root.join("inkling/tests/fixtures/tiny")).unwrap(),
            Kind::Inkling
        );
        assert_eq!(
            Kind::detect(&root.join("bonsai/tests/fixtures/tiny")).unwrap(),
            Kind::Bonsai
        );
    }
}
