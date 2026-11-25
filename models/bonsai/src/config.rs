//! The released Hadamard-aware MLX artifact, not a generic Qwen checkpoint.
use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use std::{collections::BTreeMap, path::Path};

#[derive(Clone, Debug, Deserialize)]
pub struct Config {
    pub schema_version: u32,
    pub model_type: String,
    pub text_config: TextConfig,
    pub modules: Vec<Module>,
    pub quantization: Quantization,
    pub tensor_namespace: String,
    pub gdn_activation_layout: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Module {
    pub path: String,
    pub block: usize,
    pub embedding: bool,
    pub dtype: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Quantization {
    pub bits: usize,
    pub group_size: usize,
    pub mode: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct TextConfig {
    pub model_type: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub eos_token_id: u32,
    pub max_position_embeddings: usize,
    pub full_attention_interval: usize,
    pub layer_types: Vec<String>,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_conv_kernel_dim: usize,
    pub rms_norm_eps: f32,
    pub rope_parameters: Rope,
    pub attention_bias: bool,
    pub tie_word_embeddings: bool,
    pub hidden_act: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Rope {
    pub rope_theta: f32,
    pub partial_rotary_factor: f32,
    #[serde(alias = "type")]
    pub rope_type: String,
}

impl Config {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let config: Self = serde_json::from_slice(&std::fs::read(path.as_ref())?)
            .context("reading Bonsai configuration")?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema_version == 2
                && self.model_type == "prism_hadamard_qwen35"
                && self.tensor_namespace == "mlx-vlm-qwen3_5"
                && self.gdn_activation_layout == "grouped",
            "unsupported Bonsai artifact schema, namespace, or recurrent-head layout"
        );
        ensure!(
            self.quantization.bits == 2
                && self.quantization.group_size == 128
                && self.quantization.mode == "affine",
            "Bonsai requires affine 2-bit/group-128 weights"
        );
        let t = &self.text_config;
        ensure!(
            (t.eos_token_id as usize) < t.vocab_size,
            "EOS token is outside the vocabulary"
        );
        ensure!(
            t.model_type == "qwen3_5_text"
                && !t.attention_bias
                && !t.tie_word_embeddings
                && t.hidden_act == "silu",
            "unsupported Bonsai text architecture"
        );
        for n in [
            t.hidden_size,
            t.intermediate_size,
            t.num_hidden_layers,
            t.num_attention_heads,
            t.num_key_value_heads,
            t.head_dim,
            t.vocab_size,
            t.max_position_embeddings,
            t.full_attention_interval,
            t.linear_num_key_heads,
            t.linear_num_value_heads,
            t.linear_key_head_dim,
            t.linear_value_head_dim,
            t.linear_conv_kernel_dim,
        ] {
            ensure!(n > 0 && n <= 1_048_576, "invalid Bonsai dimension {n}");
        }
        ensure!(
            t.hidden_size <= 32768
                && t.intermediate_size <= 131072
                && t.num_hidden_layers <= 256
                && t.num_attention_heads <= 256
                && t.linear_num_value_heads <= 256
                && t.head_dim <= 512
                && t.linear_key_head_dim <= 256
                && t.linear_value_head_dim <= 256
                && t.linear_conv_kernel_dim <= 16,
            "Bonsai dimensions exceed kernel limits"
        );
        ensure!(
            t.hidden_size % 128 == 0
                && t.intermediate_size % 128 == 0
                && (t.num_attention_heads * t.head_dim) % 128 == 0
                && t.linear_values() % 128 == 0,
            "packed projection widths must be divisible by 128"
        );
        ensure!(
            t.num_attention_heads % t.num_key_value_heads == 0
                && t.linear_num_value_heads % t.linear_num_key_heads == 0
                && t.linear_key_head_dim % 32 == 0,
            "invalid Bonsai head grouping"
        );
        ensure!(
            t.rms_norm_eps.is_finite()
                && t.rms_norm_eps > 0.0
                && t.rope_parameters.rope_theta.is_finite()
                && t.rope_parameters.rope_theta > 0.0
                && t.rope_parameters.rope_type == "default",
            "unsupported normalization or RoPE configuration"
        );
        let rotary = t.head_dim as f32 * t.rope_parameters.partial_rotary_factor;
        ensure!(
            rotary.is_finite()
                && rotary >= 2.0
                && rotary <= t.head_dim as f32
                && rotary.fract() == 0.0
                && rotary as usize % 2 == 0,
            "invalid rotary dimension"
        );
        ensure!(
            t.layer_types.len() == t.num_hidden_layers,
            "invalid layer schedule"
        );
        for (i, kind) in t.layer_types.iter().enumerate() {
            ensure!(
                kind == if t.is_full(i) {
                    "full_attention"
                } else {
                    "linear_attention"
                },
                "layer {i} disagrees with the attention interval"
            );
        }
        let expected = self.projections();
        ensure!(
            self.modules.len() == expected.len(),
            "incomplete packed-module manifest"
        );
        let mut seen = std::collections::BTreeSet::new();
        for m in &self.modules {
            let &(_, width) = expected
                .get(&m.path)
                .with_context(|| format!("unknown packed module {}", m.path))?;
            ensure!(seen.insert(&m.path), "duplicate packed module {}", m.path);
            ensure!(
                m.dtype == "float16" && m.embedding == (m.path == "model.embed_tokens"),
                "invalid packed module kind or activation dtype: {}",
                m.path
            );
            ensure!(
                m.block == 0
                    || (m.block.is_power_of_two()
                        && (128..=4096).contains(&m.block)
                        && width % m.block == 0),
                "invalid Hadamard block for {}",
                m.path
            );
        }
        Ok(())
    }

    /// Every packed matrix, with logical [output, input] dimensions.
    pub fn projections(&self) -> BTreeMap<String, (usize, usize)> {
        let t = &self.text_config;
        let h = t.hidden_size;
        let mut result = BTreeMap::from([
            ("model.embed_tokens".into(), (t.vocab_size, h)),
            ("lm_head".into(), (t.vocab_size, h)),
        ]);
        for i in 0..t.num_hidden_layers {
            let p = format!("model.layers.{i}");
            let mut add = |name: &str, rows, cols| {
                result.insert(format!("{p}.{name}"), (rows, cols));
            };
            add("mlp.gate_proj", t.intermediate_size, h);
            add("mlp.up_proj", t.intermediate_size, h);
            add("mlp.down_proj", h, t.intermediate_size);
            if t.is_full(i) {
                add(
                    "self_attn.q_proj",
                    t.num_attention_heads * t.head_dim * 2,
                    h,
                );
                add("self_attn.k_proj", t.num_key_value_heads * t.head_dim, h);
                add("self_attn.v_proj", t.num_key_value_heads * t.head_dim, h);
                add("self_attn.o_proj", h, t.num_attention_heads * t.head_dim);
            } else {
                add(
                    "linear_attn.in_proj_qkv",
                    t.linear_keys() * 2 + t.linear_values(),
                    h,
                );
                add("linear_attn.in_proj_z", t.linear_values(), h);
                add("linear_attn.out_proj", h, t.linear_values());
            }
        }
        result
    }
}

impl TextConfig {
    pub fn is_full(&self, layer: usize) -> bool {
        (layer + 1) % self.full_attention_interval == 0
    }
    pub fn linear_keys(&self) -> usize {
        self.linear_num_key_heads * self.linear_key_head_dim
    }
    pub fn linear_values(&self) -> usize {
        self.linear_num_value_heads * self.linear_value_head_dim
    }
    pub fn rotary_dim(&self) -> usize {
        (self.head_dim as f32 * self.rope_parameters.partial_rotary_factor) as usize
    }
    pub fn cache_bytes(&self, context: usize) -> Result<usize> {
        ensure!(
            context > 0 && context <= self.max_position_embeddings,
            "Bonsai context must be in 1..={}",
            self.max_position_embeddings
        );
        let full = (0..self.num_hidden_layers)
            .filter(|&i| self.is_full(i))
            .count();
        let kv = full
            .checked_mul(context)
            .and_then(|n| n.checked_mul(self.num_key_value_heads * self.head_dim * 4));
        let recurrent = (self.num_hidden_layers - full)
            * (self.linear_values() * self.linear_key_head_dim * 4
                + (self.linear_keys() * 2 + self.linear_values())
                    * (self.linear_conv_kernel_dim - 1)
                    * 2);
        kv.and_then(|n| n.checked_add(recurrent))
            .context("Bonsai cache size overflow")
    }
}
