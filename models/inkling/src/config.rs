use rink_core::{Result, ensure};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Config {
    pub model_type: String,
    pub eos_token_id: u32,
    pub text_config: TextConfig,
    #[serde(default)]
    pub vision_config: Option<VisionConfig>,
    #[serde(default = "default_image_token")]
    pub image_token_id: u32,
}

fn default_image_token() -> u32 {
    200054
}

/// Native checkpoint dimensions for the still-image encoder.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct VisionConfig {
    pub model_type: String,
    pub text_hidden_size: usize,
    pub native_patch_size: usize,
    pub adapter_merge_size: usize,
    pub projection_hidden_dim: usize,
    pub projector_bias: bool,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub mlp_ratio: f32,
    #[serde(default)]
    pub intermediate_size: Option<usize>,
    #[serde(default = "default_vision_channels")]
    pub num_channels: usize,
    pub posemb_grid_size: usize,
    pub layer_norm_eps: f32,
    #[serde(default = "default_vision_activation")]
    pub hidden_act: String,
    #[serde(default)]
    pub rope_parameters: Option<serde_json::Value>,
}

fn default_vision_activation() -> String {
    "gelu".into()
}
fn default_vision_channels() -> usize {
    3
}

impl VisionConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.hidden_act == "gelu",
            "vision encoder requires GELU activation"
        );
        ensure!(self.num_channels == 3, "vision encoder requires RGB input");
        if let Some(rope) = &self.rope_parameters {
            ensure!(
                rope.get("rope_type").and_then(|v| v.as_str()) == Some("axial")
                    && rope.get("rope_theta").and_then(|v| v.as_f64()) == Some(10_000.)
                    && rope.as_object().is_some_and(|v| v.len() == 2),
                "vision encoder requires the native axial RoPE parameters"
            );
        }
        ensure!(
            self.model_type == "inkling2_vision",
            "unsupported vision architecture"
        );
        ensure!(
            self.native_patch_size == 20 && self.adapter_merge_size == 2,
            "image processing requires 20-pixel patches and a 2x2 merge"
        );
        ensure!(
            (1..=128).contains(&self.num_hidden_layers)
                && (1..=4096).contains(&self.hidden_size)
                && self.num_attention_heads > 0
                && self.hidden_size % self.num_attention_heads == 0,
            "invalid vision dimensions"
        );
        let dim = self.hidden_size / self.num_attention_heads;
        ensure!(
            dim % 8 == 0 && dim <= 128,
            "vision head width must be a multiple of 8, at most 128"
        );
        ensure!(
            self.mlp_ratio.is_finite()
                && self.mlp_ratio >= 1.
                && self.mlp_ratio <= 16.
                && (self.hidden_size as f32 * self.mlp_ratio).fract() == 0.,
            "invalid vision MLP ratio"
        );
        ensure!(
            self.intermediate_size
                .is_none_or(|n| n == self.intermediate_size()),
            "vision intermediate size must match the MLP ratio"
        );
        ensure!(
            (1..=128).contains(&self.posemb_grid_size)
                && (1..=16384).contains(&self.projection_hidden_dim)
                && (1..=8192).contains(&self.text_hidden_size)
                && !self.projector_bias,
            "unsupported vision projector or position grid"
        );
        ensure!(
            self.layer_norm_eps.is_finite() && self.layer_norm_eps > 0.,
            "invalid vision normalization epsilon"
        );
        Ok(())
    }
    pub fn intermediate_size(&self) -> usize {
        (self.hidden_size as f32 * self.mlp_ratio) as usize
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct LoopConfig {
    pub core_start: usize,
    pub core_end: usize,
    pub num_passes: usize,
    pub gate_rank: usize,
    pub is_varloop: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct TextConfig {
    pub model_type: String,
    pub model_max_length: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub vocab_size: usize,
    pub unpadded_vocab_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub swa_num_attention_heads: usize,
    pub swa_num_key_value_heads: usize,
    pub swa_head_dim: usize,
    pub d_rel: usize,
    pub rel_extent: usize,
    pub sliding_window_size: usize,
    pub local_layer_ids: Vec<usize>,
    pub sconv_kernel_size: usize,
    pub n_layers_per_attn_res_block: usize,
    pub n_routed_experts: usize,
    pub num_experts_per_tok: usize,
    pub n_shared_experts: usize,
    pub intermediate_size: usize,
    pub rms_norm_eps: f32,
    pub qk_l2_clip_threshold: f32,
    pub rel_l2_clip_threshold: f32,
    pub log_scaling_n_floor: usize,
    pub log_scaling_alpha: f32,
    pub mlp_gate_softcap: f32,
    pub mlp_up_softcap: f32,
    pub final_logit_softcapping: f32,
    pub route_scale: f32,
    pub attn_res_normalize_values: bool,
    pub use_value_norm_no_gamma: bool,
    pub use_gate_bias: bool,
    pub q_bias: bool,
    pub o_bias: bool,
    pub loop_config: LoopConfig,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Visit {
    pub layer: usize,
    pub pass: usize,
    pub cache: usize,
}

impl Config {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let config: Self = serde_json::from_slice(&std::fs::read(path)?)?;
        config.validate()?;
        Ok(config)
    }
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.model_type == "inkling2_loop"
                && self.text_config.model_type == "inkling2_loop_text",
            "expected an original Inkling 2 Mini configuration"
        );
        self.text_config.validate()?;
        if let Some(vision) = &self.vision_config {
            vision.validate()?;
            ensure!(
                vision.text_hidden_size == self.text_config.hidden_size,
                "vision and text hidden sizes differ"
            );
            ensure!(
                (self.image_token_id as usize) < self.text_config.unpadded_vocab_size,
                "image token is outside vocabulary"
            );
        }
        ensure!(
            (self.eos_token_id as usize) < self.text_config.unpadded_vocab_size,
            "EOS is outside vocabulary"
        );
        Ok(())
    }
}
impl TextConfig {
    pub fn validate(&self) -> Result<()> {
        let l = &self.loop_config;
        ensure!(
            (1..=128).contains(&self.num_hidden_layers),
            "invalid layer count"
        );
        ensure!(
            l.core_start > 0 && l.core_start < l.core_end && l.core_end <= self.num_hidden_layers,
            "invalid loop range"
        );
        ensure!(
            (1..=16).contains(&l.num_passes) && (1..=1024).contains(&l.gate_rank) && !l.is_varloop,
            "unsupported loop configuration"
        );
        ensure!(
            self.n_layers_per_attn_res_block > 0
                && self
                    .num_hidden_layers
                    .div_ceil(self.n_layers_per_attn_res_block)
                    < 32,
            "invalid attention residual block size"
        );
        ensure!(
            l.core_start % self.n_layers_per_attn_res_block == 0
                && l.core_end % self.n_layers_per_attn_res_block == 0,
            "loop must align with residual blocks"
        );
        ensure!(
            (1..=8192).contains(&self.hidden_size) && (1..=32768).contains(&self.intermediate_size),
            "unsupported hidden dimensions"
        );
        ensure!(
            (1..=256).contains(&self.n_routed_experts)
                && (1..=32).contains(&self.num_experts_per_tok)
                && self.num_experts_per_tok <= self.n_routed_experts,
            "invalid expert routing"
        );
        ensure!(self.n_shared_experts == 0, "shared experts are unsupported");
        for (q, k, d) in [
            (
                self.num_attention_heads,
                self.num_key_value_heads,
                self.head_dim,
            ),
            (
                self.swa_num_attention_heads,
                self.swa_num_key_value_heads,
                self.swa_head_dim,
            ),
        ] {
            ensure!(
                q > 0 && q <= 128 && k > 0 && q % k == 0 && d > 0 && d <= 512,
                "invalid attention geometry"
            );
        }
        ensure!(
            self.d_rel > 0
                && self.d_rel <= 128
                && self.rel_extent > 0
                && self.rel_extent <= 1048576,
            "invalid relative attention dimensions"
        );
        ensure!(
            self.sliding_window_size > 0 && self.sliding_window_size <= self.model_max_length,
            "invalid sliding window"
        );
        ensure!(
            self.local_layer_ids
                .iter()
                .all(|&i| i < self.num_hidden_layers),
            "local layer ID out of range"
        );
        ensure!(
            (1..=16).contains(&self.sconv_kernel_size),
            "invalid convolution size"
        );
        ensure!(
            self.model_max_length > 0
                && self.model_max_length <= 1048576
                && self.vocab_size > 0
                && self.vocab_size <= 1048576
                && self.unpadded_vocab_size > 0
                && self.unpadded_vocab_size <= self.vocab_size,
            "invalid context or vocabulary"
        );
        for x in [
            self.rms_norm_eps,
            self.qk_l2_clip_threshold,
            self.rel_l2_clip_threshold,
            self.mlp_gate_softcap,
            self.mlp_up_softcap,
            self.final_logit_softcapping,
            self.route_scale,
        ] {
            ensure!(
                x.is_finite() && x > 0.,
                "non-positive or non-finite model constant"
            );
        }
        ensure!(
            self.log_scaling_n_floor > 0
                && self.log_scaling_alpha.is_finite()
                && self.log_scaling_alpha >= 0.,
            "invalid log scaling"
        );
        ensure!(
            self.attn_res_normalize_values
                && self.use_value_norm_no_gamma
                && self.use_gate_bias
                && !self.q_bias
                && !self.o_bias,
            "unsupported normalization or bias configuration"
        );
        Ok(())
    }
    pub fn is_local(&self, layer: usize) -> bool {
        self.local_layer_ids.contains(&layer)
    }
    pub fn attention_shape(&self, layer: usize) -> (usize, usize, usize) {
        if self.is_local(layer) {
            (
                self.swa_num_attention_heads,
                self.swa_num_key_value_heads,
                self.swa_head_dim,
            )
        } else {
            (
                self.num_attention_heads,
                self.num_key_value_heads,
                self.head_dim,
            )
        }
    }
    /// Cache indices follow execution order; physical weights remain shared across passes.
    pub fn schedule(&self) -> Vec<Visit> {
        let l = &self.loop_config;
        let layers = (0..l.core_start)
            .map(|i| (i, 0))
            .chain((0..l.num_passes).flat_map(|p| (l.core_start..l.core_end).map(move |i| (i, p))))
            .chain((l.core_end..self.num_hidden_layers).map(|i| (i, 0)));
        layers
            .enumerate()
            .map(|(cache, (layer, pass))| Visit { layer, pass, cache })
            .collect()
    }
}
