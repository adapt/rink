//! Native Inkling still-image encoder. Matrix weights retain their checkpoint
//! precision; activations, LayerNorm, attention and accumulation use float32.
use crate::{
    checkpoint::Checkpoint,
    config::VisionConfig,
    image::Image,
    kernels::{Kernels, Params},
};
use anyhow::{Result, ensure};
use rink_core::Accelerator;
use safetensors::Dtype;
use std::collections::HashMap;

struct Weight<B> {
    buffer: B,
    dtype: usize,
}

pub struct VisionEncoder<A: Accelerator> {
    config: VisionConfig,
    weights: HashMap<String, Weight<A::Buffer>>,
}

impl<A: Accelerator> VisionEncoder<A> {
    pub fn load(device: &mut A, checkpoint: &Checkpoint, config: &VisionConfig) -> Result<Self> {
        config.validate()?;
        let h = config.hidden_size;
        let m = config.intermediate_size();
        let mut specs = vec![
            (
                "backbone.embeddings.patch_embedding.weight".to_owned(),
                vec![h, 1200],
            ),
            (
                "backbone.embeddings.position_embedding".into(),
                vec![config.posemb_grid_size.pow(2), h],
            ),
            ("backbone.embeddings.class_embedding".into(), vec![h]),
            ("backbone.embeddings.cls_position_embedding".into(), vec![h]),
            ("backbone.ln_pre.weight".into(), vec![h]),
            ("backbone.ln_pre.bias".into(), vec![h]),
            (
                "proj_in.weight".into(),
                vec![config.projection_hidden_dim, h * 4],
            ),
            (
                "proj_out.weight".into(),
                vec![config.text_hidden_size, config.projection_hidden_dim],
            ),
        ];
        for i in 0..config.num_hidden_layers {
            let prefix = format!("backbone.layers.{i}");
            for ln in ["ln_1", "ln_2"] {
                for suffix in ["weight", "bias"] {
                    specs.push((format!("{prefix}.{ln}.{suffix}"), vec![h]));
                }
            }
            for ls in ["ls_1", "ls_2"] {
                specs.push((format!("{prefix}.{ls}.gamma"), vec![h]));
            }
            for proj in ["q_proj", "k_proj", "v_proj", "out_proj"] {
                specs.push((format!("{prefix}.self_attn.{proj}.weight"), vec![h, h]));
                specs.push((format!("{prefix}.self_attn.{proj}.bias"), vec![h]));
            }
            for (proj, rows, cols) in [("c_fc", m, h), ("c_proj", h, m)] {
                specs.push((format!("{prefix}.mlp.{proj}.weight"), vec![rows, cols]));
                specs.push((format!("{prefix}.mlp.{proj}.bias"), vec![rows]));
            }
        }
        let mut weights = HashMap::new();
        for (name, shape) in specs {
            let t = checkpoint.tensor(&format!("model.visual.{name}"))?;
            ensure!(
                t.shape() == shape,
                "{name}: expected {shape:?}, got {:?}",
                t.shape()
            );
            ensure!(
                shape.iter().product::<usize>() <= u32::MAX as usize,
                "vision tensor too large"
            );
            let (buffer, dtype) = if shape.len() == 1 {
                let values = crate::model::floats(&t)?;
                (device.upload(bytemuck::cast_slice(&values))?, 0)
            } else {
                let dtype = match t.dtype() {
                    Dtype::F32 => 0,
                    Dtype::BF16 => 1,
                    Dtype::F16 => 2,
                    other => anyhow::bail!("{name}: unsupported vision weight dtype {other:?}"),
                };
                (device.upload(t.data())?, dtype)
            };
            weights.insert(name, Weight { buffer, dtype });
        }
        Ok(Self {
            config: config.clone(),
            weights,
        })
    }

    /// Return one decoder embedding per 40x40 image tile, in row-major order.
    /// This does not alter the text decoder or its KV caches.
    pub fn encode(&self, device: &mut A, kernels: &Kernels<A>, image: &Image) -> Result<A::Buffer> {
        self.encode_with_progress(device, kernels, image, |_, _| {})
    }

    /// Report completed layers after their existing GPU synchronization.
    /// The final projector runs after the last layer update.
    pub fn encode_with_progress(
        &self,
        device: &mut A,
        kernels: &Kernels<A>,
        image: &Image,
        mut progress: impl FnMut(usize, usize),
    ) -> Result<A::Buffer> {
        let c = &self.config;
        let h = c.hidden_size;
        let m = c.intermediate_size();
        let (rows, cols) = image.grid();
        let tokens = rows * cols;
        let patches = tokens * 4;
        let seq = patches + 1;
        // The MLP is the largest native-patch workspace (m >= h). At 16K,
        // wider configurations can exceed the kernels' 32-bit element indices.
        ensure!(
            seq.checked_mul(m).is_some_and(|n| n <= u32::MAX as usize),
            "image is too large for this vision encoder's 32-bit indexing"
        );
        progress(0, c.num_hidden_layers);
        let alloc = |device: &mut A, count: usize| device.allocate(count * 4);
        // A bounded set of workspaces, reused for all layers. Attention does not
        // allocate a quadratic score matrix.
        let a = alloc(device, seq * h)?;
        let b = alloc(device, seq * h)?;
        let norm = alloc(device, seq * h)?;
        let q = alloc(device, seq * h)?;
        let k = alloc(device, seq * h)?;
        let v = alloc(device, seq * h)?;
        let attn = alloc(device, seq * h)?;
        let projected = alloc(device, seq * h)?;
        let up = alloc(device, seq * m)?;
        let pixels = device.upload(bytemuck::cast_slice(&image.patches()))?;
        let mut run = Runner {
            device,
            kernels,
            weights: &self.weights,
        };
        run.linear(
            &projected,
            &pixels,
            "backbone.embeddings.patch_embedding.weight",
            h,
            1200,
            patches,
            0,
        )?;
        let pos = &self.weights["backbone.embeddings.position_embedding"];
        run.dispatch(
            "vision_positions",
            &[
                &a,
                &projected,
                &pos.buffer,
                run.weight("backbone.embeddings.class_embedding"),
                run.weight("backbone.embeddings.cls_position_embedding"),
            ],
            &[h, rows * 2, cols * 2, c.posemb_grid_size, pos.dtype],
            &[],
            seq * h,
            128,
        )?;
        run.norm(&b, &a, "backbone.ln_pre", h, seq, c.layer_norm_eps)?;
        // b is the live residual stream at both the start and end of each layer.
        for i in 0..c.num_hidden_layers {
            let p = format!("backbone.layers.{i}");
            run.norm(&norm, &b, &format!("{p}.ln_1"), h, seq, c.layer_norm_eps)?;
            for (dst, name) in [(&q, "q_proj"), (&k, "k_proj"), (&v, "v_proj")] {
                run.linear(
                    dst,
                    &norm,
                    &format!("{p}.self_attn.{name}.weight"),
                    h,
                    h,
                    seq,
                    0,
                )?;
            }
            run.bias(
                &v,
                Some(&format!("{p}.self_attn.v_proj.bias")),
                h,
                seq,
                false,
            )?;
            run.dispatch(
                "vision_rope",
                &[
                    &q,
                    &k,
                    run.weight(&format!("{p}.self_attn.q_proj.bias")),
                    run.weight(&format!("{p}.self_attn.k_proj.bias")),
                ],
                &[
                    seq,
                    c.num_attention_heads,
                    h / c.num_attention_heads,
                    cols * 2,
                ],
                &[10_000.],
                seq * h / 2,
                128,
            )?;
            run.dispatch(
                "vision_attention",
                &[&attn, &q, &k, &v],
                &[seq, c.num_attention_heads, h / c.num_attention_heads],
                &[],
                seq.div_ceil(8) * c.num_attention_heads * 32,
                32,
            )?;
            run.linear(
                &projected,
                &attn,
                &format!("{p}.self_attn.out_proj.weight"),
                h,
                h,
                seq,
                0,
            )?;
            run.residual(
                &a,
                &b,
                &projected,
                &format!("{p}.self_attn.out_proj.bias"),
                &format!("{p}.ls_1.gamma"),
                h,
                seq,
            )?;
            run.norm(&norm, &a, &format!("{p}.ln_2"), h, seq, c.layer_norm_eps)?;
            run.linear(&up, &norm, &format!("{p}.mlp.c_fc.weight"), m, h, seq, 0)?;
            run.bias(&up, Some(&format!("{p}.mlp.c_fc.bias")), m, seq, true)?;
            run.linear(
                &projected,
                &up,
                &format!("{p}.mlp.c_proj.weight"),
                h,
                m,
                seq,
                0,
            )?;
            run.residual(
                &b,
                &a,
                &projected,
                &format!("{p}.mlp.c_proj.bias"),
                &format!("{p}.ls_2.gamma"),
                h,
                seq,
            )?;
            run.device.synchronize()?;
            progress(i + 1, c.num_hidden_layers);
        }
        let merged = alloc(run.device, tokens * c.projection_hidden_dim)?;
        let features = alloc(run.device, tokens * c.text_hidden_size)?;
        // Skip CLS; the four native patches in a tile are already contiguous.
        run.linear(
            &merged,
            &b,
            "proj_in.weight",
            c.projection_hidden_dim,
            h * 4,
            tokens,
            h,
        )?;
        run.bias(&merged, None, c.projection_hidden_dim, tokens, true)?;
        run.linear(
            &features,
            &merged,
            "proj_out.weight",
            c.text_hidden_size,
            c.projection_hidden_dim,
            tokens,
            0,
        )?;
        run.device.synchronize()?;
        Ok(features)
    }
}

struct Runner<'a, A: Accelerator> {
    device: &'a mut A,
    kernels: &'a Kernels<A>,
    weights: &'a HashMap<String, Weight<A::Buffer>>,
}

impl<'a, A: Accelerator> Runner<'a, A> {
    fn weight(&self, name: &str) -> &'a A::Buffer {
        &self.weights[name].buffer
    }
    fn dispatch(
        &mut self,
        name: &str,
        buffers: &[&A::Buffer],
        ints: &[usize],
        floats: &[f32],
        threads: usize,
        group: usize,
    ) -> Result<()> {
        // SAFETY: validated checkpoint dimensions and allocated F32 workspaces
        // match vision.metal and the shared matmul ABI.
        unsafe {
            self.kernels.run(
                self.device,
                name,
                buffers,
                Params::new(ints, floats),
                threads,
                group,
            )
        }
    }
    #[allow(clippy::too_many_arguments)]
    fn linear(
        &mut self,
        out: &A::Buffer,
        x: &A::Buffer,
        name: &str,
        rows: usize,
        cols: usize,
        batch: usize,
        offset: usize,
    ) -> Result<()> {
        let w = &self.weights[name];
        self.dispatch(
            "matmul_packed",
            &[out, &w.buffer, x],
            &[rows, cols, w.dtype, batch, offset],
            &[],
            rows.div_ceil(32) * batch.div_ceil(16) * 256,
            256,
        )
    }
    fn norm(
        &mut self,
        out: &A::Buffer,
        x: &A::Buffer,
        name: &str,
        width: usize,
        rows: usize,
        eps: f32,
    ) -> Result<()> {
        self.dispatch(
            "vision_norm",
            &[
                out,
                x,
                self.weight(&format!("{name}.weight")),
                self.weight(&format!("{name}.bias")),
            ],
            &[width, rows],
            &[eps],
            rows * 32,
            32,
        )
    }
    fn bias(
        &mut self,
        out: &A::Buffer,
        name: Option<&str>,
        width: usize,
        rows: usize,
        gelu: bool,
    ) -> Result<()> {
        self.dispatch(
            "vision_bias",
            &[out, name.map_or(out, |n| self.weight(n))],
            &[width, rows, usize::from(gelu), usize::from(name.is_some())],
            &[],
            width * rows,
            128,
        )
    }
    #[allow(clippy::too_many_arguments)]
    fn residual(
        &mut self,
        out: &A::Buffer,
        x: &A::Buffer,
        projected: &A::Buffer,
        bias: &str,
        scale: &str,
        width: usize,
        rows: usize,
    ) -> Result<()> {
        self.dispatch(
            "vision_residual",
            &[out, x, projected, self.weight(bias), self.weight(scale)],
            &[width, rows],
            &[],
            width * rows,
            128,
        )
    }
}
