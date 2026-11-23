//! Native Inkling still-image encoder. Matrix weights retain their checkpoint
//! precision; activations, LayerNorm, attention and accumulation use float32.
use crate::{
    checkpoint::Checkpoint,
    config::VisionConfig,
    image::Image,
    kernels::{Kernels, Params},
};
use crate::{name::name, workspace::Pool};
use anyhow::{Result, ensure};
use rink_core::{Accelerator, Buffer};
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
        progress: impl FnMut(usize, usize),
    ) -> Result<A::Buffer> {
        self.encode_with_pool(device, kernels, image, &mut Pool::new(), progress)
            .map(|(features, _storage)| features)
    }

    pub(crate) fn encode_with_pool(
        &self,
        device: &mut A,
        kernels: &Kernels<A>,
        image: &Image,
        pool: &mut Pool<A::Buffer>,
        mut progress: impl FnMut(usize, usize),
    ) -> Result<(A::Buffer, A::Buffer)> {
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
        // Keep simultaneously live vision tensors in distinct Metal resources.
        // This preserves the resource boundaries between independent projections.
        // Whole blocks still return to the text pool when their lifetimes end.
        let alloc =
            |device: &mut A, pool: &mut Pool<A::Buffer>, count: usize| pool.take(device, count * 4);
        // A bounded set of workspaces, reused for all layers. Attention does not
        // allocate a quadratic score matrix.
        let a = alloc(device, pool, seq * h)?;
        let b = pool.take(device, seq * h * 4)?;
        let norm = alloc(device, pool, seq * h)?;
        let q = alloc(device, pool, seq * h)?;
        let k = alloc(device, pool, seq * h)?;
        let v = alloc(device, pool, seq * h)?;
        // The attention result is consumed before a becomes the residual.
        // Normalized inputs are consumed before each projection writes its
        // result. Ordered dispatch lets these nonoverlapping lifetimes share
        // storage, saving two seq*h buffers without changing any operation.
        let attn = a.clone();
        let projected = norm.clone();
        let up = alloc(device, pool, seq * m)?;
        let pixels = alloc(device, pool, image.token_count() * 4800)?;
        // SAFETY: write_patches initializes exactly every float in the patch tensor.
        unsafe {
            device.initialize_f32(&pixels, image.token_count() * 4800, |values| {
                image.write_patches(values)
            })?;
        }
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
        // The last pixel read is encoded. No block is borrowed again until the
        // encoder layers have synchronized; pending commands retain the buffer.
        pool.put(pixels);
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
            let p = name!("backbone.layers.{i}");
            run.norm(&norm, &b, &name!("{p}.ln_1"), h, seq, c.layer_norm_eps)?;
            for (dst, name) in [(&q, "q_proj"), (&k, "k_proj"), (&v, "v_proj")] {
                run.linear(
                    dst,
                    &norm,
                    &name!("{p}.self_attn.{name}.weight"),
                    h,
                    h,
                    seq,
                    0,
                )?;
            }
            run.bias(&v, Some(&name!("{p}.self_attn.v_proj.bias")), h, seq, false)?;
            run.dispatch(
                "vision_rope",
                &[
                    &q,
                    &k,
                    run.weight(&name!("{p}.self_attn.q_proj.bias")),
                    run.weight(&name!("{p}.self_attn.k_proj.bias")),
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
            let (attention_kernel, attention_threads, query_tile) =
                if h / c.num_attention_heads == 64 {
                    if seq >= 2048 {
                        ("vision_attention_64x64", 256, 64)
                    } else {
                        ("vision_attention_64x16", 256, 16)
                    }
                } else {
                    ("vision_attention", 32, 8)
                };
            run.dispatch(
                attention_kernel,
                &[&attn, &q, &k, &v],
                &[seq, c.num_attention_heads, h / c.num_attention_heads],
                &[],
                seq.div_ceil(query_tile) * c.num_attention_heads * attention_threads,
                attention_threads,
            )?;
            run.linear(
                &projected,
                &attn,
                &name!("{p}.self_attn.out_proj.weight"),
                h,
                h,
                seq,
                0,
            )?;
            run.residual(
                &a,
                &b,
                &projected,
                &name!("{p}.self_attn.out_proj.bias"),
                &name!("{p}.ls_1.gamma"),
                h,
                seq,
            )?;
            run.norm(&norm, &a, &name!("{p}.ln_2"), h, seq, c.layer_norm_eps)?;
            run.linear(&up, &norm, &name!("{p}.mlp.c_fc.weight"), m, h, seq, 0)?;
            run.bias(&up, Some(&name!("{p}.mlp.c_fc.bias")), m, seq, true)?;
            run.linear(
                &projected,
                &up,
                &name!("{p}.mlp.c_proj.weight"),
                h,
                m,
                seq,
                0,
            )?;
            run.residual(
                &b,
                &a,
                &projected,
                &name!("{p}.mlp.c_proj.bias"),
                &name!("{p}.ls_2.gamma"),
                h,
                seq,
            )?;
            run.device.synchronize()?;
            progress(i + 1, c.num_hidden_layers);
        }
        // Only the residual stream feeds the projector. The last layer has
        // synchronized, so its other workspaces can be released now.
        drop((attn, projected));
        for buffer in [a, norm, q, k, v, up] {
            pool.put(buffer);
        }
        let merged = alloc(run.device, pool, tokens * c.projection_hidden_dim)?;
        let storage = pool.take(run.device, tokens * c.text_hidden_size * 4)?;
        let features = storage.view(0, tokens * c.text_hidden_size * 4)?;
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
        pool.put(merged);
        pool.put(b);
        Ok((features, storage))
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
        // Reuse the decoder's ordered FP32 matrix tiles for large image
        // projections. Small images retain the smaller workgroups.
        let wide = batch >= 512 && rows >= 384;
        let tile = if wide && rows >= 1024 {
            64
        } else if wide {
            32
        } else {
            16
        };
        let compact =
            tile == 64 && (rows >= 2048 || batch >= 4096) && cols % 32 == 0 && w.dtype == 1;
        let threads = if compact { 128 } else { 256 };
        self.dispatch(
            if compact {
                "matmul_prefill_compact"
            } else if tile == 64 {
                "matmul_prefill_m64"
            } else if wide {
                "matmul_prefill_m32"
            } else {
                "matmul_packed"
            },
            &[out, &w.buffer, x],
            &[rows, cols, w.dtype, batch, offset],
            &[],
            rows.div_ceil(if wide { 64 } else { 32 }) * batch.div_ceil(tile) * threads,
            threads,
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
                self.weight(&name!("{name}.weight")),
                self.weight(&name!("{name}.bias")),
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
