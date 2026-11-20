use crate::{
    checkpoint::Checkpoint,
    config::{Config, TextConfig, Visit},
    kernels::{Kernels, Params},
};
use anyhow::{Context, Result, bail, ensure};
use rink_core::{Accelerator, AutoregressiveModel};
use safetensors::Dtype;
use std::{collections::HashMap, path::Path};

#[derive(Clone)]
struct Weight<B> {
    buffer: B,
    dtype: usize,
}
struct Expert<B> {
    weight: B,
    scales: B,
    factors: B,
    format: usize,
}
type ExpertPair<B> = (Expert<B>, Expert<B>);

struct Cache<B> {
    keys: B,
    values: B,
    conv: Vec<B>,
    capacity: usize,
}

/// Single-sequence, streaming text decoder for the original Inkling 2 Mini checkpoint.
/// Uses float32 activations and accumulation, with native packed expert storage.
pub struct Inkling<A: Accelerator> {
    device: A,
    kernels: Kernels<A>,
    config: Config,
    weights: HashMap<String, Weight<A::Buffer>>,
    experts: Vec<ExpertPair<A::Buffer>>,
    caches: Vec<Cache<A::Buffer>>,
    capacity: usize,
    position: usize,
    failed: bool,
}

impl<A: Accelerator> Inkling<A> {
    pub fn load(mut device: A, directory: &Path, capacity: usize) -> Result<Self> {
        let config = Config::load(directory.join("config.json"))?;
        ensure!(
            capacity > 0 && capacity <= config.text_config.model_max_length,
            "context capacity must be in 1..={}",
            config.text_config.model_max_length
        );
        for layer in 0..config.text_config.num_hidden_layers {
            let (_, heads, dim) = config.text_config.attention_shape(layer);
            ensure!(
                capacity
                    .checked_mul(heads)
                    .and_then(|n| n.checked_mul(dim))
                    .is_some_and(|n| n <= u32::MAX as usize),
                "KV cache exceeds kernel indexing range"
            );
        }
        let checkpoint = Checkpoint::open(directory)?;
        let kernels = Kernels::compile(&mut device)?;
        let weights = load_weights(&mut device, &checkpoint, &config.text_config)?;
        let mut experts = Vec::new();
        let c = &config.text_config;
        for layer in 0..c.num_hidden_layers {
            let prefix = format!("layers.{layer}.mlp.experts");
            experts.push((
                load_expert(
                    &mut device,
                    &checkpoint,
                    &format!("{prefix}.w13_weight"),
                    c.n_routed_experts,
                    c.intermediate_size * 2,
                    c.hidden_size,
                )?,
                load_expert(
                    &mut device,
                    &checkpoint,
                    &format!("{prefix}.w2_weight"),
                    c.n_routed_experts,
                    c.hidden_size,
                    c.intermediate_size,
                )?,
            ));
        }
        let mut model = Self {
            device,
            kernels,
            config,
            weights,
            experts,
            caches: Vec::new(),
            capacity,
            position: 0,
            failed: false,
        };
        model.reset()?;
        Ok(model)
    }
    pub fn config(&self) -> &Config {
        &self.config
    }
    pub fn device_name(&self) -> &str {
        self.device.name()
    }

    fn alloc(&mut self, n: usize) -> Result<A::Buffer> {
        self.device
            .allocate(n.checked_mul(4).context("buffer size overflow")?)
    }
    fn w(&self, name: &str) -> A::Buffer {
        self.weights[name].buffer.clone()
    }
    // All calls below are internal: tensor shapes are validated when loading the model,
    // activation lengths follow the validated configuration, and cache indices are bounded.
    fn run(
        &mut self,
        name: &str,
        buffers: &[&A::Buffer],
        ints: &[usize],
        floats: &[f32],
        threads: usize,
        group: usize,
    ) -> Result<()> {
        // SAFETY: each call site follows the ABI in kernels.metal, with owned device buffers.
        unsafe {
            self.kernels.run(
                &mut self.device,
                name,
                buffers,
                Params::new(ints, floats),
                threads,
                group,
            )
        }
    }
    fn copy(
        &mut self,
        dst: &A::Buffer,
        dst_offset: usize,
        src: &A::Buffer,
        src_offset: usize,
        n: usize,
    ) -> Result<()> {
        self.run(
            "copy_vec",
            &[dst, src],
            &[n, dst_offset, src_offset],
            &[],
            n,
            128,
        )
    }
    fn linear(&mut self, name: &str, x: &A::Buffer, rows: usize, cols: usize) -> Result<A::Buffer> {
        let w = self.weights[name].clone();
        let y = self.alloc(rows)?;
        self.run(
            "matvec",
            &[&y, &w.buffer, x],
            &[rows, cols, w.dtype],
            &[],
            rows * 32,
            128,
        )?;
        Ok(y)
    }
    fn norm(
        &mut self,
        x: &A::Buffer,
        name: Option<&str>,
        width: usize,
        rows: usize,
    ) -> Result<A::Buffer> {
        let w = name.map(|n| self.w(n)).unwrap_or_else(|| x.clone());
        let y = self.alloc(width * rows)?;
        self.run(
            "rms",
            &[&y, x, &w],
            &[width, rows, usize::from(name.is_some())],
            &[self.config.text_config.rms_norm_eps],
            rows * 32,
            32,
        )?;
        Ok(y)
    }
    fn add(&mut self, a: &A::Buffer, b: &A::Buffer) -> Result<A::Buffer> {
        let n = self.config.text_config.hidden_size;
        let y = self.alloc(n)?;
        self.run("add", &[&y, a, b], &[n], &[], n, 128)?;
        Ok(y)
    }
    fn read_blocks(
        &mut self,
        blocks: &[A::Buffer],
        query: &str,
        offset: usize,
    ) -> Result<A::Buffer> {
        let n = self.config.text_config.hidden_size;
        let stack = self.alloc(blocks.len() * n)?;
        for (i, b) in blocks.iter().enumerate() {
            self.copy(&stack, i * n, b, 0, n)?;
        }
        let q = self.w(query);
        let y = self.alloc(n)?;
        self.run(
            "block_read",
            &[&y, &stack, &q],
            &[n, blocks.len(), offset],
            &[],
            32,
            32,
        )?;
        Ok(y)
    }
    fn message_gate(&mut self, x: &A::Buffer, msg: &A::Buffer, prefix: &str) -> Result<A::Buffer> {
        let h = self.config.text_config.hidden_size;
        let rank = self.config.text_config.loop_config.gate_rank;
        let cat = self.alloc(2 * h)?;
        self.run(
            "concat_norm",
            &[&cat, x, msg],
            &[h],
            &[self.config.text_config.rms_norm_eps],
            32,
            32,
        )?;
        let low = self.linear(&format!("{prefix}.down_weight"), &cat, rank, 2 * h)?;
        let forget = self.linear(&format!("{prefix}.forget_weight"), &low, h, rank)?;
        let gain = self.linear(&format!("{prefix}.gain_weight"), &low, h, rank)?;
        let y = self.alloc(h)?;
        self.run(
            "controller",
            &[&y, x, msg, &forget, &gain],
            &[h],
            &[],
            h,
            128,
        )?;
        Ok(y)
    }
    fn conv(
        &mut self,
        x: &A::Buffer,
        dynamic_input: Option<&A::Buffer>,
        prefix: &str,
        visit: Visit,
        state: usize,
        channels: usize,
    ) -> Result<A::Buffer> {
        let c = &self.config.text_config;
        let k = c.sconv_kernel_size;
        let (_, heads, dim) = c.attention_shape(visit.layer);
        let h = c.hidden_size;
        let (w, dyn_weights) = if let Some(input) = dynamic_input {
            (
                self.w(&format!("{prefix}.static_w")),
                self.linear(&format!("{prefix}.weight_proj"), input, heads * k, h)?,
            )
        } else {
            (self.w(&format!("{prefix}.weight")), x.clone())
        };
        let history = self.caches[visit.cache].conv[state].clone();
        let y = self.alloc(channels)?;
        self.run(
            "convolution",
            &[&y, x, &w, &history, &dyn_weights],
            &[
                channels,
                k,
                self.position,
                usize::from(dynamic_input.is_some()),
                dim,
            ],
            &[],
            channels,
            128,
        )?;
        Ok(y)
    }
    fn clip(
        &mut self,
        x: &A::Buffer,
        scale: Option<&str>,
        width: usize,
        heads: usize,
        threshold: f32,
        tau: f32,
    ) -> Result<A::Buffer> {
        let w = scale.map(|n| self.w(n)).unwrap_or_else(|| x.clone());
        let y = self.alloc(width * heads)?;
        self.run(
            "clip",
            &[&y, x, &w],
            &[width, heads, usize::from(scale.is_some())],
            &[threshold, tau],
            heads * 32,
            32,
        )?;
        Ok(y)
    }
    fn attention(&mut self, inputs: &[A::Buffer], visit: Visit) -> Result<A::Buffer> {
        let c = self.config.text_config.clone();
        let (qh, kh, d) = c.attention_shape(visit.layer);
        let local = c.is_local(visit.layer);
        let p = format!("layers.{}.attn", visit.layer);
        let q = self.linear(
            &format!("{p}.wq_du.weight"),
            &inputs[0],
            qh * d,
            c.hidden_size,
        )?;
        let k = self.linear(
            &format!("{p}.wk_dv.weight"),
            &inputs[1],
            kh * d,
            c.hidden_size,
        )?;
        let v = self.linear(
            &format!("{p}.wv_dv.weight"),
            &inputs[2],
            kh * d,
            c.hidden_size,
        )?;
        let r = self.linear(
            &format!("{p}.wr_du.weight"),
            &inputs[0],
            qh * c.d_rel,
            c.hidden_size,
        )?;
        let gate = self.linear(
            &format!("{p}.gate_attn.weight"),
            &inputs[0],
            qh * d,
            c.hidden_size,
        )?;
        let sink = self.linear(
            &format!("{p}.attn_sink_proj.weight"),
            &inputs[0],
            qh,
            c.hidden_size,
        )?;
        let k = self.conv(
            &k,
            if local { None } else { Some(&inputs[1]) },
            &format!("{p}.k_sconv"),
            visit,
            0,
            kh * d,
        )?;
        let v = self.conv(
            &v,
            if local { None } else { Some(&inputs[2]) },
            &format!("{p}.v_sconv"),
            visit,
            1,
            kh * d,
        )?;
        let tau = if local {
            1.
        } else {
            1. + c.log_scaling_alpha
                * (((self.position + 1) as f32 / c.log_scaling_n_floor as f32).max(1.)).ln()
        };
        let q = self.clip(
            &q,
            Some(&format!("{p}.q_l2_clip.scale")),
            d,
            qh,
            c.qk_l2_clip_threshold,
            tau,
        )?;
        let k = self.clip(&k, None, d, kh, c.qk_l2_clip_threshold, 1.)?;
        let v = self.norm(&v, None, d, kh)?;
        let r = self.clip(
            &r,
            Some(&format!("{p}.rel_l2_clip.scale")),
            c.d_rel,
            qh,
            c.rel_l2_clip_threshold,
            tau,
        )?;
        let cache = &self.caches[visit.cache];
        let (keys, values, capacity) = (cache.keys.clone(), cache.values.clone(), cache.capacity);
        self.copy(&keys, (self.position % capacity) * kh * d, &k, 0, kh * d)?;
        self.copy(&values, (self.position % capacity) * kh * d, &v, 0, kh * d)?;
        let n = (self.position + 1).min(capacity);
        let extent = if local {
            c.sliding_window_size
        } else {
            c.rel_extent
        };
        let geometry = [qh, kh, d, self.position, capacity, n, c.d_rel, extent];
        let scores = self.alloc(qh * n)?;
        let profiles = self.w(&format!("{p}.rel_logits_proj.proj"));
        self.run(
            "attn_scores",
            &[&scores, &q, &keys, &r, &profiles],
            &geometry,
            &[],
            qh * n * 32,
            128,
        )?;
        let attended = self.alloc(qh * d)?;
        self.run(
            "attn_values",
            &[&attended, &scores, &values, &sink],
            &geometry,
            &[],
            qh * 32,
            32,
        )?;
        let gated = self.alloc(qh * d)?;
        self.run(
            "gate",
            &[&gated, &attended, &gate],
            &[qh * d],
            &[],
            qh * d,
            128,
        )?;
        self.linear(&format!("{p}.wo_ud.weight"), &gated, c.hidden_size, qh * d)
    }
    fn expert_projection(
        &mut self,
        layer: usize,
        down: bool,
        x: &A::Buffer,
        routing: &A::Buffer,
    ) -> Result<A::Buffer> {
        let c = &self.config.text_config;
        let (rows, cols) = if down {
            (c.hidden_size, c.intermediate_size)
        } else {
            (2 * c.intermediate_size, c.hidden_size)
        };
        let top = c.num_experts_per_tok;
        let e = if down {
            &self.experts[layer].1
        } else {
            &self.experts[layer].0
        };
        let (w, scales, factors, format) = (
            e.weight.clone(),
            e.scales.clone(),
            e.factors.clone(),
            e.format,
        );
        let y = self.alloc(rows * top)?;
        self.run(
            "expert_matvec",
            &[&y, &w, &scales, &factors, x, routing],
            &[rows, cols, top, format, usize::from(down)],
            &[],
            rows * top * 32,
            128,
        )?;
        Ok(y)
    }
    fn moe(&mut self, x: &A::Buffer, layer: usize) -> Result<A::Buffer> {
        let c = self.config.text_config.clone();
        let logits = self.linear(
            &format!("layers.{layer}.mlp.gate.weight"),
            x,
            c.n_routed_experts,
            c.hidden_size,
        )?;
        let bias = self.w(&format!("layers.{layer}.mlp.gate.bias"));
        let routing = self.alloc(2 * c.num_experts_per_tok)?;
        self.run(
            "route",
            &[&routing, &logits, &bias],
            &[c.n_routed_experts, c.num_experts_per_tok],
            &[c.route_scale],
            1,
            1,
        )?;
        let gu = self.expert_projection(layer, false, x, &routing)?;
        let activated = self.alloc(c.intermediate_size * c.num_experts_per_tok)?;
        self.run(
            "swiglu",
            &[&activated, &gu],
            &[c.intermediate_size, c.num_experts_per_tok],
            &[c.mlp_gate_softcap, c.mlp_up_softcap],
            c.intermediate_size * c.num_experts_per_tok,
            128,
        )?;
        let down = self.expert_projection(layer, true, &activated, &routing)?;
        let y = self.alloc(c.hidden_size)?;
        self.run(
            "expert_sum",
            &[&y, &down, &routing],
            &[c.hidden_size, c.num_experts_per_tok],
            &[],
            c.hidden_size,
            128,
        )?;
        Ok(y)
    }
    fn layer(
        &mut self,
        hidden: &A::Buffer,
        blocks: &mut Vec<A::Buffer>,
        previous: Option<&[A::Buffer]>,
        visit: Visit,
    ) -> Result<A::Buffer> {
        let c = self.config.text_config.clone();
        let h = c.hidden_size;
        let prefix = format!("layers.{}", visit.layer);
        let local = c.is_local(visit.layer);
        let mut sources = blocks.clone();
        sources.push(hidden.clone());
        let mut inputs = Vec::new();
        for input in 0..3 {
            let x = if visit.layer == 0 {
                hidden.clone()
            } else {
                self.read_blocks(
                    &sources,
                    &format!("attn_res.attn_queries.{}", visit.layer - 1),
                    if local { 0 } else { input * h },
                )?
            };
            let x = if let Some(previous) = previous {
                let (bank, index) = match input {
                    0 => ("middle_cycle_cross_pass_queries", 0),
                    1 => ("middle_cycle_cross_pass_kv_queries", 0),
                    _ => ("middle_cycle_cross_pass_kv_queries", 1),
                };
                let msg = self.read_blocks(
                    previous,
                    bank,
                    ((visit.layer - c.loop_config.core_start) * 2 + index) * h,
                )?;
                self.message_gate(
                    &x,
                    &msg,
                    &format!("{prefix}.controllers.{}", ["attn", "k", "v"][input]),
                )?
            } else {
                x
            };
            let norm = if local {
                format!("{prefix}.attn_norm.weight")
            } else {
                format!("{prefix}.{}_input_norm.weight", ["q", "k", "v"][input])
            };
            inputs.push(self.norm(&x, Some(&norm), h, 1)?);
        }
        let attn = self.attention(&inputs, visit)?;
        let attn = self.conv(&attn, None, &format!("{prefix}.attn_sconv"), visit, 2, h)?;
        let residual = if visit.layer % c.n_layers_per_attn_res_block == 0 {
            blocks.push(hidden.clone());
            attn
        } else {
            self.add(hidden, &attn)?
        };
        let mut sources = blocks.clone();
        sources.push(residual.clone());
        let x = self.read_blocks(
            &sources,
            &format!("attn_res.mlp_queries.{}", visit.layer),
            0,
        )?;
        let x = if let Some(previous) = previous {
            let msg = self.read_blocks(
                previous,
                "middle_cycle_cross_pass_queries",
                ((visit.layer - c.loop_config.core_start) * 2 + 1) * h,
            )?;
            self.message_gate(&x, &msg, &format!("{prefix}.controllers.mlp"))?
        } else {
            x
        };
        let x = self.norm(&x, Some(&format!("{prefix}.mlp_norm.weight")), h, 1)?;
        let x = self.moe(&x, visit.layer)?;
        let x = self.conv(&x, None, &format!("{prefix}.mlp_sconv"), visit, 3, h)?;
        self.add(&residual, &x)
    }
    fn forward(&mut self, token: u32) -> Result<Vec<f32>> {
        let c = self.config.text_config.clone();
        let embed = self.weights["embed.weight"].clone();
        let hidden = self.alloc(c.hidden_size)?;
        self.run(
            "embedding",
            &[&hidden, &embed.buffer],
            &[c.hidden_size, token as usize, embed.dtype],
            &[],
            c.hidden_size,
            128,
        )?;
        let mut hidden = self.norm(&hidden, Some("embed_norm.weight"), c.hidden_size, 1)?;
        let mut blocks = Vec::new();
        let mut prelude = None;
        let mut previous = None;
        for visit in c.schedule() {
            if visit.layer == c.loop_config.core_start {
                if visit.pass == 0 {
                    prelude = Some((hidden.clone(), blocks.clone()));
                } else {
                    let (pre_hidden, pre_blocks) = prelude.as_ref().expect("validated schedule");
                    // Exclude the prelude and the first core boundary. Only completed core
                    // blocks plus the final partial block are messages for the next pass.
                    let mut messages = blocks[pre_blocks.len() + 1..].to_vec();
                    messages.push(hidden);
                    previous = Some(messages);
                    hidden = pre_hidden.clone();
                    blocks = pre_blocks.clone();
                }
            }
            let in_core =
                visit.layer >= c.loop_config.core_start && visit.layer < c.loop_config.core_end;
            hidden = self.layer(
                &hidden,
                &mut blocks,
                if in_core { previous.as_deref() } else { None },
                visit,
            )?;
        }
        blocks.push(hidden);
        let hidden = self.read_blocks(&blocks, "attn_res.output_query", 0)?;
        let hidden = self.norm(&hidden, Some("norm.weight"), c.hidden_size, 1)?;
        let logits = self.linear(
            "unembed.weight",
            &hidden,
            c.unpadded_vocab_size,
            c.hidden_size,
        )?;
        let capped = self.alloc(c.unpadded_vocab_size)?;
        self.run(
            "softcap",
            &[&capped, &logits],
            &[c.unpadded_vocab_size],
            &[c.final_logit_softcapping],
            c.unpadded_vocab_size,
            128,
        )?;
        let bytes = self.device.read(&capped)?;
        let logits: Vec<_> = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        ensure!(
            logits.iter().all(|x| x.is_finite()),
            "non-finite logits at position {}",
            self.position
        );
        Ok(logits)
    }
}

impl<A: Accelerator> AutoregressiveModel for Inkling<A> {
    fn context_capacity(&self) -> usize {
        self.capacity
    }
    fn position(&self) -> usize {
        self.position
    }
    fn reset(&mut self) -> Result<()> {
        self.device.synchronize()?;
        self.failed = true;
        self.caches.clear();
        let c = self.config.text_config.clone();
        for visit in c.schedule() {
            let (_, heads, dim) = c.attention_shape(visit.layer);
            let capacity = if c.is_local(visit.layer) {
                self.capacity.min(c.sliding_window_size)
            } else {
                self.capacity
            };
            let kv = heads * dim;
            let mut conv = Vec::new();
            for n in [kv, kv, c.hidden_size, c.hidden_size] {
                conv.push(self.alloc(n * c.sconv_kernel_size)?);
            }
            let keys = self.alloc(capacity * kv)?;
            let values = self.alloc(capacity * kv)?;
            self.caches.push(Cache {
                keys,
                values,
                conv,
                capacity,
            });
        }
        self.position = 0;
        self.failed = false;
        Ok(())
    }
    fn step(&mut self, token: u32) -> Result<Vec<f32>> {
        ensure!(!self.failed, "session failed; reset before reuse");
        ensure!(
            self.position < self.capacity,
            "context capacity {} exhausted",
            self.capacity
        );
        ensure!(
            (token as usize) < self.config.text_config.unpadded_vocab_size,
            "token {token} outside vocabulary"
        );
        self.failed = true;
        let result = self.forward(token)?;
        self.position += 1;
        self.failed = false;
        Ok(result)
    }
}

fn tensor<'a>(
    cp: &'a Checkpoint,
    name: &str,
    shape: &[usize],
) -> Result<safetensors::tensor::TensorView<'a>> {
    let tensor = cp.tensor(&format!("model.llm.{name}"))?;
    ensure!(
        tensor.shape() == shape,
        "{name}: expected shape {shape:?}, got {:?}",
        tensor.shape()
    );
    ensure!(
        shape
            .iter()
            .try_fold(1usize, |a, b| a.checked_mul(*b))
            .is_some_and(|n| n <= u32::MAX as usize),
        "{name}: tensor exceeds kernel indexing range"
    );
    Ok(tensor)
}
fn floats(t: &safetensors::tensor::TensorView<'_>) -> Result<Vec<f32>> {
    Ok(match t.dtype() {
        Dtype::F32 => t
            .data()
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect(),
        Dtype::BF16 => t
            .data()
            .chunks_exact(2)
            .map(|b| half::bf16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).to_f32())
            .collect(),
        Dtype::F16 => t
            .data()
            .chunks_exact(2)
            .map(|b| half::f16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).to_f32())
            .collect(),
        d => bail!("expected floating-point tensor, got {d:?}"),
    })
}
fn load_weights<A: Accelerator>(
    device: &mut A,
    cp: &Checkpoint,
    c: &TextConfig,
) -> Result<HashMap<String, Weight<A::Buffer>>> {
    let mut specs = Vec::new();
    let h = c.hidden_size;
    let mut add = |name: String, shape: Vec<usize>, convert: bool| {
        specs.push((name, shape, convert));
    };
    for name in ["embed.weight", "unembed.weight"] {
        add(name.into(), vec![c.vocab_size, h], false);
    }
    for name in ["embed_norm.weight", "norm.weight", "attn_res.output_query"] {
        add(name.into(), vec![h], true);
    }
    let core_len = c.loop_config.core_end - c.loop_config.core_start;
    for name in [
        "middle_cycle_cross_pass_queries",
        "middle_cycle_cross_pass_kv_queries",
    ] {
        add(name.into(), vec![2 * core_len, h], true);
    }
    for i in 0..c.num_hidden_layers {
        let p = format!("layers.{i}");
        let local = c.is_local(i);
        let (q, k, d) = c.attention_shape(i);
        if i > 0 {
            add(
                format!("attn_res.attn_queries.{}", i - 1),
                if local { vec![h] } else { vec![3, h] },
                true,
            );
        }
        add(format!("attn_res.mlp_queries.{i}"), vec![h], true);
        for (name, rows, cols) in [
            ("wq_du", q * d, h),
            ("wk_dv", k * d, h),
            ("wv_dv", k * d, h),
            ("wr_du", q * c.d_rel, h),
            ("wo_ud", h, q * d),
            ("gate_attn", q * d, h),
            ("attn_sink_proj", q, h),
        ] {
            add(format!("{p}.attn.{name}.weight"), vec![rows, cols], false);
        }
        add(format!("{p}.attn.q_l2_clip.scale"), vec![q, d], true);
        add(
            format!("{p}.attn.rel_l2_clip.scale"),
            vec![q, c.d_rel],
            true,
        );
        add(
            format!("{p}.attn.rel_logits_proj.proj"),
            vec![
                c.d_rel,
                if local {
                    c.sliding_window_size
                } else {
                    c.rel_extent
                },
            ],
            true,
        );
        for kind in ["k", "v"] {
            if local {
                add(
                    format!("{p}.attn.{kind}_sconv.weight"),
                    vec![k * d, 1, c.sconv_kernel_size],
                    true,
                );
            } else {
                add(
                    format!("{p}.attn.{kind}_sconv.static_w"),
                    vec![c.sconv_kernel_size, k * d],
                    true,
                );
                add(
                    format!("{p}.attn.{kind}_sconv.weight_proj"),
                    vec![k * c.sconv_kernel_size, h],
                    false,
                );
            }
        }
        for name in ["attn_sconv", "mlp_sconv"] {
            add(
                format!("{p}.{name}.weight"),
                vec![h, 1, c.sconv_kernel_size],
                true,
            );
        }
        for name in if local {
            vec!["attn_norm", "mlp_norm"]
        } else {
            vec!["q_input_norm", "k_input_norm", "v_input_norm", "mlp_norm"]
        } {
            add(format!("{p}.{name}.weight"), vec![h], true);
        }
        add(
            format!("{p}.mlp.gate.weight"),
            vec![c.n_routed_experts, h],
            false,
        );
        add(format!("{p}.mlp.gate.bias"), vec![c.n_routed_experts], true);
        if (c.loop_config.core_start..c.loop_config.core_end).contains(&i) {
            for kind in ["attn", "k", "v", "mlp"] {
                add(
                    format!("{p}.controllers.{kind}.down_weight"),
                    vec![c.loop_config.gate_rank, 2 * h],
                    false,
                );
                for name in ["forget_weight", "gain_weight"] {
                    add(
                        format!("{p}.controllers.{kind}.{name}"),
                        vec![h, c.loop_config.gate_rank],
                        false,
                    );
                }
            }
        }
    }
    let mut weights = HashMap::new();
    for (name, shape, convert) in specs {
        let t = tensor(cp, &name, &shape)?;
        let (buffer, dtype) = if convert {
            (device.upload(bytemuck::cast_slice(&floats(&t)?))?, 0)
        } else {
            let dtype = match t.dtype() {
                Dtype::F32 => 0,
                Dtype::BF16 => 1,
                Dtype::F16 => 2,
                d => bail!("{name}: unsupported dense dtype {d:?}"),
            };
            (device.upload(t.data())?, dtype)
        };
        weights.insert(name, Weight { buffer, dtype });
    }
    Ok(weights)
}

fn load_expert<A: Accelerator>(
    device: &mut A,
    cp: &Checkpoint,
    name: &str,
    experts: usize,
    rows: usize,
    cols: usize,
) -> Result<Expert<A::Buffer>> {
    let t = cp.tensor(&format!("model.llm.{name}"))?;
    let (format, block, packed) = match t.dtype() {
        Dtype::U8 => (0, 16, 2),
        Dtype::F8_E4M3 => (1, 32, 1),
        Dtype::F32 => (2, 1, 1),
        d => bail!("{name}: unsupported expert format {d:?}"),
    };
    ensure!(
        cols % block == 0,
        "{name}: input dimension must be divisible by {block}"
    );
    tensor(cp, name, &[experts, rows, cols / packed])?;
    ensure!(
        experts
            .checked_mul(rows)
            .and_then(|n| n.checked_mul(cols))
            .is_some_and(|n| n <= u32::MAX as usize),
        "expert matrix exceeds kernel indexing range"
    );
    let weight = device.upload(t.data())?;
    let scales = if format < 2 {
        let s = tensor(cp, &format!("{name}_scale"), &[experts, rows, cols / block])?;
        ensure!(
            s.dtype()
                == if format == 0 {
                    Dtype::F8_E4M3
                } else {
                    Dtype::U8
                },
            "{name}: wrong scale dtype"
        );
        device.upload(s.data())?
    } else {
        device.upload(&[0u8; 4])?
    };
    let factors = if format == 0 {
        let amax = floats(&tensor(cp, &format!("{name}_amax"), &[experts])?)?;
        let denominator = floats(&tensor(
            cp,
            &format!("{name}_outer_scale_denominator"),
            &[1],
        )?)?[0];
        ensure!(
            denominator.is_finite() && denominator > 0.,
            "{name}: invalid quantization denominator"
        );
        let factors: Vec<_> = amax.into_iter().map(|a| a / denominator).collect();
        ensure!(
            factors.iter().all(|a| a.is_finite() && *a >= 0.),
            "{name}: invalid quantization scale"
        );
        device.upload(bytemuck::cast_slice(&factors))?
    } else {
        device.upload(&[0u8; 4])?
    };
    Ok(Expert {
        weight,
        scales,
        factors,
        format,
    })
}
