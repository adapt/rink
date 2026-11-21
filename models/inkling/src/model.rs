use crate::{
    checkpoint::Checkpoint,
    config::{Config, TextConfig, Visit},
    kernels::{Kernels, Params},
};
use anyhow::{Context, Result, bail, ensure};
use rink_core::{Accelerator, AutoregressiveModel, Buffer};
use safetensors::Dtype;
use std::{collections::HashMap, path::Path};

/// Storage precision for keys and values; attention arithmetic stays float32.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum KvCachePrecision {
    #[default]
    F16,
    F32,
}

impl KvCachePrecision {
    fn bytes(self) -> usize {
        match self {
            Self::F16 => 2,
            Self::F32 => 4,
        }
    }

    fn shader_flag(self) -> usize {
        usize::from(self == Self::F16)
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

struct ExpertGroups<B> {
    slots: B,
    tiles: B,
    max_tiles: usize,
}
type ExpertPair<B> = (Expert<B>, Expert<B>);

struct Cache<B> {
    keys: B,
    values: B,
    conv: Vec<B>,
    capacity: usize,
}

#[derive(Clone)]
struct Blocks<B> {
    buffer: B,
    count: usize,
    offset: usize,
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
    kv_cache: KvCachePrecision,
    attention_workspace: Option<A::Buffer>,
    capacity: usize,
    position: usize,
    failed: bool,
    scratch: Vec<A::Buffer>,
    scratch_cursor: usize,
    batch: usize,
}

impl<A: Accelerator> Inkling<A> {
    pub fn load(device: A, directory: &Path, capacity: usize) -> Result<Self> {
        Self::load_with_kv_cache(device, directory, capacity, KvCachePrecision::default())
    }

    pub fn load_with_kv_cache(
        mut device: A,
        directory: &Path,
        capacity: usize,
        kv_cache: KvCachePrecision,
    ) -> Result<Self> {
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
            kv_cache,
            attention_workspace: None,
            capacity,
            position: 0,
            failed: false,
            scratch: Vec::new(),
            scratch_cursor: 0,
            batch: 1,
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
    pub fn accelerator(&self) -> &A {
        &self.device
    }
    pub fn accelerator_mut(&mut self) -> &mut A {
        &mut self.device
    }

    fn alloc(&mut self, n: usize) -> Result<A::Buffer> {
        let bytes = n.checked_mul(4).context("buffer size overflow")?;
        let index = self.scratch_cursor;
        if index == self.scratch.len() {
            self.scratch.push(self.device.allocate(bytes)?);
        } else if self.scratch[index].byte_len() < bytes {
            self.scratch[index] = self.device.allocate(bytes)?;
        }
        self.scratch_cursor += 1;
        Ok(self.scratch[index].clone())
    }
    fn alloc_state(&mut self, n: usize) -> Result<A::Buffer> {
        self.device
            .allocate(n.checked_mul(4).context("buffer size overflow")?)
    }

    fn attention_workspace(&mut self, n: usize) -> Result<A::Buffer> {
        let bytes = n
            .checked_mul(4)
            .context("attention workspace size overflow")?;
        if self
            .attention_workspace
            .as_ref()
            .is_none_or(|b| b.byte_len() < bytes)
        {
            self.attention_workspace = Some(self.device.allocate(bytes)?);
        }
        // Every partition buffer is consumed by attention_finish before the next
        // layer overwrites it. The accelerator's ordered stream preserves this
        // dependency, including across loop passes, without a CPU synchronization.
        Ok(self.attention_workspace.as_ref().unwrap().clone())
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
        self.linear_offset(name, x, rows, cols, 0)
    }
    fn linear_offset(
        &mut self,
        name: &str,
        x: &A::Buffer,
        rows: usize,
        cols: usize,
        offset: usize,
    ) -> Result<A::Buffer> {
        let w = self.weights[name].clone();
        let y = self.alloc(rows * self.batch)?;
        // Keep the original scalar/packed boundary: their reduction orders differ.
        // Scheduling changes must not switch a scalar projection to packed dots.
        let vectorized = cols % 4 == 0 && rows >= 128;
        if self.batch > 1 {
            let tiled = self.batch >= 16;
            self.run(
                if tiled { "matmul_tiled" } else { "matmul" },
                &[&y, &w.buffer, x],
                &[rows, cols, w.dtype, self.batch, offset],
                &[],
                if tiled {
                    rows.div_ceil(32) * self.batch.div_ceil(16) * 256
                } else {
                    rows.div_ceil(8) * self.batch.div_ceil(8) * 32
                },
                if tiled { 256 } else { 32 },
            )?;
            return Ok(y);
        }
        self.run(
            if vectorized {
                "matvec_packed"
            } else {
                "matvec_prefetch"
            },
            &[&y, &w.buffer, x],
            &[rows, cols, w.dtype, 1, offset],
            &[],
            rows * 32,
            if rows < 128 { 32 } else { 128 },
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
        let rows = rows * self.batch;
        let y = self.alloc(width * rows)?;
        let group = if width >= 512 { 256 } else { 32 };
        self.run(
            if group == 256 { "rms_parallel" } else { "rms" },
            &[&y, x, &w],
            &[width, rows, usize::from(name.is_some())],
            &[self.config.text_config.rms_norm_eps],
            rows * group,
            group,
        )?;
        Ok(y)
    }
    fn read_blocks(
        &mut self,
        blocks: &Blocks<A::Buffer>,
        live: &A::Buffer,
        query: &str,
        offset: usize,
    ) -> Result<A::Buffer> {
        let n = self.config.text_config.hidden_size;
        let q = self.w(query);
        let y = self.alloc(n * self.batch)?;
        // Each shard repeats the small score reduction, then owns disjoint
        // output channels. This exposes parallel work without splitting sums.
        // Prompt batches already provide parallelism, so use fewer shards.
        let shards = n.div_ceil(256).min(if self.batch < 8 { 8 } else { 4 });
        self.run(
            "block_read_prefetch",
            &[&y, &blocks.buffer, live, &q],
            &[
                n,
                blocks.count + 1,
                offset,
                blocks.offset,
                self.batch,
                shards,
            ],
            &[],
            self.batch * shards * 256,
            256,
        )?;
        Ok(y)
    }
    fn message_gate(&mut self, x: &A::Buffer, msg: &A::Buffer, prefix: &str) -> Result<A::Buffer> {
        let h = self.config.text_config.hidden_size;
        let rank = self.config.text_config.loop_config.gate_rank;
        let cat = self.alloc(2 * h * self.batch)?;
        let group = if h >= 512 { 256 } else { 32 };
        self.run(
            if group == 256 {
                "concat_norm_parallel"
            } else {
                "concat_norm"
            },
            &[&cat, x, msg],
            &[h, self.batch],
            &[self.config.text_config.rms_norm_eps],
            self.batch * group,
            group,
        )?;
        let low = self.linear(&format!("{prefix}.down_weight"), &cat, rank, 2 * h)?;
        let forget = self.weights[&format!("{prefix}.forget_weight")].clone();
        let gain = self.weights[&format!("{prefix}.gain_weight")].clone();
        let y = self.alloc(h * self.batch)?;
        let vectorized = rank % 4 == 0 && h >= 128;
        let (kernel, threads, group) = if self.batch > 1 {
            (
                "controller_matmul",
                h.div_ceil(8) * self.batch.div_ceil(8) * 32,
                32,
            )
        } else {
            (
                "controller_matvec",
                if vectorized {
                    h.div_ceil(4) * 32
                } else {
                    h * 32
                },
                128,
            )
        };
        self.run(
            kernel,
            &[&y, x, msg, &low, &forget.buffer, &gain.buffer],
            &[
                h,
                rank,
                forget.dtype,
                gain.dtype,
                self.batch,
                usize::from(vectorized),
            ],
            &[],
            threads,
            group,
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
        let y = self.alloc(channels * self.batch)?;
        self.run(
            "convolution",
            &[&y, x, &w, &history, &dyn_weights],
            &[
                channels,
                k,
                self.position,
                usize::from(dynamic_input.is_some()),
                dim,
                self.batch,
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
        log_scale: bool,
    ) -> Result<A::Buffer> {
        let w = scale.map(|n| self.w(n)).unwrap_or_else(|| x.clone());
        let y = self.alloc(width * heads * self.batch)?;
        self.run(
            "clip",
            &[&y, x, &w],
            &[
                width,
                heads,
                usize::from(scale.is_some()),
                self.batch,
                self.position,
                self.config.text_config.log_scaling_n_floor,
            ],
            &[
                threshold,
                if log_scale {
                    self.config.text_config.log_scaling_alpha
                } else {
                    0.
                },
            ],
            heads * self.batch * 32,
            32,
        )?;
        Ok(y)
    }
    fn residual_conv(
        &mut self,
        x: &A::Buffer,
        routing: Option<&A::Buffer>,
        residual: Option<&A::Buffer>,
        visit: Visit,
    ) -> Result<A::Buffer> {
        let c = &self.config.text_config;
        let h = c.hidden_size;
        let taps = c.sconv_kernel_size;
        let top = routing.map_or(0, |_| c.num_experts_per_tok);
        let (kind, state) = if routing.is_some() {
            ("mlp", 3)
        } else {
            ("attn", 2)
        };
        // Large prompt chunks benefit from parallel mixture reduction and
        // elementwise kernels. Keep the fusion for decode and short chunks.
        if self.batch >= 8 {
            let input = if let Some(routing) = routing {
                let mixture = self.alloc(h * self.batch)?;
                self.run(
                    "expert_sum",
                    &[&mixture, x, routing],
                    &[h, top, self.batch],
                    &[],
                    h * self.batch,
                    128,
                )?;
                mixture
            } else {
                x.clone()
            };
            let convolved = self.conv(
                &input,
                None,
                &format!("layers.{}.{kind}_sconv", visit.layer),
                visit,
                state,
                h,
            )?;
            if let Some(residual) = residual {
                let n = h * self.batch;
                let y = self.alloc(n)?;
                self.run("add", &[&y, residual, &convolved], &[n], &[], n, 128)?;
                return Ok(y);
            }
            return Ok(convolved);
        }
        let w = self.w(&format!("layers.{}.{kind}_sconv.weight", visit.layer));
        let history = self.caches[visit.cache].conv[state].clone();
        let geometry = [
            h,
            taps,
            self.position,
            self.batch,
            top,
            usize::from(residual.is_some()),
        ];
        let y = self.alloc(h * self.batch)?;
        self.run(
            "residual_convolution",
            &[
                &y,
                x,
                &w,
                &history,
                residual.unwrap_or(x),
                routing.unwrap_or(x),
            ],
            &geometry,
            &[],
            h,
            128,
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
        let q = self.clip(
            &q,
            Some(&format!("{p}.q_l2_clip.scale")),
            d,
            qh,
            c.qk_l2_clip_threshold,
            !local,
        )?;
        let k = self.clip(&k, None, d, kh, c.qk_l2_clip_threshold, false)?;
        let v = self.norm(&v, None, d, kh)?;
        let r = self.clip(
            &r,
            Some(&format!("{p}.rel_l2_clip.scale")),
            c.d_rel,
            qh,
            c.rel_l2_clip_threshold,
            !local,
        )?;
        let cache = &self.caches[visit.cache];
        let (keys, values, capacity) = (cache.keys.clone(), cache.values.clone(), cache.capacity);
        let n = (self.position + self.batch).min(capacity);
        let extent = if local {
            c.sliding_window_size
        } else {
            c.rel_extent
        };
        let parts = n.div_ceil(128);
        let geometry = [
            qh,
            kh,
            d,
            self.position,
            capacity,
            n,
            c.d_rel,
            extent,
            parts,
            self.batch,
            self.kv_cache.shader_flag(),
        ];
        // Reserve for this batch at the configured capacity, avoiding reallocations
        // as history grows. All visits share this allocation, including local layers.
        let partials =
            self.attention_workspace(self.batch * qh * capacity.div_ceil(128) * (d + 2))?;
        let profiles = self.w(&format!("{p}.rel_logits_proj.proj"));
        self.run(
            "attention_partition",
            &[&partials, &q, &keys, &values, &k, &v, &r, &profiles],
            &geometry,
            &[],
            self.batch * qh * parts * 128,
            128,
        )?;
        let gated = self.alloc(self.batch * qh * d)?;
        self.run(
            "attention_finish",
            &[&gated, &partials, &sink, &gate],
            &geometry,
            &[],
            self.batch * qh * 32,
            32,
        )?;
        self.run(
            "cache_write",
            &[&keys, &values, &k, &v],
            &[
                kh * d,
                capacity,
                self.position,
                self.batch,
                self.kv_cache.shader_flag(),
            ],
            &[],
            self.batch * kh * d,
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
        groups: Option<&ExpertGroups<A::Buffer>>,
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
        let fused = !down && self.batch < 8;
        let output_rows = if fused { rows / 2 } else { rows };
        let caps = [c.mlp_gate_softcap, c.mlp_up_softcap];
        let y = self.alloc(output_rows * top * self.batch)?;
        if let Some(groups) = groups {
            self.run(
                "expert_matmul_tiled",
                &[&y, &w, &scales, &factors, x, &groups.slots, &groups.tiles],
                &[rows, cols, top, format, usize::from(down), self.batch],
                &caps,
                rows.div_ceil(32) * groups.max_tiles * 128,
                128,
            )?;
        } else {
            self.run(
                if fused {
                    "expert_swiglu_matvec"
                } else {
                    "expert_matvec"
                },
                &[&y, &w, &scales, &factors, x, routing],
                &[rows, cols, top, format, usize::from(down), self.batch],
                &caps,
                output_rows * top * self.batch * 32,
                128,
            )?;
        }
        if down || fused {
            return Ok(y);
        }
        let n = (rows / 2) * top * self.batch;
        let activated = self.alloc(n)?;
        self.run(
            "swiglu",
            &[&activated, &y],
            &[rows / 2, top, self.batch],
            &caps,
            n,
            128,
        )?;
        Ok(activated)
    }
    fn group_experts(&mut self, routing: &A::Buffer) -> Result<ExpertGroups<A::Buffer>> {
        let experts = self.config.text_config.n_routed_experts;
        let top = self.config.text_config.num_experts_per_tok;
        let count = self.batch * top;
        // Sum ceil(count_per_expert / 8), bounded by total slots plus at most
        // seven padding slots for each nonempty expert.
        let max_tiles = (count + 7 * experts.min(count)) / 8;
        let slots = self.alloc(experts * (self.batch + 1))?;
        let tiles = self.alloc(1 + 2 * max_tiles)?;
        self.run(
            "expert_groups",
            &[&slots, routing],
            &[experts, top, self.batch],
            &[],
            experts * 32,
            32,
        )?;
        self.run(
            "expert_tiles",
            &[&tiles, &slots],
            &[experts, self.batch],
            &[],
            32,
            32,
        )?;
        Ok(ExpertGroups {
            slots,
            tiles,
            max_tiles,
        })
    }
    fn moe(&mut self, x: &A::Buffer, residual: &A::Buffer, visit: Visit) -> Result<A::Buffer> {
        let c = self.config.text_config.clone();
        let layer = visit.layer;
        let logits = self.linear(
            &format!("layers.{layer}.mlp.gate.weight"),
            x,
            c.n_routed_experts,
            c.hidden_size,
        )?;
        let bias = self.w(&format!("layers.{layer}.mlp.gate.bias"));
        let routing = self.alloc(2 * c.num_experts_per_tok * self.batch)?;
        self.run(
            "route_parallel",
            &[&routing, &logits, &bias],
            &[c.n_routed_experts, c.num_experts_per_tok, self.batch],
            &[c.route_scale],
            self.batch * 32,
            32,
        )?;
        let groups = if self.batch >= 8 {
            Some(self.group_experts(&routing)?)
        } else {
            None
        };
        let activated = self.expert_projection(layer, false, x, &routing, groups.as_ref())?;
        let down = self.expert_projection(layer, true, &activated, &routing, groups.as_ref())?;
        self.residual_conv(&down, Some(&routing), Some(residual), visit)
    }
    fn layer(
        &mut self,
        hidden: &A::Buffer,
        blocks: &mut Blocks<A::Buffer>,
        previous: Option<&(Blocks<A::Buffer>, A::Buffer)>,
        visit: Visit,
    ) -> Result<A::Buffer> {
        let c = self.config.text_config.clone();
        let h = c.hidden_size;
        let prefix = format!("layers.{}", visit.layer);
        let local = c.is_local(visit.layer);
        let mut inputs: Vec<A::Buffer> = Vec::new();
        let shared = if visit.layer == 0 {
            Some(hidden.clone())
        } else if local {
            Some(self.read_blocks(
                blocks,
                hidden,
                &format!("attn_res.attn_queries.{}", visit.layer - 1),
                0,
            )?)
        } else {
            None
        };
        for input in 0..3 {
            if input > 0 && local && previous.is_none() {
                inputs.push(inputs[0].clone());
                continue;
            }
            let x = if let Some(shared) = &shared {
                shared.clone()
            } else {
                self.read_blocks(
                    blocks,
                    hidden,
                    &format!("attn_res.attn_queries.{}", visit.layer - 1),
                    if local { 0 } else { input * h },
                )?
            };
            let x = if let Some((previous, previous_live)) = previous {
                let (bank, index) = match input {
                    0 => ("middle_cycle_cross_pass_queries", 0),
                    1 => ("middle_cycle_cross_pass_kv_queries", 0),
                    _ => ("middle_cycle_cross_pass_kv_queries", 1),
                };
                let msg = self.read_blocks(
                    previous,
                    previous_live,
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
        let boundary = visit.layer % c.n_layers_per_attn_res_block == 0;
        let residual = self.residual_conv(
            &attn,
            None,
            if boundary { None } else { Some(hidden) },
            visit,
        )?;
        if boundary {
            self.copy(
                &blocks.buffer,
                blocks.count * h * self.batch,
                hidden,
                0,
                h * self.batch,
            )?;
            blocks.count += 1;
        }
        let x = self.read_blocks(
            blocks,
            &residual,
            &format!("attn_res.mlp_queries.{}", visit.layer),
            0,
        )?;
        let x = if let Some((previous, previous_live)) = previous {
            let msg = self.read_blocks(
                previous,
                previous_live,
                "middle_cycle_cross_pass_queries",
                ((visit.layer - c.loop_config.core_start) * 2 + 1) * h,
            )?;
            self.message_gate(&x, &msg, &format!("{prefix}.controllers.mlp"))?
        } else {
            x
        };
        let x = self.norm(&x, Some(&format!("{prefix}.mlp_norm.weight")), h, 1)?;
        self.moe(&x, &residual, visit)
    }
    fn hidden(&mut self, tokens: &[u32]) -> Result<A::Buffer> {
        self.scratch_cursor = 0;
        self.batch = tokens.len();
        let c = self.config.text_config.clone();
        let embed = self.weights["embed.weight"].clone();
        let hidden = self.alloc(c.hidden_size * self.batch)?;
        if self.batch == 1 {
            self.run(
                "embedding",
                &[&hidden, &embed.buffer],
                &[c.hidden_size, tokens[0] as usize, embed.dtype],
                &[],
                c.hidden_size,
                128,
            )?;
        } else {
            let ids = self.device.upload(bytemuck::cast_slice(tokens))?;
            self.run(
                "embedding_batch",
                &[&hidden, &embed.buffer, &ids],
                &[c.hidden_size, self.batch, embed.dtype],
                &[],
                c.hidden_size * self.batch,
                128,
            )?;
        }
        let mut hidden = self.norm(&hidden, Some("embed_norm.weight"), c.hidden_size, 1)?;
        let bank_size = (c.num_hidden_layers.div_ceil(c.n_layers_per_attn_res_block) + 1)
            * c.hidden_size
            * self.batch;
        let mut blocks = Blocks {
            buffer: self.alloc(bank_size)?,
            count: 0,
            offset: 0,
        };
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
                    let offset = pre_blocks.count + 1;
                    previous = Some((
                        Blocks {
                            buffer: blocks.buffer.clone(),
                            count: blocks.count - offset,
                            offset,
                        },
                        hidden,
                    ));
                    hidden = pre_hidden.clone();
                    blocks = Blocks {
                        buffer: self.alloc(bank_size)?,
                        count: pre_blocks.count,
                        offset: 0,
                    };
                    self.copy(
                        &blocks.buffer,
                        0,
                        &pre_blocks.buffer,
                        0,
                        pre_blocks.count * c.hidden_size * self.batch,
                    )?;
                }
            }
            let in_core =
                visit.layer >= c.loop_config.core_start && visit.layer < c.loop_config.core_end;
            hidden = self.layer(
                &hidden,
                &mut blocks,
                if in_core { previous.as_ref() } else { None },
                visit,
            )?;
        }
        let hidden = self.read_blocks(&blocks, &hidden, "attn_res.output_query", 0)?;
        self.norm(&hidden, Some("norm.weight"), c.hidden_size, 1)
    }
    fn project(&mut self, hidden: &A::Buffer) -> Result<A::Buffer> {
        let c = self.config.text_config.clone();
        let offset = (self.batch - 1) * c.hidden_size;
        self.batch = 1;
        let logits = self.linear_offset(
            "unembed.weight",
            hidden,
            c.unpadded_vocab_size,
            c.hidden_size,
            offset,
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
        Ok(capped)
    }
    fn logits(&mut self, hidden: &A::Buffer) -> Result<Vec<f32>> {
        let capped = self.project(hidden)?;
        let bytes = self
            .device
            .read_prefix(&capped, self.config.text_config.unpadded_vocab_size * 4)?;
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

    fn select(&mut self, hidden: &A::Buffer) -> Result<u32> {
        let logits = self.project(hidden)?;
        let n = self.config.text_config.unpadded_vocab_size;
        let groups = n.div_ceil(1024);
        let partial = self.alloc(groups * 3)?;
        self.run(
            "argmax_partial",
            &[&partial, &logits],
            &[n],
            &[],
            groups * 32,
            32,
        )?;
        let result = self.alloc(2)?;
        self.run(
            "argmax_finish",
            &[&result, &partial],
            &[groups],
            &[],
            32,
            32,
        )?;
        let bytes = self.device.read_prefix(&result, 8)?;
        let token = u32::from_le_bytes(bytes[..4].try_into().unwrap());
        let invalid = u32::from_le_bytes(bytes[4..].try_into().unwrap());
        ensure!(
            invalid == 0,
            "non-finite logits at position {}",
            self.position
        );
        Ok(token)
    }
    fn validate_tokens(&self, tokens: &[u32]) -> Result<()> {
        ensure!(!self.failed, "session failed; reset before reuse");
        ensure!(!tokens.is_empty(), "prompt must not be empty");
        ensure!(
            tokens.len() <= self.capacity.saturating_sub(self.position),
            "context capacity {} exhausted",
            self.capacity
        );
        ensure!(
            tokens
                .iter()
                .all(|&t| (t as usize) < self.config.text_config.unpadded_vocab_size),
            "token outside vocabulary"
        );
        Ok(())
    }
    fn prefill_hidden(&mut self, tokens: &[u32]) -> Result<A::Buffer> {
        self.validate_tokens(tokens)?;
        self.failed = true;
        let mut hidden = None;
        for (i, chunk) in tokens.chunks(32).enumerate() {
            hidden = Some(self.hidden(chunk)?);
            self.position += chunk.len();
            if (i + 1) * 32 < tokens.len() {
                self.device.synchronize()?;
            }
        }
        Ok(hidden.expect("nonempty prompt"))
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
        self.failed = true;
        self.device.synchronize()?;
        // Cache reads are bounded by the absolute sequence position. At position
        // zero every visible KV entry and convolution tap is written before use;
        // old allocations can be retained without clearing their contents.
        if !self.caches.is_empty() {
            self.position = 0;
            self.scratch_cursor = 0;
            self.failed = false;
            return Ok(());
        }
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
                conv.push(self.alloc_state(n * c.sconv_kernel_size)?);
            }
            let bytes = (capacity * kv)
                .checked_mul(self.kv_cache.bytes())
                .context("KV cache size overflow")?;
            let keys = self.device.allocate(bytes)?;
            let values = self.device.allocate(bytes)?;
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
        self.validate_tokens(&[token])?;
        self.failed = true;
        let hidden = self.hidden(&[token])?;
        let result = self.logits(&hidden)?;
        self.position += 1;
        self.failed = false;
        Ok(result)
    }
    fn step_greedy(&mut self, token: u32) -> Result<u32> {
        self.validate_tokens(&[token])?;
        self.failed = true;
        let hidden = self.hidden(&[token])?;
        let result = self.select(&hidden)?;
        self.position += 1;
        self.failed = false;
        Ok(result)
    }
    fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        let hidden = self.prefill_hidden(tokens)?;
        let result = self.logits(&hidden)?;
        self.failed = false;
        Ok(result)
    }
    fn prefill_greedy(&mut self, tokens: &[u32]) -> Result<u32> {
        let hidden = self.prefill_hidden(tokens)?;
        let result = self.select(&hidden)?;
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
