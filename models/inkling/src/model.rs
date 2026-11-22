use crate::{
    PrefillProgress,
    checkpoint::Checkpoint,
    config::{Config, TextConfig, Visit},
    image::Image,
    kernels::{Kernels, Params},
    vision::VisionEncoder,
};
use crate::{
    name::name,
    workspace::{Arena, Pool},
};
use anyhow::{Context, Result, bail, ensure};
use rink_core::{Accelerator, AutoregressiveModel, Buffer};
use safetensors::Dtype;
use std::sync::Arc;
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

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
    lookup: Option<B>,
    format: usize,
}

struct ExpertGroups<B> {
    slots: B,
    tiles: B,
    max_tiles: usize,
    tile_size: usize,
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

// The prefill workspace never exceeds the former 32-token, full-context
// reservation. Query groups reuse that storage within each layer; the rest of
// the decoder keeps its full batch even at long histories.
fn attention_elements(c: &TextConfig, capacity: usize, history: usize, batch: usize) -> usize {
    (0..c.num_hidden_layers)
        .map(|layer| {
            let (heads, _, dim) = c.attention_shape(layer);
            let length = history.min(capacity).min(if c.is_local(layer) {
                c.sliding_window_size
            } else {
                capacity
            });
            batch * heads * length.div_ceil(128) * (dim + 2)
        })
        .max()
        .expect("validated nonempty layer schedule")
}

fn prefill_batch(remaining: usize, limit: usize) -> usize {
    // Retain the original short tail: batches below eight use different expert
    // kernels. Combining that tail with a large chunk would change reductions.
    if remaining < 32 {
        return remaining;
    }
    let mut batch = limit;
    while batch > remaining {
        batch /= 2;
    }
    batch
}

fn attention_query_batch(batch: usize, elements_per_query: usize, budget: usize) -> usize {
    debug_assert!(elements_per_query <= budget);
    let mut queries = batch;
    while queries * elements_per_query > budget {
        queries = queries.div_ceil(2);
    }
    queries
}

/// Single-sequence, streaming text decoder for the original Inkling 2 Mini checkpoint.
/// Uses float32 activations and accumulation, with native packed expert storage.
pub struct Inkling<A: Accelerator> {
    device: A,
    kernels: Kernels<A>,
    config: Config,
    text: Arc<TextConfig>,
    schedule: Arc<[Visit]>,
    directory: PathBuf,
    vision: Option<VisionEncoder<A>>,
    weights: HashMap<String, Weight<A::Buffer>>,
    experts: Vec<ExpertPair<A::Buffer>>,
    caches: Vec<Cache<A::Buffer>>,
    kv_cache: KvCachePrecision,
    attention_workspace: Option<A::Buffer>,
    capacity: usize,
    position: usize,
    failed: bool,
    pool: Pool<A::Buffer>,
    scratch: Arena<A::Buffer>,
    layer_scratch: Arena<A::Buffer>,
    token_ids: Option<A::Buffer>,
    in_layer: bool,
    prefill_batch_limit: usize,
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
                    &kernels,
                    &format!("{prefix}.w13_weight"),
                    c.n_routed_experts,
                    c.intermediate_size * 2,
                    c.hidden_size,
                )?,
                load_expert(
                    &mut device,
                    &checkpoint,
                    &kernels,
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
            text: Arc::new(config.text_config.clone()),
            schedule: config.text_config.schedule().into(),
            config,
            directory: directory.to_owned(),
            vision: None,
            weights,
            experts,
            caches: Vec::new(),
            kv_cache,
            attention_workspace: None,
            capacity,
            position: 0,
            failed: false,
            pool: Pool::new(),
            scratch: Arena::new(),
            layer_scratch: Arena::new(),
            token_ids: None,
            in_layer: false,
            // About 28 GiB of Metal allocation at 256K with FP16 KV storage.
            prefill_batch_limit: 1024,
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

    /// Set the maximum prompt batch (a power of two from 32 through 4096). Attention query
    /// groups shrink with history to stay within the fixed workspace budget.
    /// Resolution, precision and context capacity are unaffected.
    pub fn set_prefill_batch_limit(&mut self, tokens: usize) -> Result<()> {
        ensure!(
            (32..=4096).contains(&tokens) && tokens.is_power_of_two(),
            "prefill batch limit must be a power of two from 32 through 4096"
        );
        self.prefill_batch_limit = tokens;
        Ok(())
    }

    pub fn prefill_batch_limit(&self) -> usize {
        self.prefill_batch_limit
    }

    fn alloc(&mut self, n: usize) -> Result<A::Buffer> {
        let bytes = n.checked_mul(4).context("buffer size overflow")?;
        let arena = if self.in_layer {
            &mut self.layer_scratch
        } else {
            &mut self.scratch
        };
        arena.alloc(&mut self.device, &mut self.pool, bytes)
    }
    fn release_workspaces(&mut self) {
        self.scratch.release(&mut self.pool);
        self.layer_scratch.release(&mut self.pool);
        if let Some(buffer) = self.attention_workspace.take() {
            self.pool.put(buffer);
        }
    }

    fn alloc_state(&mut self, n: usize) -> Result<A::Buffer> {
        let bytes = n.checked_mul(4).context("buffer size overflow")?;
        // SAFETY: convolution writes the current ring slot before reading it,
        // and reads only lags <= the absolute position (also after reset).
        unsafe { self.device.allocate_uninitialized(bytes) }
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
            if let Some(buffer) = self.attention_workspace.take() {
                self.pool.put(buffer);
            }
            self.attention_workspace = Some(self.pool.take(&mut self.device, bytes)?);
        }
        // Every partition buffer is consumed by attention_finish before the next
        // query group or layer overwrites it. The ordered stream preserves this
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
            // Four-wide staging preserves each 8x8 accumulator's original
            // operation order, including short and partially filled tiles.
            self.run(
                "matmul_packed",
                &[&y, &w.buffer, x],
                &[rows, cols, w.dtype, self.batch, offset],
                &[],
                rows.div_ceil(32) * self.batch.div_ceil(16) * 256,
                256,
            )?;
            return Ok(y);
        }
        self.run(
            match (vectorized, w.dtype, cols) {
                (false, 1, 2304) => "matvec_scalar_bf16_2304",
                (false, 1, 4608) => "matvec_scalar_bf16_4608",
                (true, 1, 2304) => "matvec_bf16_2304",
                (true, 1, 3072) => "matvec_bf16_3072",
                (true, _, _) => "matvec_packed",
                _ => "matvec_prefetch",
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
        let low = self.linear(&name!("{prefix}.down_weight"), &cat, rank, 2 * h)?;
        let forget = self.weights[&*name!("{prefix}.forget_weight")].clone();
        let gain = self.weights[&*name!("{prefix}.gain_weight")].clone();
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
                self.w(&name!("{prefix}.static_w")),
                self.linear(&name!("{prefix}.weight_proj"), input, heads * k, h)?,
            )
        } else {
            (self.w(&name!("{prefix}.weight")), x.clone())
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
        output: Option<&A::Buffer>,
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
                &name!("layers.{}.{kind}_sconv", visit.layer),
                visit,
                state,
                h,
            )?;
            if let Some(residual) = residual {
                let n = h * self.batch;
                let y = match output {
                    Some(output) => output.clone(),
                    None => self.alloc(n)?,
                };
                self.run("add", &[&y, residual, &convolved], &[n], &[], n, 128)?;
                return Ok(y);
            }
            return Ok(convolved);
        }
        let w = self.w(&name!("layers.{}.{kind}_sconv.weight", visit.layer));
        let history = self.caches[visit.cache].conv[state].clone();
        let geometry = [
            h,
            taps,
            self.position,
            self.batch,
            top,
            usize::from(residual.is_some()),
        ];
        let y = match output {
            Some(output) => output.clone(),
            None => self.alloc(h * self.batch)?,
        };
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
        let c = self.text.clone();
        let (qh, kh, d) = c.attention_shape(visit.layer);
        let local = c.is_local(visit.layer);
        let p = name!("layers.{}.attn", visit.layer);
        let q = self.linear(
            &name!("{p}.wq_du.weight"),
            &inputs[0],
            qh * d,
            c.hidden_size,
        )?;
        let k = self.linear(
            &name!("{p}.wk_dv.weight"),
            &inputs[1],
            kh * d,
            c.hidden_size,
        )?;
        let v = self.linear(
            &name!("{p}.wv_dv.weight"),
            &inputs[2],
            kh * d,
            c.hidden_size,
        )?;
        let r = self.linear(
            &name!("{p}.wr_du.weight"),
            &inputs[0],
            qh * c.d_rel,
            c.hidden_size,
        )?;
        let gate = self.linear(
            &name!("{p}.gate_attn.weight"),
            &inputs[0],
            qh * d,
            c.hidden_size,
        )?;
        let sink = self.linear(
            &name!("{p}.attn_sink_proj.weight"),
            &inputs[0],
            qh,
            c.hidden_size,
        )?;
        let k = self.conv(
            &k,
            if local { None } else { Some(&inputs[1]) },
            &name!("{p}.k_sconv"),
            visit,
            0,
            kh * d,
        )?;
        let v = self.conv(
            &v,
            if local { None } else { Some(&inputs[2]) },
            &name!("{p}.v_sconv"),
            visit,
            1,
            kh * d,
        )?;
        let q = self.clip(
            &q,
            Some(&name!("{p}.q_l2_clip.scale")),
            d,
            qh,
            c.qk_l2_clip_threshold,
            !local,
        )?;
        let k = self.clip(&k, None, d, kh, c.qk_l2_clip_threshold, false)?;
        let v = self.norm(&v, None, d, kh)?;
        let r = self.clip(
            &r,
            Some(&name!("{p}.rel_l2_clip.scale")),
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
        let query_batch = attention_query_batch(
            self.batch,
            qh * parts * (d + 2),
            attention_elements(&c, self.capacity, self.capacity, 32),
        );
        let mut geometry = [
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
            0, // Query offset within the original batch; fresh KV keeps its base.
        ];
        // Prefill reserves a fixed budget before its first chunk. Decode keeps
        // its full-capacity reservation too, so history growth never reallocates.
        let partials = self.attention_workspace(
            query_batch
                * qh
                * parts.max(if self.batch == 1 {
                    capacity.div_ceil(128)
                } else {
                    0
                })
                * (d + 2),
        )?;
        let profiles = self.w(&name!("{p}.rel_logits_proj.proj"));
        // More score groups hide latency on small decode grids. Both larger
        // decode grids and prompt batches benefit from ordered value prefetch.
        // All paths retain the same dot products, softmax and partition sums.
        let (partition_kernel, partition_threads, query_tile, partition_heads) =
            if local && d == 128 && qh == 3 * kh && self.batch >= 4 {
                ("attention_partition_heads3", 256, 1, kh)
            } else if !local && d == 128 && self.batch >= 4 && parts >= 8 {
                ("attention_partition_queries4", 256, 4, qh)
            } else if self.batch == 1 && parts <= 2 {
                ("attention_partition_wide", 512, 1, qh)
            } else {
                ("attention_partition_prefetch", 256, 1, qh)
            };
        let (finish_kernel, finish_threads) = if self.batch > 1 && (129..=2048).contains(&parts) {
            ("attention_finish_cached", 128)
        } else {
            ("attention_finish", 32)
        };
        let gated = self.alloc(self.batch * qh * d)?;
        for offset in (0..self.batch).step_by(query_batch) {
            let count = query_batch.min(self.batch - offset);
            geometry[9] = count;
            geometry[11] = offset;
            self.run(
                partition_kernel,
                &[&partials, &q, &keys, &values, &k, &v, &r, &profiles],
                &geometry,
                &[],
                count.div_ceil(query_tile) * partition_heads * parts * partition_threads,
                partition_threads,
            )?;
            self.run(
                finish_kernel,
                &[&gated, &partials, &sink, &gate],
                &geometry,
                &[],
                count * qh * finish_threads,
                finish_threads,
            )?;
        }
        // All groups must consume the original ring before any fresh KV is
        // written. Their outputs stay in the full-batch gated buffer above.
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
        self.linear(&name!("{p}.wo_ud.weight"), &gated, c.hidden_size, qh * d)
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
        let lookup = e.lookup.clone().unwrap_or_else(|| factors.clone());
        let fused = !down && self.batch < 8;
        let output_rows = if fused { rows / 2 } else { rows };
        let caps = [c.mlp_gate_softcap, c.mlp_up_softcap];
        let y = self.alloc(output_rows * top * self.batch)?;
        if let Some(groups) = groups {
            // NVFP4 tables are already resident for decode. Reuse the same
            // exact values in prefill without allocating another workspace.
            let threads = if groups.tile_size == 32 { 256 } else { 128 };
            self.run(
                if groups.tile_size == 32 {
                    "expert_matmul_m32k32"
                } else if format == 0 {
                    "expert_matmul_nvfp4"
                } else {
                    "expert_matmul_tiled"
                },
                &[
                    &y,
                    &w,
                    &scales,
                    &factors,
                    x,
                    &groups.slots,
                    &groups.tiles,
                    &lookup,
                ],
                &[rows, cols, top, format, usize::from(down), self.batch],
                &caps,
                rows.div_ceil(32) * groups.max_tiles * threads,
                threads,
            )?;
        } else {
            self.run(
                if format == 0 {
                    if fused {
                        "expert_swiglu_nvfp4"
                    } else {
                        "expert_matvec_nvfp4"
                    }
                } else if format == 1
                    && rows == 2304
                    && ((fused && cols == 2304) || (down && cols == 1152))
                {
                    if fused {
                        "expert_swiglu_mxfp8_native"
                    } else {
                        "expert_matvec_mxfp8_native"
                    }
                } else if fused {
                    "expert_swiglu_matvec"
                } else {
                    "expert_matvec"
                },
                &[&y, &w, &scales, &factors, x, routing, &lookup],
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
        // Larger tiles pay off when enough tokens reuse each expert. Keep small
        // batches on eight-slot tiles to avoid multiplying mostly padding.
        let tile_size = if self.batch >= 512 { 32 } else { 8 };
        // Sum ceil(count_per_expert / tile_size), bounded by total slots plus
        // at most tile_size - 1 padding slots for each nonempty expert.
        let max_tiles = (count + (tile_size - 1) * experts.min(count)) / tile_size;
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
            &[experts, self.batch, tile_size],
            &[],
            32,
            32,
        )?;
        Ok(ExpertGroups {
            slots,
            tiles,
            max_tiles,
            tile_size,
        })
    }
    fn moe(
        &mut self,
        x: &A::Buffer,
        residual: &A::Buffer,
        visit: Visit,
        output: &A::Buffer,
    ) -> Result<A::Buffer> {
        let c = self.text.clone();
        let layer = visit.layer;
        let logits = self.linear(
            &name!("layers.{layer}.mlp.gate.weight"),
            x,
            c.n_routed_experts,
            c.hidden_size,
        )?;
        let bias = self.w(&name!("layers.{layer}.mlp.gate.bias"));
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
        self.residual_conv(&down, Some(&routing), Some(residual), visit, Some(output))
    }
    fn layer(
        &mut self,
        hidden: &A::Buffer,
        blocks: &mut Blocks<A::Buffer>,
        previous: Option<&(Blocks<A::Buffer>, A::Buffer)>,
        visit: Visit,
        output: &A::Buffer,
    ) -> Result<A::Buffer> {
        // The caller protects the live, prelude and previous-pass output slots.
        // All other activation lifetimes end with this layer's ordered commands.
        self.layer_scratch.reset();
        self.in_layer = true;
        let result = self.layer_inner(hidden, blocks, previous, visit, output);
        self.in_layer = false;
        result
    }
    fn layer_inner(
        &mut self,
        hidden: &A::Buffer,
        blocks: &mut Blocks<A::Buffer>,
        previous: Option<&(Blocks<A::Buffer>, A::Buffer)>,
        visit: Visit,
        output: &A::Buffer,
    ) -> Result<A::Buffer> {
        let c = self.text.clone();
        let h = c.hidden_size;
        let prefix = name!("layers.{}", visit.layer);
        let local = c.is_local(visit.layer);
        let mut inputs: [A::Buffer; 3] = std::array::from_fn(|_| hidden.clone());
        let shared = if visit.layer == 0 {
            Some(hidden.clone())
        } else if local {
            Some(self.read_blocks(
                blocks,
                hidden,
                &name!("attn_res.attn_queries.{}", visit.layer - 1),
                0,
            )?)
        } else {
            None
        };
        for input in 0..3 {
            if input > 0 && local && previous.is_none() {
                inputs[input] = inputs[0].clone();
                continue;
            }
            let x = if let Some(shared) = &shared {
                shared.clone()
            } else {
                self.read_blocks(
                    blocks,
                    hidden,
                    &name!("attn_res.attn_queries.{}", visit.layer - 1),
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
                    &name!("{prefix}.controllers.{}", ["attn", "k", "v"][input]),
                )?
            } else {
                x
            };
            let norm = if local {
                name!("{prefix}.attn_norm.weight")
            } else {
                name!("{prefix}.{}_input_norm.weight", ["q", "k", "v"][input])
            };
            inputs[input] = self.norm(&x, Some(&norm), h, 1)?;
        }
        let attn = self.attention(&inputs, visit)?;
        let boundary = visit.layer % c.n_layers_per_attn_res_block == 0;
        let residual = self.residual_conv(
            &attn,
            None,
            if boundary { None } else { Some(hidden) },
            visit,
            None,
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
            &name!("attn_res.mlp_queries.{}", visit.layer),
            0,
        )?;
        let x = if let Some((previous, previous_live)) = previous {
            let msg = self.read_blocks(
                previous,
                previous_live,
                "middle_cycle_cross_pass_queries",
                ((visit.layer - c.loop_config.core_start) * 2 + 1) * h,
            )?;
            self.message_gate(&x, &msg, &name!("{prefix}.controllers.mlp"))?
        } else {
            x
        };
        let x = self.norm(&x, Some(&name!("{prefix}.mlp_norm.weight")), h, 1)?;
        self.moe(&x, &residual, visit, output)
    }
    fn hidden(&mut self, tokens: &[u32]) -> Result<A::Buffer> {
        self.hidden_with_image(tokens, None)
    }
    fn hidden_with_image(
        &mut self,
        tokens: &[u32],
        image: Option<(&A::Buffer, usize, usize, usize)>,
    ) -> Result<A::Buffer> {
        self.scratch.reset();
        self.batch = tokens.len();
        let c = self.text.clone();
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
            let bytes = std::mem::size_of_val(tokens);
            if self.token_ids.as_ref().is_none_or(|b| b.byte_len() < bytes) {
                // Host writes happen only after all previous reads complete.
                self.device.synchronize()?;
                self.token_ids = None;
                // SAFETY: write below initializes every token the kernel reads.
                self.token_ids = Some(unsafe { self.device.allocate_uninitialized(bytes)? });
            }
            let ids = self.token_ids.as_ref().unwrap().clone();
            self.device.write(&ids, 0, bytemuck::cast_slice(tokens))?;
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
        // The reference normalizes text embeddings before replacing image
        // placeholders. Projected vision features must bypass embed_norm.
        if let Some((features, src, dst, count)) = image {
            self.copy(
                &hidden,
                dst * c.hidden_size,
                features,
                src * c.hidden_size,
                count * c.hidden_size,
            )?;
        }
        let bank_size = (c.num_hidden_layers.div_ceil(c.n_layers_per_attn_res_block) + 1)
            * c.hidden_size
            * self.batch;
        let mut blocks = Blocks {
            buffer: self.alloc(bank_size)?,
            count: 0,
            offset: 0,
        };
        // At most the current, prelude, and previous-pass values are pinned.
        // A fourth output can always be overwritten after its last encoded read.
        let n = c.hidden_size * self.batch;
        let outputs = [
            self.alloc(n)?,
            self.alloc(n)?,
            self.alloc(n)?,
            self.alloc(n)?,
        ];
        let mut hidden_slot = None;
        let mut prelude_slot = None;
        let mut previous_slot = None;
        let mut prelude = None;
        let mut previous = None;
        for visit in self.schedule.clone().iter().copied() {
            if visit.layer == c.loop_config.core_start {
                if visit.pass == 0 {
                    prelude_slot = hidden_slot;
                    prelude = Some((hidden.clone(), blocks.clone()));
                } else {
                    let (pre_hidden, pre_blocks) = prelude.as_ref().expect("validated schedule");
                    // Exclude the prelude and the first core boundary. Only completed core
                    // blocks plus the final partial block are messages for the next pass.
                    let offset = pre_blocks.count + 1;
                    previous_slot = hidden_slot;
                    previous = Some((
                        Blocks {
                            buffer: blocks.buffer.clone(),
                            count: blocks.count - offset,
                            offset,
                        },
                        hidden,
                    ));
                    hidden = pre_hidden.clone();
                    hidden_slot = prelude_slot;
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
            let slot = (0..4)
                .find(|&slot| ![hidden_slot, prelude_slot, previous_slot].contains(&Some(slot)))
                .unwrap();
            hidden = self.layer(
                &hidden,
                &mut blocks,
                if in_core { previous.as_ref() } else { None },
                visit,
                &outputs[slot],
            )?;
            hidden_slot = Some(slot);
        }
        let hidden = self.read_blocks(&blocks, &hidden, "attn_res.output_query", 0)?;
        self.norm(&hidden, Some("norm.weight"), c.hidden_size, 1)
    }
    fn project(&mut self, hidden: &A::Buffer) -> Result<A::Buffer> {
        let c = self.text.clone();
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
        let mut logits = vec![0f32; self.config.text_config.unpadded_vocab_size];
        self.device
            .read_into(&capped, bytemuck::cast_slice_mut(&mut logits))?;
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
        let mut bytes = [0u8; 8];
        self.device.read_into(&result, &mut bytes)?;
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
    fn prefill_hidden(
        &mut self,
        tokens: &[u32],
        progress: &mut dyn FnMut(PrefillProgress),
    ) -> Result<A::Buffer> {
        self.validate_tokens(tokens)?;
        self.failed = true;
        self.attention_workspace(attention_elements(
            &self.config.text_config,
            self.capacity,
            self.capacity,
            tokens.len().min(32),
        ))?;
        progress(PrefillProgress::Prompt {
            completed_tokens: 0,
            total_tokens: tokens.len(),
        });
        let mut hidden = None;
        let mut completed = 0;
        while completed < tokens.len() {
            let batch = prefill_batch(tokens.len() - completed, self.prefill_batch_limit);
            let chunk = &tokens[completed..completed + batch];
            hidden = Some(self.hidden(chunk)?);
            self.position += chunk.len();
            completed += chunk.len();
            if completed < tokens.len() {
                self.device.synchronize()?;
                progress(PrefillProgress::Prompt {
                    completed_tokens: completed,
                    total_tokens: tokens.len(),
                });
            }
        }
        Ok(hidden.expect("nonempty prompt"))
    }

    fn prefill_image_hidden(
        &mut self,
        tokens: &[u32],
        image: &Image,
        progress: &mut dyn FnMut(PrefillProgress),
    ) -> Result<A::Buffer> {
        self.validate_tokens(tokens)?;
        self.config
            .vision_config
            .as_ref()
            .context("checkpoint has no vision encoder")?;
        let start = tokens
            .iter()
            .position(|&t| t == self.config.image_token_id)
            .context("prompt has no image placeholders")?;
        let end = start
            .checked_add(image.token_count())
            .context("image token count overflow")?;
        ensure!(
            end <= tokens.len()
                && tokens[start..end]
                    .iter()
                    .all(|&t| t == self.config.image_token_id)
                && !tokens[end..].contains(&self.config.image_token_id),
            "prompt must contain one contiguous image placeholder per image tile"
        );
        self.failed = true;
        self.device.synchronize()?;
        self.release_workspaces();
        let config = self.config.vision_config.as_ref().unwrap();
        if self.vision.is_none() {
            progress(PrefillProgress::LoadingImageEncoder);
            let checkpoint = Checkpoint::open(&self.directory)?;
            self.vision = Some(VisionEncoder::load(&mut self.device, &checkpoint, config)?);
            progress(PrefillProgress::ImageEncoderLoaded);
        }
        let (features, feature_storage) = self.vision.as_ref().unwrap().encode_with_pool(
            &mut self.device,
            &self.kernels,
            image,
            &mut self.pool,
            |completed_layers, total_layers| {
                progress(PrefillProgress::ImageEncoding {
                    completed_layers,
                    total_layers,
                });
            },
        )?;
        progress(PrefillProgress::ImageEncoded);
        self.attention_workspace(attention_elements(
            &self.config.text_config,
            self.capacity,
            self.capacity,
            tokens.len().min(32),
        ))?;
        progress(PrefillProgress::Prompt {
            completed_tokens: 0,
            total_tokens: tokens.len(),
        });
        let mut hidden = None;
        let mut offset = 0;
        while offset < tokens.len() {
            let batch = prefill_batch(tokens.len() - offset, self.prefill_batch_limit);
            let chunk = &tokens[offset..offset + batch];
            let lo = start.max(offset);
            let hi = end.min(offset + chunk.len());
            let replacement = (lo < hi).then(|| (&features, lo - start, lo - offset, hi - lo));
            hidden = Some(self.hidden_with_image(chunk, replacement)?);
            self.position += chunk.len();
            offset += chunk.len();
            if offset < tokens.len() {
                self.device.synchronize()?;
                progress(PrefillProgress::Prompt {
                    completed_tokens: offset,
                    total_tokens: tokens.len(),
                });
            }
        }
        drop(features);
        self.pool.put(feature_storage);
        Ok(hidden.expect("nonempty prompt"))
    }

    /// Append a mixed image/text prompt. Image placeholders must be contiguous
    /// and match `image.token_count()`. Previous turns remain in the KV cache.
    pub fn prefill_image(&mut self, tokens: &[u32], image: &Image) -> Result<Vec<f32>> {
        let hidden = self.prefill_image_hidden(tokens, image, &mut |_| {})?;
        let result = self.logits(&hidden)?;
        self.failed = false;
        Ok(result)
    }

    pub fn prefill_image_greedy(&mut self, tokens: &[u32], image: &Image) -> Result<u32> {
        self.prefill_image_greedy_with_progress(tokens, image, |_| {})
    }

    /// Append an image prompt and report loading, encoding and prefill progress.
    pub fn prefill_image_greedy_with_progress(
        &mut self,
        tokens: &[u32],
        image: &Image,
        mut progress: impl FnMut(PrefillProgress),
    ) -> Result<u32> {
        let hidden = self.prefill_image_hidden(tokens, image, &mut progress)?;
        let result = self.select(&hidden)?;
        self.failed = false;
        progress(PrefillProgress::Complete);
        Ok(result)
    }

    /// Append a text prompt and report completed prefill chunks.
    pub fn prefill_greedy_with_progress(
        &mut self,
        tokens: &[u32],
        mut progress: impl FnMut(PrefillProgress),
    ) -> Result<u32> {
        let hidden = self.prefill_hidden(tokens, &mut progress)?;
        let result = self.select(&hidden)?;
        self.failed = false;
        progress(PrefillProgress::Complete);
        Ok(result)
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
            self.scratch.reset();
            self.failed = false;
            return Ok(());
        }
        let c = self.text.clone();
        for visit in self.schedule.clone().iter().copied() {
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
            // SAFETY: attention reads only previous positions committed by
            // cache_write; current positions come from the fresh K/V tensors.
            let keys = unsafe { self.device.allocate_uninitialized(bytes)? };
            let values = unsafe { self.device.allocate_uninitialized(bytes)? };
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
        let hidden = self.prefill_hidden(tokens, &mut |_| {})?;
        let result = self.logits(&hidden)?;
        self.failed = false;
        Ok(result)
    }
    fn prefill_greedy(&mut self, tokens: &[u32]) -> Result<u32> {
        self.prefill_greedy_with_progress(tokens, |_| {})
    }
}

fn tensor<'a>(
    cp: &'a Checkpoint,
    name: &str,
    shape: &[usize],
) -> Result<safetensors::tensor::TensorView<'a>> {
    let tensor = cp.tensor(&name!("model.llm.{name}"))?;
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
pub(crate) fn floats(t: &safetensors::tensor::TensorView<'_>) -> Result<Vec<f32>> {
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
    kernels: &Kernels<A>,
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
    let lookup = if format == 0 {
        // SAFETY: nvfp4_decode_table writes every table element before lookup.
        let table = unsafe { device.allocate_uninitialized(experts * 4096 * 4)? };
        // SAFETY: 4096 entries cover every FP4 code and FP8 scale per expert.
        unsafe {
            kernels.run(
                device,
                "nvfp4_decode_table",
                &[&table, &factors],
                Params::new(&[experts], &[]),
                experts * 4096,
                128,
            )?;
        }
        device.synchronize()?;
        Some(table)
    } else {
        None
    };
    Ok(Expert {
        lookup,
        weight,
        scales,
        factors,
        format,
    })
}

#[cfg(test)]
mod workspace_tests {
    use super::*;

    #[test]
    fn full_batches_and_query_groups_fit_one_workspace_through_256k() {
        let config: Config =
            serde_json::from_str(include_str!("../tests/fixtures/official-config.json")).unwrap();
        let c = &config.text_config;
        for capacity in [128, 4096, 65536, 131072, 262144] {
            let budget = attention_elements(c, capacity, capacity, 32);
            for limit in [32, 64, 128, 256, 512, 1024, 2048, 4096] {
                for start in [0, 1, capacity / 2 - 1, capacity - 33] {
                    let mut position = start;
                    while position < capacity {
                        let n = prefill_batch(capacity - position, limit);
                        assert!(n > 0 && n <= limit && n <= capacity - position);
                        if capacity - position >= limit {
                            assert_eq!(n, limit, "history must not shrink the decoder batch");
                        }
                        assert!(n < 32 || n % 32 == 0);
                        for layer in 0..c.num_hidden_layers {
                            let (heads, _, dim) = c.attention_shape(layer);
                            let length = (position + n).min(capacity).min(if c.is_local(layer) {
                                c.sliding_window_size
                            } else {
                                capacity
                            });
                            let per_query = heads * length.div_ceil(128) * (dim + 2);
                            let group = attention_query_batch(n, per_query, budget);
                            assert!(group > 0 && group <= n);
                            assert!(group * per_query <= budget);
                        }
                        position += n;
                    }
                }
            }
        }
        // Only attention groups shrink. Keep the original short-tail boundary.
        let budget = attention_elements(c, 262144, 262144, 32);
        assert_eq!(budget * 4, 780 * 1024 * 1024);
        for (history, expected) in [(65536, 128), (131072, 64), (262144, 32)] {
            let per_query = attention_elements(c, 262144, history, 1);
            assert_eq!(attention_query_batch(128, per_query, budget), expected);
        }
        assert_eq!(prefill_batch(10342, 128), 128);
        assert_eq!(prefill_batch(38, 128), 32);
        assert_eq!(prefill_batch(6, 128), 6);
    }
}
