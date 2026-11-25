//! One cached sequence. Dense activations follow the released FP16 runtime;
//! recurrent state and all reductions use FP32. Only text weights are uploaded.
use crate::{
    config::{Config, TextConfig},
    kernels::{Kernels, Params},
};
use anyhow::{Context, Result, ensure};
use half::{bf16, f16};
use rink_core::{Accelerator, AutoregressiveModel, Buffer, checkpoint::Checkpoint};
use safetensors::{Dtype, tensor::TensorView};
use std::{collections::BTreeMap, path::Path};

struct Packed<B> {
    weight: B,
    scales: B,
    biases: B,
    signs: Option<B>,
    block: usize,
    rotation_id: usize,
    symmetric: bool,
    rows: usize,
    cols: usize,
}
struct Dense<B> {
    weight: B,
    rows: usize,
    cols: usize,
}
enum Attention<B> {
    Full {
        q: Packed<B>,
        k: Packed<B>,
        v: Packed<B>,
        o: Packed<B>,
        qnorm: B,
        knorm: B,
        keys: B,
        values: B,
    },
    Linear {
        qkv: Packed<B>,
        z: Packed<B>,
        a: Dense<B>,
        b: Dense<B>,
        o: Packed<B>,
        conv: B,
        norm: B,
        alog: B,
        dt: B,
        history: B,
        state: B,
    },
}
struct Layer<B> {
    input_norm: B,
    post_norm: B,
    attention: Attention<B>,
    gate: Packed<B>,
    up: Packed<B>,
    down: Packed<B>,
}
const ATTENTION_QUERIES: usize = 8;
fn attention_parts(capacity: usize) -> usize {
    capacity.div_ceil(256).max(capacity.min(4096).div_ceil(32))
}

pub const DEFAULT_PREFILL_BATCH: usize = 512;

struct Scratch<B> {
    ids: B,
    x: B,
    norm: B,
    rotation: B,
    qgate: B,
    q: B,
    k: B,
    v: B,
    qkv: B,
    conv: B,
    z: B,
    a: B,
    b: B,
    att: B,
    normalized: B,
    residual: B,
    gate: B,
    up: B,
    activation: B,
    logits: B,
    delta_gates: B,
    choices: B,
    token: B,
    partials: B,
}

impl<B: Buffer> Scratch<B> {
    fn allocate<A: Accelerator<Buffer = B>>(
        device: &mut A,
        t: &TextConfig,
        capacity: usize,
        batch: usize,
    ) -> Result<Self> {
        let partials = batch.min(ATTENTION_QUERIES)
            * t.num_attention_heads
            * attention_parts(capacity)
            * (t.head_dim + 2);
        ensure!(
            partials <= u32::MAX as usize,
            "attention workspace exceeds shader index range"
        );
        let h = t.hidden_size;
        let att = (t.num_attention_heads * t.head_dim).max(t.linear_values());
        let q = (t.num_attention_heads * t.head_dim).max(t.linear_keys());
        let k = (t.num_key_value_heads * t.head_dim).max(t.linear_keys());
        let qkv = t.linear_keys() * 2 + t.linear_values();
        let mut half = |n| device.allocate(n * batch * 2);
        Ok(Self {
            ids: half(2)?,
            x: half(h)?,
            norm: half(h)?,
            rotation: half(h.max(t.intermediate_size).max(att))?,
            qgate: half(t.num_attention_heads * t.head_dim * 2)?,
            q: half(q)?,
            k: half(k)?,
            v: half(t.num_key_value_heads * t.head_dim)?,
            qkv: half(qkv)?,
            conv: half(qkv)?,
            z: half(t.linear_values())?,
            a: half(t.linear_num_value_heads)?,
            b: half(t.linear_num_value_heads)?,
            att: half(att)?,
            normalized: half(t.linear_values())?,
            residual: half(h)?,
            gate: half(t.intermediate_size)?,
            up: half(t.intermediate_size)?,
            activation: half(t.intermediate_size)?,
            delta_gates: device.allocate(batch * t.linear_num_value_heads * 8)?,
            logits: device.allocate(t.vocab_size * 2)?,
            choices: device.allocate(t.vocab_size.div_ceil(1024) * 8)?,
            token: device.allocate(4)?,
            partials: device.allocate(
                batch.min(ATTENTION_QUERIES)
                    * t.num_attention_heads
                    * attention_parts(capacity)
                    * (t.head_dim + 2)
                    * 4,
            )?,
        })
    }
}

pub struct Bonsai<A: Accelerator> {
    device: A,
    kernels: Kernels<A>,
    config: Config,
    embedding: Packed<A::Buffer>,
    head: Packed<A::Buffer>,
    norm: A::Buffer,
    layers: Vec<Layer<A::Buffer>>,
    scratch: Scratch<A::Buffer>,
    capacity: usize,
    prefill_batch_limit: usize,
    position: usize,
    failed: bool,
}

fn tensor<'a>(cp: &'a Checkpoint, name: &str, shape: &[usize]) -> Result<TensorView<'a>> {
    let t = cp.tensor(&format!("language_model.{name}"))?;
    ensure!(
        t.shape() == shape,
        "invalid shape for {name}: {:?}, expected {shape:?}",
        t.shape()
    );
    Ok(t)
}
fn half_values(t: &TensorView<'_>) -> Result<Vec<u16>> {
    let convert = |value: f32| -> Result<u16> {
        let half = f16::from_f32(value);
        ensure!(
            value.is_finite() && half.is_finite(),
            "non-finite or overflowing FP16 weights"
        );
        Ok(half.to_bits())
    };
    match t.dtype() {
        Dtype::F32 => t
            .data()
            .chunks_exact(4)
            .map(|b| convert(f32::from_le_bytes(b.try_into().unwrap())))
            .collect(),
        Dtype::F16 => t
            .data()
            .chunks_exact(2)
            .map(|b| convert(f16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).to_f32()))
            .collect(),
        Dtype::BF16 => t
            .data()
            .chunks_exact(2)
            .map(|b| convert(bf16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).to_f32()))
            .collect(),
        dtype => anyhow::bail!("unsupported floating tensor dtype {dtype:?}"),
    }
}

fn valid_signs(t: &TensorView<'_>) -> bool {
    // Validate before narrowing: values near +/-1 must not round into valid signs.
    match t.dtype() {
        Dtype::F32 => t.data().chunks_exact(4).all(|b| {
            let v = f32::from_le_bytes(b.try_into().unwrap());
            v == 1.0 || v == -1.0
        }),
        Dtype::F16 | Dtype::BF16 => {
            let one = if t.dtype() == Dtype::F16 {
                0x3c00
            } else {
                0x3f80
            };
            t.data().chunks_exact(2).all(|b| {
                let v = u16::from_le_bytes(b.try_into().unwrap());
                v == one || v == (one | 0x8000)
            })
        }
        _ => false,
    }
}
fn symmetric_affine(scales: &TensorView<'_>, biases: &TensorView<'_>) -> bool {
    scales.dtype() == Dtype::F16
        && biases.dtype() == Dtype::F16
        && scales.shape() == biases.shape()
        && scales
            .data()
            .chunks_exact(2)
            .zip(biases.data().chunks_exact(2))
            .all(|(s, b)| {
                let scale = f16::from_bits(u16::from_le_bytes(s.try_into().unwrap())).to_f32();
                let bias = f16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).to_f32();
                scale.is_finite() && scale.abs() <= 32752. && bias == -scale
            })
}

fn upload_half<A: Accelerator>(
    device: &mut A,
    cp: &Checkpoint,
    name: &str,
    shape: &[usize],
) -> Result<A::Buffer> {
    let t = tensor(cp, name, shape)?;
    if t.dtype() == Dtype::F16 {
        ensure!(
            t.data()
                .chunks_exact(2)
                .all(|b| f16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).is_finite()),
            "non-finite FP16 weights: {name}"
        );
        return device.upload(t.data());
    }
    let values = half_values(&t)?;
    device.upload(bytemuck::cast_slice(&values))
}

impl<A: Accelerator> Bonsai<A> {
    pub fn load(device: A, directory: impl AsRef<Path>, capacity: usize) -> Result<Self> {
        Self::load_with_prefill_batch(device, directory, capacity, DEFAULT_PREFILL_BATCH)
    }
    pub fn load_with_prefill_batch(
        mut device: A,
        directory: impl AsRef<Path>,
        capacity: usize,
        batch: usize,
    ) -> Result<Self> {
        ensure!(
            (1..=4096).contains(&batch) && batch.is_power_of_two(),
            "prefill batch must be a power of two from 1 through 4096"
        );
        let directory = directory.as_ref();
        let config = Config::load(directory.join("config.json"))?;
        let t = &config.text_config;
        t.cache_bytes(capacity)?;
        ensure!(
            capacity
                .checked_mul(t.num_key_value_heads * t.head_dim)
                .is_some_and(|n| n <= u32::MAX as usize),
            "KV cache exceeds shader index range"
        );
        let cp = Checkpoint::open(directory)?;
        let mut packed = BTreeMap::new();
        let mut rotations: BTreeMap<(usize, Vec<u16>), (usize, A::Buffer)> = BTreeMap::new();
        let shapes = config.projections();
        for m in &config.modules {
            let (rows, cols) = shapes[&m.path];
            ensure!(
                rows.checked_mul(cols)
                    .is_some_and(|n| n <= u32::MAX as usize),
                "matrix exceeds shader index range"
            );
            let w = tensor(&cp, &format!("{}.weight", m.path), &[rows, cols / 16])?;
            ensure!(
                w.dtype() == Dtype::U32,
                "packed weights must be U32: {}",
                m.path
            );
            let symmetric = symmetric_affine(
                &tensor(&cp, &format!("{}.scales", m.path), &[rows, cols / 128])?,
                &tensor(&cp, &format!("{}.biases", m.path), &[rows, cols / 128])?,
            );
            let weight = device.upload(w.data())?;
            let scales = upload_half(
                &mut device,
                &cp,
                &format!("{}.scales", m.path),
                &[rows, cols / 128],
            )?;
            let biases = upload_half(
                &mut device,
                &cp,
                &format!("{}.biases", m.path),
                &[rows, cols / 128],
            )?;
            let mut rotation_id = 0;
            let signs = if m.block != 0 {
                let signs = tensor(&cp, &format!("{}.signs", m.path), &[cols])?;
                ensure!(valid_signs(&signs), "invalid Hadamard signs: {}", m.path);
                let v = half_values(&signs)?;
                let next_id = rotations.len() + 1;
                let (id, buffer) = if let Some(existing) = rotations.get(&(m.block, v.clone())) {
                    existing.clone()
                } else {
                    let buffer = device.upload(bytemuck::cast_slice(&v))?;
                    rotations.insert((m.block, v), (next_id, buffer.clone()));
                    (next_id, buffer)
                };
                rotation_id = id;
                Some(buffer)
            } else {
                ensure!(
                    !cp.contains(&format!("language_model.{}.signs", m.path)),
                    "unexpected signs without Hadamard transform"
                );
                None
            };
            packed.insert(
                m.path.clone(),
                Packed {
                    weight,
                    scales,
                    biases,
                    signs,
                    block: m.block,
                    rotation_id,
                    symmetric,
                    rows,
                    cols,
                },
            );
        }
        let mut take = |name: &str| {
            packed
                .remove(name)
                .with_context(|| format!("missing packed projection {name}"))
        };
        let embedding = take("model.embed_tokens")?;
        let head = take("lm_head")?;
        let mut layers = Vec::new();
        for i in 0..t.num_hidden_layers {
            let prefix = format!("model.layers.{i}");
            let input_norm = upload_half(
                &mut device,
                &cp,
                &format!("{prefix}.input_layernorm.weight"),
                &[t.hidden_size],
            )?;
            let post_norm = upload_half(
                &mut device,
                &cp,
                &format!("{prefix}.post_attention_layernorm.weight"),
                &[t.hidden_size],
            )?;
            let attention = if t.is_full(i) {
                let p = format!("{prefix}.self_attn");
                let bytes = capacity * t.num_key_value_heads * t.head_dim * 2;
                // SAFETY: only the prefix written by cache_write is read by attention.
                let keys = unsafe { device.allocate_uninitialized(bytes)? };
                // SAFETY: as above; reset discards the valid prefix without reading it.
                let values = unsafe { device.allocate_uninitialized(bytes)? };
                Attention::Full {
                    q: take(&format!("{p}.q_proj"))?,
                    k: take(&format!("{p}.k_proj"))?,
                    v: take(&format!("{p}.v_proj"))?,
                    o: take(&format!("{p}.o_proj"))?,
                    qnorm: upload_half(
                        &mut device,
                        &cp,
                        &format!("{p}.q_norm.weight"),
                        &[t.head_dim],
                    )?,
                    knorm: upload_half(
                        &mut device,
                        &cp,
                        &format!("{p}.k_norm.weight"),
                        &[t.head_dim],
                    )?,
                    keys,
                    values,
                }
            } else {
                let p = format!("{prefix}.linear_attn");
                let qkv = t.linear_keys() * 2 + t.linear_values();
                let hv = t.linear_num_value_heads;
                let a = Dense {
                    weight: upload_half(
                        &mut device,
                        &cp,
                        &format!("{p}.in_proj_a.weight"),
                        &[hv, t.hidden_size],
                    )?,
                    rows: hv,
                    cols: t.hidden_size,
                };
                let b = Dense {
                    weight: upload_half(
                        &mut device,
                        &cp,
                        &format!("{p}.in_proj_b.weight"),
                        &[hv, t.hidden_size],
                    )?,
                    rows: hv,
                    cols: t.hidden_size,
                };
                Attention::Linear {
                    qkv: take(&format!("{p}.in_proj_qkv"))?,
                    z: take(&format!("{p}.in_proj_z"))?,
                    o: take(&format!("{p}.out_proj"))?,
                    a,
                    b,
                    conv: upload_half(
                        &mut device,
                        &cp,
                        &format!("{p}.conv1d.weight"),
                        &[qkv, t.linear_conv_kernel_dim, 1],
                    )?,
                    norm: upload_half(
                        &mut device,
                        &cp,
                        &format!("{p}.norm.weight"),
                        &[t.linear_value_head_dim],
                    )?,
                    alog: upload_half(&mut device, &cp, &format!("{p}.A_log"), &[hv])?,
                    dt: upload_half(&mut device, &cp, &format!("{p}.dt_bias"), &[hv])?,
                    history: device.allocate((qkv * (t.linear_conv_kernel_dim - 1) * 2).max(4))?,
                    state: device.allocate(t.linear_values() * t.linear_key_head_dim * 4)?,
                }
            };
            layers.push(Layer {
                input_norm,
                post_norm,
                attention,
                gate: take(&format!("{prefix}.mlp.gate_proj"))?,
                up: take(&format!("{prefix}.mlp.up_proj"))?,
                down: take(&format!("{prefix}.mlp.down_proj"))?,
            });
        }
        let norm = upload_half(&mut device, &cp, "model.norm.weight", &[t.hidden_size])?;
        drop(cp);
        let prefill_batch_limit = capacity.min(batch);
        let scratch = Scratch::allocate(&mut device, t, capacity, prefill_batch_limit)?;
        let kernels = Kernels::compile(&mut device)?;
        Ok(Self {
            device,
            kernels,
            config,
            embedding,
            head,
            norm,
            layers,
            scratch,
            capacity,
            prefill_batch_limit,
            position: 0,
            failed: false,
        })
    }
    pub fn config(&self) -> &Config {
        &self.config
    }
    pub fn accelerator(&self) -> &A {
        &self.device
    }
    pub fn accelerator_mut(&mut self) -> &mut A {
        &mut self.device
    }

    pub fn prefill_batch_limit(&self) -> usize {
        self.prefill_batch_limit
    }

    pub fn set_prefill_batch_limit(&mut self, limit: usize) -> Result<()> {
        ensure!(!self.failed, "Bonsai session failed; reset before reuse");
        ensure!(
            (1..=4096).contains(&limit) && limit.is_power_of_two(),
            "prefill batch must be a power of two from 1 through 4096"
        );
        let limit = limit.min(self.capacity);
        if limit != self.prefill_batch_limit {
            let scratch = Scratch::allocate(
                &mut self.device,
                &self.config.text_config,
                self.capacity,
                limit,
            )?;
            self.scratch = scratch;
            self.prefill_batch_limit = limit;
        }
        Ok(())
    }

    fn forward(&mut self, tokens: &[u32], logits: bool) -> Result<()> {
        ensure!(!self.failed, "Bonsai session failed; reset before reuse");
        let batch = tokens.len();
        ensure!(
            batch > 0 && batch <= self.prefill_batch_limit,
            "invalid forward batch"
        );
        ensure!(
            tokens
                .iter()
                .all(|&token| (token as usize) < self.config.text_config.vocab_size),
            "token out of vocabulary"
        );
        ensure!(
            batch <= self.capacity.saturating_sub(self.position),
            "context is full"
        );
        self.failed = true;
        if batch > 1 {
            self.device
                .write(&self.scratch.ids, 0, bytemuck::cast_slice(tokens))?;
        }
        let t = &self.config.text_config;
        let s = &self.scratch;
        let mut r = Runner {
            device: &mut self.device,
            kernels: &self.kernels,
            scratch: s,
            config: t,
            batch,
        };
        let e = &self.embedding;
        let embedding_out = if e.block > 0 { &s.norm } else { &s.x };
        if batch == 1 {
            r.run(
                "embedding",
                &[&e.weight, &e.scales, &e.biases, embedding_out],
                &[e.cols, tokens[0] as usize],
                &[],
                e.cols,
                256,
            )?;
        } else {
            r.run(
                "embedding_batch",
                &[&e.weight, &e.scales, &e.biases, embedding_out, &s.ids],
                &[e.cols, batch],
                &[],
                batch * e.cols,
                256,
            )?;
        }
        if let Some(signs) = &e.signs {
            r.run(
                if e.block == 1024 {
                    "hadamard1024"
                } else {
                    "hadamard"
                },
                &[&s.norm, signs, &s.x],
                &[e.block, 1, e.cols],
                &[],
                batch * e.cols / e.block * 256,
                256,
            )?;
        }
        for layer in &self.layers {
            r.norm(
                &s.x,
                &layer.input_norm,
                &s.norm,
                t.hidden_size,
                1,
                t.hidden_size,
                0,
                t.hidden_size,
                1.0,
                true,
            )?;
            match &layer.attention {
                Attention::Full {
                    q,
                    k,
                    v,
                    o,
                    qnorm,
                    knorm,
                    keys,
                    values,
                } => {
                    r.packed_group(&[(q, &s.qgate), (k, &s.k), (v, &s.v)], &s.norm)?;
                    r.norm(
                        &s.qgate,
                        qnorm,
                        &s.q,
                        t.head_dim,
                        t.num_attention_heads,
                        2 * t.head_dim,
                        0,
                        2 * t.num_attention_heads * t.head_dim,
                        1.0,
                        true,
                    )?;
                    // rms permits in-place operation: each group's reduction finishes before writes.
                    r.norm(
                        &s.k,
                        knorm,
                        &s.k,
                        t.head_dim,
                        t.num_key_value_heads,
                        t.head_dim,
                        0,
                        t.num_key_value_heads * t.head_dim,
                        1.0,
                        true,
                    )?;
                    for (x, heads) in [(&s.q, t.num_attention_heads), (&s.k, t.num_key_value_heads)]
                    {
                        r.run(
                            "rope",
                            &[x],
                            &[t.head_dim, t.rotary_dim(), heads, self.position, batch],
                            &[t.rope_parameters.rope_theta],
                            batch * heads * t.rotary_dim() / 2,
                            256,
                        )?;
                    }
                    let n = t.num_key_value_heads * t.head_dim;
                    r.run(
                        "cache_write",
                        &[&s.k, &s.v, keys, values],
                        &[n, self.position, batch],
                        &[],
                        batch * n,
                        256,
                    )?;
                    let partition = if self.position + batch <= 4096 {
                        32
                    } else {
                        256
                    };
                    let max_parts = (self.position + batch).div_ceil(partition);
                    let available = self.prefill_batch_limit.min(ATTENTION_QUERIES)
                        * attention_parts(self.capacity);
                    let query_limit = (available / max_parts).max(1).min(batch);
                    for query in (0..batch).step_by(query_limit) {
                        let count = (batch - query).min(query_limit);
                        let parts = (self.position + query + count).div_ceil(partition);
                        r.run(
                            "attention_part",
                            &[&s.q, keys, values, &s.partials],
                            &[
                                parts,
                                t.head_dim,
                                t.num_attention_heads,
                                t.num_key_value_heads,
                                self.position + 1,
                                query,
                                count,
                                partition,
                            ],
                            &[(t.head_dim as f32).sqrt().recip()],
                            count * parts * t.num_attention_heads * 128,
                            128,
                        )?;
                        r.run(
                            "attention_finish",
                            &[&s.partials, &s.qgate, &s.att],
                            &[parts, t.head_dim, t.num_attention_heads, query, count],
                            &[],
                            count * t.num_attention_heads * t.head_dim,
                            256,
                        )?;
                    }
                    r.packed(o, &s.att, &s.residual)?;
                }
                Attention::Linear {
                    qkv,
                    z,
                    a,
                    b,
                    o,
                    conv,
                    norm,
                    alog,
                    dt,
                    history,
                    state,
                } => {
                    r.packed_group(&[(qkv, &s.qkv), (z, &s.z)], &s.norm)?;
                    r.dense(a, &s.norm, &s.a)?;
                    r.dense(b, &s.norm, &s.b)?;
                    let n = t.linear_keys() * 2 + t.linear_values();
                    r.run(
                        if batch == 1 { "conv" } else { "conv_batch" },
                        &[&s.qkv, conv, history, &s.conv],
                        &[n, t.linear_conv_kernel_dim, batch],
                        &[],
                        batch * n,
                        256,
                    )?;
                    if batch > 1 {
                        r.run(
                            "conv_save",
                            &[&s.qkv, history],
                            &[n, t.linear_conv_kernel_dim, batch],
                            &[],
                            n,
                            256,
                        )?;
                    }
                    r.run(
                        "delta_gates",
                        &[&s.a, &s.b, alog, dt, &s.delta_gates],
                        &[t.linear_num_value_heads, batch],
                        &[],
                        t.linear_num_value_heads * batch,
                        256,
                    )?;
                    let dk = t.linear_key_head_dim;
                    let hk = t.linear_num_key_heads;
                    r.norm(
                        &s.conv,
                        norm,
                        &s.q,
                        dk,
                        hk,
                        dk,
                        0,
                        n,
                        1.0 / dk as f32,
                        false,
                    )?;
                    r.norm(
                        &s.conv,
                        norm,
                        &s.k,
                        dk,
                        hk,
                        dk,
                        t.linear_keys(),
                        n,
                        1.0 / (dk as f32).sqrt(),
                        false,
                    )?;
                    r.run(
                        "delta",
                        &[&s.q, &s.k, &s.conv, &s.delta_gates, state, &s.att],
                        &[
                            dk,
                            t.linear_value_head_dim,
                            t.linear_num_value_heads,
                            hk,
                            batch,
                        ],
                        &[],
                        t.linear_values() * 32,
                        128,
                    )?;
                    r.norm(
                        &s.att,
                        norm,
                        &s.normalized,
                        t.linear_value_head_dim,
                        t.linear_num_value_heads,
                        t.linear_value_head_dim,
                        0,
                        t.linear_values(),
                        1.0,
                        true,
                    )?;
                    r.run(
                        "swiglu",
                        &[&s.z, &s.normalized, &s.att],
                        &[batch * t.linear_values()],
                        &[],
                        batch * t.linear_values(),
                        256,
                    )?;
                    r.packed(o, &s.att, &s.residual)?;
                }
            }
            r.run(
                "add",
                &[&s.x, &s.residual],
                &[batch * t.hidden_size],
                &[],
                batch * t.hidden_size,
                256,
            )?;
            r.norm(
                &s.x,
                &layer.post_norm,
                &s.norm,
                t.hidden_size,
                1,
                t.hidden_size,
                0,
                t.hidden_size,
                1.0,
                true,
            )?;
            r.packed_group(&[(&layer.gate, &s.gate), (&layer.up, &s.up)], &s.norm)?;
            r.run(
                "swiglu",
                &[&s.gate, &s.up, &s.activation],
                &[batch * t.intermediate_size],
                &[],
                batch * t.intermediate_size,
                256,
            )?;
            r.packed(&layer.down, &s.activation, &s.residual)?;
            r.run(
                "add",
                &[&s.x, &s.residual],
                &[batch * t.hidden_size],
                &[],
                batch * t.hidden_size,
                256,
            )?;
        }
        if logits {
            r.batch = 1;
            let last =
                s.x.view((batch - 1) * t.hidden_size * 2, t.hidden_size * 2)?;
            r.norm(
                &last,
                &self.norm,
                &s.norm,
                t.hidden_size,
                1,
                t.hidden_size,
                0,
                t.hidden_size,
                1.0,
                true,
            )?;
            r.packed(&self.head, &s.norm, &s.logits)?;
        }
        self.position += batch;
        self.failed = false;
        Ok(())
    }
    fn greedy_token(&mut self) -> Result<u32> {
        self.failed = true;
        let vocab = self.config.text_config.vocab_size;
        let parts = vocab.div_ceil(1024);
        // SAFETY: the first pass writes one choice per 1024 logits. The final
        // pass reads those choices and writes the single allocated token.
        unsafe {
            self.kernels.run(
                &mut self.device,
                "greedy_part",
                &[&self.scratch.logits, &self.scratch.choices],
                Params::new(&[vocab], &[]),
                parts * 256,
                256,
            )?;
            self.kernels.run(
                &mut self.device,
                "greedy_finish",
                &[&self.scratch.choices, &self.scratch.token],
                Params::new(&[parts], &[]),
                256,
                256,
            )?;
        }
        let mut bytes = [0u8; 4];
        self.device.read_into(&self.scratch.token, &mut bytes)?;
        let token = u32::from_le_bytes(bytes);
        ensure!(
            (token as usize) < vocab,
            "Bonsai produced non-finite logits"
        );
        self.failed = false;
        Ok(token)
    }
    fn logits(&mut self) -> Result<Vec<f32>> {
        self.failed = true;
        let bytes = self.device.read(&self.scratch.logits)?;
        let logits: Vec<_> = bytes
            .chunks_exact(2)
            .map(|b| f16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).to_f32())
            .collect();
        ensure!(
            logits.iter().all(|x| x.is_finite()),
            "Bonsai produced non-finite logits"
        );
        self.failed = false;
        Ok(logits)
    }
}

struct Runner<'a, A: Accelerator> {
    device: &'a mut A,
    kernels: &'a Kernels<A>,
    scratch: &'a Scratch<A::Buffer>,
    config: &'a TextConfig,
    batch: usize,
}
impl<A: Accelerator> Runner<'_, A> {
    #[allow(clippy::too_many_arguments)]
    fn run(
        &mut self,
        name: &str,
        buffers: &[&A::Buffer],
        ints: &[usize],
        floats: &[f32],
        threads: usize,
        group: usize,
    ) -> Result<()> {
        // SAFETY: private forward graph, validated weight shapes/configuration and
        // scratch/cache extents. Each kernel's write/read prefix is described above.
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
    fn packed(&mut self, m: &Packed<A::Buffer>, x: &A::Buffer, out: &A::Buffer) -> Result<()> {
        self.packed_group(&[(m, out)], x)
    }
    fn packed_group(
        &mut self,
        projections: &[(&Packed<A::Buffer>, &A::Buffer)],
        x: &A::Buffer,
    ) -> Result<()> {
        let mut rotation_id = usize::MAX;
        for &(m, out) in projections {
            if let Some(signs) = &m.signs {
                if m.rotation_id != rotation_id {
                    self.run(
                        if m.block == 1024 {
                            "hadamard1024"
                        } else {
                            "hadamard"
                        },
                        &[x, signs, &self.scratch.rotation],
                        &[m.block, 0, m.cols],
                        &[],
                        self.batch * m.cols / m.block * 256,
                        256,
                    )?;
                    rotation_id = m.rotation_id;
                }
            }
            self.multiply(
                m,
                if m.signs.is_some() {
                    &self.scratch.rotation
                } else {
                    x
                },
                out,
            )?;
        }
        Ok(())
    }
    fn multiply(
        &mut self,
        m: &Packed<A::Buffer>,
        input: &A::Buffer,
        out: &A::Buffer,
    ) -> Result<()> {
        let tile = if self.batch <= 32 { 32 } else { 64 };
        self.run(
            if self.batch == 1 {
                if m.symmetric {
                    "packed_mv_symmetric"
                } else {
                    "packed_mv"
                }
            } else if tile == 32 {
                "packed_mm32"
            } else {
                "packed_mm64"
            },
            &[&m.weight, &m.scales, &m.biases, input, out],
            &[m.cols, m.rows, self.batch],
            &[],
            if self.batch == 1 {
                m.rows.div_ceil(if m.symmetric { 2 } else { 1 }) * 32
            } else {
                m.rows.div_ceil(64) * self.batch.div_ceil(tile) * 128
            },
            128,
        )
    }
    fn dense(&mut self, m: &Dense<A::Buffer>, x: &A::Buffer, out: &A::Buffer) -> Result<()> {
        self.run(
            if self.batch == 1 {
                "dense_mv"
            } else {
                "dense_mm"
            },
            &[&m.weight, x, out],
            &[m.cols, m.rows, self.batch],
            &[],
            if self.batch == 1 {
                m.rows * 256
            } else {
                m.rows.div_ceil(64) * self.batch.div_ceil(64) * 128
            },
            if self.batch == 1 { 256 } else { 128 },
        )
    }
    #[allow(clippy::too_many_arguments)]
    fn norm(
        &mut self,
        x: &A::Buffer,
        w: &A::Buffer,
        out: &A::Buffer,
        width: usize,
        heads: usize,
        stride: usize,
        offset: usize,
        token_stride: usize,
        scale: f32,
        weighted: bool,
    ) -> Result<()> {
        self.run(
            if width <= 256 { "rms_small" } else { "rms" },
            &[x, w, out],
            &[
                width,
                heads,
                stride,
                usize::from(weighted),
                offset,
                self.batch,
                token_stride,
            ],
            &[
                if weighted {
                    self.config.rms_norm_eps
                } else {
                    1e-6
                },
                scale,
            ],
            self.batch * heads * if width <= 256 { 32 } else { 256 },
            if width <= 256 { 128 } else { 256 },
        )
    }
}

impl<A: Accelerator> AutoregressiveModel for Bonsai<A> {
    fn context_capacity(&self) -> usize {
        self.capacity
    }
    fn position(&self) -> usize {
        self.position
    }
    fn reset(&mut self) -> Result<()> {
        self.failed = true;
        for l in &self.layers {
            if let Attention::Linear { history, state, .. } = &l.attention {
                for b in [history, state] {
                    // SAFETY: both buffers have a length divisible by four; zeros writes each word once.
                    unsafe {
                        self.kernels.run(
                            &mut self.device,
                            "zeros",
                            &[b],
                            Params::new(&[b.byte_len() / 4], &[]),
                            b.byte_len() / 4,
                            256,
                        )?;
                    }
                }
            }
        }
        self.device.synchronize()?;
        self.position = 0;
        self.failed = false;
        Ok(())
    }
    fn step(&mut self, token: u32) -> Result<Vec<f32>> {
        self.forward(&[token], true)?;
        self.logits()
    }
    fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        self.prefill_with_progress(tokens, |_| {})
    }
    fn step_greedy(&mut self, token: u32) -> Result<u32> {
        self.forward(&[token], true)?;
        self.greedy_token()
    }
    fn prefill_greedy(&mut self, tokens: &[u32]) -> Result<u32> {
        self.prefill_greedy_with_progress(tokens, |_| {})
    }
}

impl<A: Accelerator> Bonsai<A> {
    pub fn prefill_with_progress(
        &mut self,
        tokens: &[u32],
        mut progress: impl FnMut(usize),
    ) -> Result<Vec<f32>> {
        self.prefill_forward(tokens, &mut progress)?;
        let logits = self.logits()?;
        progress(tokens.len());
        Ok(logits)
    }
    pub fn prefill_greedy_with_progress(
        &mut self,
        tokens: &[u32],
        mut progress: impl FnMut(usize),
    ) -> Result<u32> {
        self.prefill_forward(tokens, &mut progress)?;
        let token = self.greedy_token()?;
        progress(tokens.len());
        Ok(token)
    }
    fn prefill_forward(&mut self, tokens: &[u32], progress: &mut impl FnMut(usize)) -> Result<()> {
        ensure!(!self.failed, "Bonsai session failed; reset before reuse");
        ensure!(
            !tokens.is_empty() && tokens.len() <= self.capacity.saturating_sub(self.position),
            "empty prompt or context capacity exceeded"
        );
        ensure!(
            tokens
                .iter()
                .all(|&v| (v as usize) < self.config.text_config.vocab_size),
            "token out of vocabulary"
        );
        let mut completed = 0;
        for chunk in tokens.chunks(self.prefill_batch_limit) {
            completed += chunk.len();
            self.forward(chunk, completed == tokens.len())?;
            if completed < tokens.len() {
                self.failed = true;
                self.device.synchronize()?;
                self.failed = false;
                progress(completed);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn symmetric_fast_path_requires_exact_representable_fp16_weights() {
        for (scale, bias, want) in [
            (0.01f32, -0.01f32, true),
            (-0.01, 0.01, true),
            (0.01, -0.02, false),
            (32752., -32752., true),
            (32768., -32768., false),
            (f32::NAN, 0., false),
            (f32::INFINITY, f32::NEG_INFINITY, false),
        ] {
            let s = f16::from_f32(scale).to_bits().to_le_bytes();
            let b = f16::from_f32(bias).to_bits().to_le_bytes();
            let scales = TensorView::new(Dtype::F16, vec![1], &s).unwrap();
            let biases = TensorView::new(Dtype::F16, vec![1], &b).unwrap();
            assert_eq!(symmetric_affine(&scales, &biases), want);
        }
        let s = 1.0f32.to_le_bytes();
        let b = (-1.0f32).to_le_bytes();
        assert!(!symmetric_affine(
            &TensorView::new(Dtype::F32, vec![1], &s).unwrap(),
            &TensorView::new(Dtype::F32, vec![1], &b).unwrap()
        ));
    }

    #[test]
    fn sign_validation_precedes_fp16_rounding() {
        for value in [1.0f32, -1.0, 1.0001, -0.9999, 0.0, f32::NAN] {
            let bytes = value.to_le_bytes();
            let t = TensorView::new(Dtype::F32, vec![1], &bytes).unwrap();
            assert_eq!(valid_signs(&t), value == 1.0 || value == -1.0);
        }
    }
}
