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
struct Scratch<B> {
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
    partials: B,
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
    pub fn load(mut device: A, directory: impl AsRef<Path>, capacity: usize) -> Result<Self> {
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
            let signs = if m.block != 0 {
                let signs = tensor(&cp, &format!("{}.signs", m.path), &[cols])?;
                ensure!(valid_signs(&signs), "invalid Hadamard signs: {}", m.path);
                let v = half_values(&signs)?;
                Some(device.upload(bytemuck::cast_slice(&v))?)
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
        let h = t.hidden_size;
        let att = (t.num_attention_heads * t.head_dim).max(t.linear_values());
        let q = (t.num_attention_heads * t.head_dim).max(t.linear_keys());
        let k = (t.num_key_value_heads * t.head_dim).max(t.linear_keys());
        let qkv = t.linear_keys() * 2 + t.linear_values();
        let mut half = |n| device.allocate(n * 2);
        let scratch = Scratch {
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
            logits: half(t.vocab_size)?,
            partials: device
                .allocate(t.num_attention_heads * capacity.div_ceil(256) * (t.head_dim + 2) * 4)?,
        };
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

    fn forward(&mut self, token: u32, logits: bool) -> Result<()> {
        ensure!(!self.failed, "Bonsai session failed; reset before reuse");
        ensure!(
            (token as usize) < self.config.text_config.vocab_size,
            "token out of vocabulary"
        );
        ensure!(self.position < self.capacity, "context is full");
        self.failed = true;
        let t = &self.config.text_config;
        let s = &self.scratch;
        let mut r = Runner {
            device: &mut self.device,
            kernels: &self.kernels,
            scratch: s,
            config: t,
        };
        let e = &self.embedding;
        r.run(
            "embedding",
            &[
                &e.weight,
                &e.scales,
                &e.biases,
                if e.block > 0 { &s.norm } else { &s.x },
            ],
            &[e.cols, token as usize],
            &[],
            e.cols,
            256,
        )?;
        if let Some(signs) = &e.signs {
            r.run(
                "hadamard",
                &[&s.norm, signs, &s.x],
                &[e.block, 1],
                &[],
                e.cols / e.block * 256,
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
                    r.packed(q, &s.norm, &s.qgate)?;
                    r.packed(k, &s.norm, &s.k)?;
                    r.packed(v, &s.norm, &s.v)?;
                    r.norm(
                        &s.qgate,
                        qnorm,
                        &s.q,
                        t.head_dim,
                        t.num_attention_heads,
                        2 * t.head_dim,
                        0,
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
                        1.0,
                        true,
                    )?;
                    for (x, heads) in [(&s.q, t.num_attention_heads), (&s.k, t.num_key_value_heads)]
                    {
                        r.run(
                            "rope",
                            &[x],
                            &[t.head_dim, t.rotary_dim(), heads, self.position],
                            &[t.rope_parameters.rope_theta],
                            heads * t.rotary_dim() / 2,
                            256,
                        )?;
                    }
                    let n = t.num_key_value_heads * t.head_dim;
                    r.run(
                        "cache_write",
                        &[&s.k, &s.v, keys, values],
                        &[n, self.position],
                        &[],
                        n,
                        256,
                    )?;
                    let parts = (self.position + 1).div_ceil(256);
                    r.run(
                        "attention_part",
                        &[&s.q, keys, values, &s.partials],
                        &[
                            parts,
                            t.head_dim,
                            t.num_attention_heads,
                            t.num_key_value_heads,
                            self.position + 1,
                        ],
                        &[(t.head_dim as f32).sqrt().recip()],
                        parts * t.num_attention_heads * 32,
                        128,
                    )?;
                    r.run(
                        "attention_finish",
                        &[&s.partials, &s.qgate, &s.att],
                        &[parts, t.head_dim, t.num_attention_heads],
                        &[],
                        t.num_attention_heads * t.head_dim,
                        256,
                    )?;
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
                    r.packed(qkv, &s.norm, &s.qkv)?;
                    r.packed(z, &s.norm, &s.z)?;
                    r.dense(a, &s.norm, &s.a)?;
                    r.dense(b, &s.norm, &s.b)?;
                    let n = t.linear_keys() * 2 + t.linear_values();
                    r.run(
                        "conv",
                        &[&s.qkv, conv, history, &s.conv],
                        &[n, t.linear_conv_kernel_dim],
                        &[],
                        n,
                        256,
                    )?;
                    let dk = t.linear_key_head_dim;
                    let hk = t.linear_num_key_heads;
                    r.norm(&s.conv, norm, &s.q, dk, hk, dk, 0, 1.0 / dk as f32, false)?;
                    r.norm(
                        &s.conv,
                        norm,
                        &s.k,
                        dk,
                        hk,
                        dk,
                        t.linear_keys(),
                        1.0 / (dk as f32).sqrt(),
                        false,
                    )?;
                    r.run(
                        "delta",
                        &[&s.q, &s.k, &s.conv, &s.a, &s.b, alog, dt, state, &s.att],
                        &[dk, t.linear_value_head_dim, t.linear_num_value_heads, hk],
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
                        1.0,
                        true,
                    )?;
                    r.run(
                        "swiglu",
                        &[&s.z, &s.normalized, &s.att],
                        &[t.linear_values()],
                        &[],
                        t.linear_values(),
                        256,
                    )?;
                    r.packed(o, &s.att, &s.residual)?;
                }
            }
            r.run(
                "add",
                &[&s.x, &s.residual],
                &[t.hidden_size],
                &[],
                t.hidden_size,
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
                1.0,
                true,
            )?;
            r.packed(&layer.gate, &s.norm, &s.gate)?;
            r.packed(&layer.up, &s.norm, &s.up)?;
            r.run(
                "swiglu",
                &[&s.gate, &s.up, &s.activation],
                &[t.intermediate_size],
                &[],
                t.intermediate_size,
                256,
            )?;
            r.packed(&layer.down, &s.activation, &s.residual)?;
            r.run(
                "add",
                &[&s.x, &s.residual],
                &[t.hidden_size],
                &[],
                t.hidden_size,
                256,
            )?;
        }
        if logits {
            r.norm(
                &s.x,
                &self.norm,
                &s.norm,
                t.hidden_size,
                1,
                t.hidden_size,
                0,
                1.0,
                true,
            )?;
            r.packed(&self.head, &s.norm, &s.logits)?;
        }
        self.position += 1;
        self.failed = false;
        Ok(())
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
        let input = if let Some(signs) = &m.signs {
            self.run(
                "hadamard",
                &[x, signs, &self.scratch.rotation],
                &[m.block, 0],
                &[],
                m.cols / m.block * 256,
                256,
            )?;
            &self.scratch.rotation
        } else {
            x
        };
        self.run(
            "packed_mv",
            &[&m.weight, &m.scales, &m.biases, input, out],
            &[m.cols, m.rows],
            &[],
            m.rows * 32,
            128,
        )
    }
    fn dense(&mut self, m: &Dense<A::Buffer>, x: &A::Buffer, out: &A::Buffer) -> Result<()> {
        self.run(
            "dense_mv",
            &[&m.weight, x, out],
            &[m.cols, m.rows],
            &[],
            m.rows * 32,
            128,
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
        scale: f32,
        weighted: bool,
    ) -> Result<()> {
        self.run(
            "rms",
            &[x, w, out],
            &[width, heads, stride, usize::from(weighted), offset],
            &[
                if weighted {
                    self.config.rms_norm_eps
                } else {
                    1e-6
                },
                scale,
            ],
            heads * 256,
            256,
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
        self.forward(token, true)?;
        self.logits()
    }
    fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        self.prefill_with_progress(tokens, |_| {})
    }
}

impl<A: Accelerator> Bonsai<A> {
    pub fn prefill_with_progress(
        &mut self,
        tokens: &[u32],
        mut progress: impl FnMut(usize),
    ) -> Result<Vec<f32>> {
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
        for (i, &token) in tokens.iter().enumerate() {
            self.forward(token, i + 1 == tokens.len())?;
            if (i + 1) % 32 == 0 {
                self.failed = true;
                self.device.synchronize()?;
                self.failed = false;
                progress(i + 1);
            }
        }
        let logits = self.logits()?;
        progress(tokens.len());
        Ok(logits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sign_validation_precedes_fp16_rounding() {
        for value in [1.0f32, -1.0, 1.0001, -0.9999, 0.0, f32::NAN] {
            let bytes = value.to_le_bytes();
            let t = TensorView::new(Dtype::F32, vec![1], &bytes).unwrap();
            assert_eq!(valid_signs(&t), value == 1.0 || value == -1.0);
        }
    }
}
