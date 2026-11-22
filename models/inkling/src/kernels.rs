//! The model owns shader source and its ABI. A CUDA implementation can add a source
//! variant here without changing the model's execution graph or the core crate.
use bytemuck::{Pod, Zeroable};
use rink_core::{Accelerator, Argument, KernelSource, Result};
use std::collections::HashMap;

pub const SOURCE: KernelSource<'static> = KernelSource {
    name: "inkling-2-mini-v1",
    metal: Some(concat!(
        include_str!("kernels.metal"),
        "\n",
        include_str!("vision.metal")
    )),
    cuda: None,
    subgroup_size: Some(32),
};

#[repr(C)]
#[derive(Clone, Copy, Default, Pod, Zeroable)]
pub struct Params {
    pub u: [u32; 16],
    pub f: [f32; 16],
}
impl Params {
    pub fn new(ints: &[usize], floats: &[f32]) -> Self {
        let mut p = Self::default();
        for (dst, &src) in p.u.iter_mut().zip(ints) {
            *dst = u32::try_from(src).expect("validated kernel dimension");
        }
        p.f[..floats.len()].copy_from_slice(floats);
        p
    }
}

pub struct Kernels<A: Accelerator> {
    entries: HashMap<&'static str, A::Kernel>,
}
impl<A: Accelerator> Kernels<A> {
    pub fn compile(device: &mut A) -> Result<Self> {
        let mut entries = HashMap::new();
        for name in [
            "vision_positions",
            "vision_norm",
            "vision_bias",
            "vision_residual",
            "vision_rope",
            "vision_attention",
            "vision_attention_64x16",
            "vision_attention_64x32",
            "nvfp4_decode_table",
            "expert_matvec_nvfp4",
            "expert_swiglu_nvfp4",
            "expert_matvec_mxfp8_native",
            "expert_swiglu_mxfp8_native",
            "matvec_scalar_bf16_2304",
            "matvec_scalar_bf16_4608",
            "matvec_bf16_2304",
            "matvec_bf16_3072",
            "copy_vec",
            "embedding",
            "embedding_batch",
            "matvec",
            "matvec_prefetch",
            "matvec4",
            "matvec_packed",
            "matmul",
            "matmul_tiled",
            "matmul_packed",
            "cache_write",
            "attention_partition",
            "attention_partition_wide",
            "attention_partition_prefetch",
            "attention_partition_queries4",
            "attention_finish",
            "attention_finish_cached",
            "rms",
            "rms_parallel",
            "clip",
            "add",
            "block_read",
            "block_read_parallel",
            "block_read_prefetch",
            "concat_norm",
            "concat_norm_parallel",
            "controller",
            "controller_matvec",
            "controller_matmul",
            "convolution",
            "residual_convolution",
            "attn_scores",
            "attn_values",
            "gate",
            "route",
            "route_parallel",
            "expert_matvec",
            "expert_swiglu_matvec",
            "expert_groups",
            "expert_tiles",
            "expert_matmul",
            "expert_matmul_tiled",
            "expert_matmul_nvfp4",
            "swiglu",
            "expert_sum",
            "softcap",
            "argmax_partial",
            "argmax_finish",
        ] {
            entries.insert(name, device.compile(&SOURCE, name)?);
        }
        Ok(Self { entries })
    }
    /// # Safety
    /// Buffers, dimensions and dispatch geometry must match the documented shader ABI.
    pub unsafe fn run(
        &self,
        device: &mut A,
        name: &str,
        buffers: &[&A::Buffer],
        p: Params,
        threads: usize,
        group: usize,
    ) -> Result<()> {
        let mut args: Vec<_> = buffers.iter().map(|b| Argument::Buffer(*b)).collect();
        // Parameter block is always binding 15. Unused slots need no resource binding.
        while args.len() < 15 {
            args.push(Argument::Skip);
        }
        args.push(Argument::Bytes(bytemuck::bytes_of(&p)));
        // SAFETY: forwarded caller contract; argument bytes live until dispatch copies them.
        unsafe { device.dispatch(&self.entries[name], &args, threads, group) }
    }
}
