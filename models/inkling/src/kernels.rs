//! The model owns shader source and its ABI. A CUDA implementation can add a source
//! variant here without changing the model's execution graph or the core crate.
use bytemuck::{Pod, Zeroable};
use rink_core::{Accelerator, Argument, KernelSource, Result};
use std::collections::HashMap;

pub const SOURCE: KernelSource<'static> = KernelSource {
    name: "inkling-2-mini-v1",
    metal: Some(include_str!("kernels.metal")),
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
            "copy_vec",
            "embedding",
            "matvec",
            "rms",
            "clip",
            "add",
            "block_read",
            "concat_norm",
            "controller",
            "convolution",
            "attn_scores",
            "attn_values",
            "gate",
            "route",
            "expert_matvec",
            "swiglu",
            "expert_sum",
            "softcap",
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
        // Parameter block is always binding 15. Unused bindings alias the output.
        while args.len() < 15 {
            args.push(Argument::Buffer(buffers[0]));
        }
        args.push(Argument::Bytes(bytemuck::bytes_of(&p)));
        // SAFETY: forwarded caller contract; argument bytes live until dispatch copies them.
        unsafe { device.dispatch(&self.entries[name], &args, threads, group) }
    }
}
