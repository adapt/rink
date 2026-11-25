//! Bonsai owns the shader ABI. No dependency on another model's implementation.
use bytemuck::{Pod, Zeroable};
use rink_core::{Accelerator, Argument, KernelSource, Result};
use std::collections::BTreeMap;

pub const SOURCE: KernelSource<'static> = KernelSource {
    name: "bonsai-2.5-v1",
    metal: Some(concat!(
        include_str!("kernels.metal"),
        "\n",
        include_str!("prefill.metal")
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
            *dst = u32::try_from(src).expect("validated Bonsai dimension");
        }
        p.f[..floats.len()].copy_from_slice(floats);
        p
    }
}
pub struct Kernels<A: Accelerator> {
    entries: BTreeMap<&'static str, A::Kernel>,
}
impl<A: Accelerator> Kernels<A> {
    pub fn compile(device: &mut A) -> Result<Self> {
        let mut entries = BTreeMap::new();
        for name in [
            "embedding",
            "embedding_batch",
            "hadamard",
            "hadamard1024",
            "packed_mv",
            "packed_mv_symmetric",
            "packed_mm32",
            "packed_mm64",
            "packed_mm32_symmetric",
            "packed_mm64_symmetric",
            "mlp_mv",
            "mlp_mm",
            "mlp_mm32",
            "dense_mv",
            "dense_mm",
            "rms",
            "rms_small",
            "rope",
            "cache_write",
            "attention_part",
            "attention_finish",
            "conv",
            "conv_batch",
            "conv_save",
            "delta_gates",
            "delta",
            "delta128",
            "swiglu",
            "add",
            "zeros",
            "greedy_part",
            "greedy_finish",
        ] {
            entries.insert(name, device.compile(&SOURCE, name)?);
        }
        Ok(Self { entries })
    }
    /// # Safety
    /// Dimensions, buffers, and dispatch geometry must satisfy the shader ABI.
    pub unsafe fn run(
        &self,
        device: &mut A,
        name: &str,
        buffers: &[&A::Buffer],
        p: Params,
        threads: usize,
        group: usize,
    ) -> Result<()> {
        assert!(buffers.len() <= 15);
        let mut args = std::array::from_fn::<_, 16, _>(|_| Argument::Skip);
        for (arg, b) in args.iter_mut().zip(buffers) {
            *arg = Argument::Buffer(*b);
        }
        args[15] = Argument::Bytes(bytemuck::bytes_of(&p));
        // SAFETY: the caller supplies the validated model-owned ABI.
        unsafe { device.dispatch(&self.entries[name], &args, threads, group) }
    }
}
