//! Hardware-independent buffer, kernel and autoregressive model contracts.
//! Model crates own their kernels; accelerator crates own compilation and execution.
pub use anyhow::{Error, Result, bail, ensure};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AcceleratorKind {
    Metal,
    Cuda,
}

pub trait Buffer: Clone {
    fn byte_len(&self) -> usize;
    /// A bounded view sharing the original allocation. Offsets must be 256-byte aligned.
    fn view(&self, offset: usize, bytes: usize) -> Result<Self>;
}

/// Backend source variants of a model-owned kernel library.
pub struct KernelSource<'a> {
    pub name: &'a str,
    pub metal: Option<&'a str>,
    pub cuda: Option<&'a str>,
    /// Required SIMD/subgroup width, if the shader uses lane-level collectives.
    pub subgroup_size: Option<usize>,
}

pub enum Argument<'a, B> {
    Buffer(&'a B),
    /// Leave an unused argument slot untouched.
    Skip,
    Bytes(&'a [u8]),
}

/// A device and ordered command stream. Buffers remain valid through synchronization.
///
/// # Safety
/// Implementations must retain all resources referenced by pending dispatches, enforce
/// device ownership, and synchronize before reading or releasing pending GPU work.
pub unsafe trait Accelerator {
    type Buffer: Buffer;
    type Kernel;
    fn kind(&self) -> AcceleratorKind;
    fn name(&self) -> &str;
    fn allocate(&mut self, bytes: usize) -> Result<Self::Buffer>;
    fn upload(&mut self, bytes: &[u8]) -> Result<Self::Buffer>;
    /// Allocate storage that will be overwritten before it is read.
    ///
    /// # Safety
    /// Every byte read by the CPU or a kernel must first be initialized. This
    /// applies to views and padding as well as the logical tensor elements.
    unsafe fn allocate_uninitialized(&mut self, bytes: usize) -> Result<Self::Buffer> {
        self.allocate(bytes)
    }
    /// Synchronize before replacing a bounded range, including through aliases.
    fn write(&mut self, buffer: &Self::Buffer, offset: usize, data: &[u8]) -> Result<()>;
    /// Initialize float storage without requiring a second full host buffer.
    ///
    /// # Safety
    /// `fill` must initialize all `count` elements before returning normally.
    unsafe fn initialize_f32(
        &mut self,
        buffer: &Self::Buffer,
        count: usize,
        fill: impl FnOnce(&mut [std::mem::MaybeUninit<f32>]),
    ) -> Result<()> {
        ensure!(
            count <= buffer.byte_len() / 4,
            "initialization exceeds buffer"
        );
        let mut values = vec![std::mem::MaybeUninit::<f32>::uninit(); count];
        fill(&mut values);
        // SAFETY: the caller guarantees every element was initialized.
        let bytes = unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), count * 4) };
        self.write(buffer, 0, bytes)
    }
    fn compile(&mut self, source: &KernelSource<'_>, entry: &str) -> Result<Self::Kernel>;

    /// Enqueue a one-dimensional grid, rounded up to a whole threadgroup.
    ///
    /// # Safety
    /// The caller must match the kernel ABI, ensure all accesses fit the buffers,
    /// avoid data races, and guard threads beyond the logical grid size.
    unsafe fn dispatch(
        &mut self,
        kernel: &Self::Kernel,
        args: &[Argument<'_, Self::Buffer>],
        threads: usize,
        group_size: usize,
    ) -> Result<()>;
    fn synchronize(&mut self) -> Result<()>;
    /// Synchronizes pending work before copying the entire buffer to host memory.
    fn read(&mut self, buffer: &Self::Buffer) -> Result<Vec<u8>>;
    /// Synchronize and copy a prefix into caller-owned storage.
    fn read_into(&mut self, buffer: &Self::Buffer, output: &mut [u8]) -> Result<()> {
        output.copy_from_slice(&self.read_prefix(buffer, output.len())?);
        Ok(())
    }
    /// Read only a logical prefix of a reusable allocation.
    fn read_prefix(&mut self, buffer: &Self::Buffer, bytes: usize) -> Result<Vec<u8>> {
        ensure!(bytes <= buffer.byte_len(), "read exceeds buffer size");
        let mut data = self.read(buffer)?;
        data.truncate(bytes);
        Ok(data)
    }
}

/// One sequence per session. A failed step must be followed by reset before reuse.
pub trait AutoregressiveModel {
    fn context_capacity(&self) -> usize;
    fn position(&self) -> usize;
    fn reset(&mut self) -> Result<()>;
    fn step(&mut self, token: u32) -> Result<Vec<f32>>;

    /// Process a prompt and return only the final position's logits.
    fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        ensure!(!tokens.is_empty(), "prompt must not be empty");
        ensure!(
            tokens.len() <= self.context_capacity().saturating_sub(self.position()),
            "prompt exceeds context capacity"
        );
        let mut logits = Vec::new();
        for &token in tokens {
            logits = self.step(token)?;
        }
        Ok(logits)
    }
    /// Backends may select on-device instead of returning the full vocabulary.
    fn step_greedy(&mut self, token: u32) -> Result<u32> {
        greedy(&self.step(token)?)
    }
    fn prefill_greedy(&mut self, tokens: &[u32]) -> Result<u32> {
        greedy(&self.prefill(tokens)?)
    }
}

/// Deterministic greedy sampling; padded vocabulary entries must already be removed.
pub fn greedy(logits: &[f32]) -> Result<u32> {
    ensure!(!logits.is_empty(), "cannot sample empty logits");
    ensure!(logits.iter().all(|x| x.is_finite()), "non-finite logits");
    let mut best = 0;
    for i in 1..logits.len() {
        if logits[i] > logits[best] {
            best = i;
        }
    }
    Ok(u32::try_from(best)?)
}

#[cfg(test)]
mod tests {
    #[test]
    fn greedy_is_stable_and_rejects_bad_logits() {
        assert_eq!(super::greedy(&[-2., 3., 3.]).unwrap(), 1);
        assert!(super::greedy(&[]).is_err());
        assert!(super::greedy(&[f32::NAN]).is_err());
    }
}
