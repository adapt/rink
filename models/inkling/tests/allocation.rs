#![cfg(target_os = "macos")]
use rink_core::{
    Accelerator, AcceleratorKind, Argument, AutoregressiveModel, KernelSource, Result,
};
use rink_inkling::{Image, Inkling, KvCachePrecision};
use rink_metal::{Metal, MetalBuffer, MetalKernel};
use std::path::Path;

/// NaNs expose reads that accidentally depend on newly allocated storage being zero.
struct Poison(Metal);
// SAFETY: every operation delegates to Metal's ordered stream and ownership checks.
unsafe impl Accelerator for Poison {
    type Buffer = MetalBuffer;
    type Kernel = MetalKernel;
    fn kind(&self) -> AcceleratorKind {
        self.0.kind()
    }
    fn name(&self) -> &str {
        self.0.name()
    }
    fn allocate(&mut self, bytes: usize) -> Result<MetalBuffer> {
        self.0.allocate(bytes)
    }
    unsafe fn allocate_uninitialized(&mut self, bytes: usize) -> Result<MetalBuffer> {
        self.0.upload(&vec![0xff; bytes])
    }
    fn upload(&mut self, bytes: &[u8]) -> Result<MetalBuffer> {
        self.0.upload(bytes)
    }
    fn write(&mut self, b: &MetalBuffer, offset: usize, bytes: &[u8]) -> Result<()> {
        self.0.write(b, offset, bytes)
    }
    unsafe fn initialize_f32(
        &mut self,
        b: &MetalBuffer,
        count: usize,
        fill: impl FnOnce(&mut [std::mem::MaybeUninit<f32>]),
    ) -> Result<()> {
        // SAFETY: forwards the caller's full-initialization obligation.
        unsafe { self.0.initialize_f32(b, count, fill) }
    }
    fn compile(&mut self, source: &KernelSource<'_>, entry: &str) -> Result<MetalKernel> {
        // Exercise loading and inference when the optional newer library is unavailable.
        if source.name == rink_inkling::kernels::TENSOR_SOURCE.name {
            anyhow::bail!("optional tensor library unavailable in this test");
        }
        self.0.compile(source, entry)
    }
    unsafe fn dispatch(
        &mut self,
        kernel: &MetalKernel,
        args: &[Argument<'_, MetalBuffer>],
        threads: usize,
        group: usize,
    ) -> Result<()> {
        // SAFETY: forwards the caller's dispatch contract.
        unsafe { self.0.dispatch(kernel, args, threads, group) }
    }
    fn synchronize(&mut self) -> Result<()> {
        self.0.synchronize()
    }
    fn read(&mut self, b: &MetalBuffer) -> Result<Vec<u8>> {
        self.0.read(b)
    }
    fn read_prefix(&mut self, b: &MetalBuffer, count: usize) -> Result<Vec<u8>> {
        self.0.read_prefix(b, count)
    }
    fn read_into(&mut self, b: &MetalBuffer, output: &mut [u8]) -> Result<()> {
        self.0.read_into(b, output)
    }
}

#[test]
#[ignore = "requires a Metal GPU"]
fn unsupported_tensor_library_keeps_base_kernels() {
    let mut device = Poison(Metal::new().unwrap());
    let kernels = rink_inkling::kernels::Kernels::compile(&mut device).unwrap();
    assert!(!kernels.has_tensor_prefill());
}

fn same(a: Vec<f32>, b: Vec<f32>) {
    assert_eq!(a.len(), b.len());
    for (i, (a, b)) in a.into_iter().zip(b).enumerate() {
        assert!(a.is_finite() && b.is_finite(), "non-finite logit {i}");
        assert_eq!(a.to_bits(), b.to_bits(), "logit {i}");
    }
}

#[test]
#[ignore = "requires a Metal GPU"]
fn poisoned_storage_preserves_mixed_turns_resets_and_changing_batches() {
    let directory = Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/vision"
    ));
    let image = Image::open(directory.join("input.png")).unwrap();
    let large = Image::from_rgba(800, 80, vec![127; 800 * 80 * 4]).unwrap();
    for precision in [KvCachePrecision::F16, KvCachePrecision::F32] {
        let mut reused =
            Inkling::load_with_kv_cache(Poison(Metal::new().unwrap()), directory, 128, precision)
                .unwrap();
        for limit in [32, 128, 64, 32, 128] {
            reused.reset().unwrap();
            reused.set_prefill_batch_limit(limit).unwrap();
            let mut fresh =
                Inkling::load_with_kv_cache(Metal::new().unwrap(), directory, 128, precision)
                    .unwrap();
            fresh.set_prefill_batch_limit(limit).unwrap();
            let tokens: Vec<_> = (0..33).map(|i| 1 + i % 10).collect();
            same(
                reused.prefill(&tokens).unwrap(),
                fresh.prefill(&tokens).unwrap(),
            );
            for token in [3, 2] {
                same(reused.step(token).unwrap(), fresh.step(token).unwrap());
            }
            for image in [&image, &large] {
                let tokens = vec![13; image.token_count()];
                same(
                    reused.prefill_image(&tokens, image).unwrap(),
                    fresh.prefill_image(&tokens, image).unwrap(),
                );
                same(reused.step(1).unwrap(), fresh.step(1).unwrap());
            }
            assert_eq!(reused.position(), fresh.position());
        }
    }
}
