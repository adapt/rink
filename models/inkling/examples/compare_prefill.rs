//! Alternate old/new prefill kernels in one loaded model to reduce run-order noise.
//! compare_prefill [rounds=6] [both|experts|dense] [model directory]
#[cfg(target_os = "macos")]
mod comparison {
    use anyhow::{Result, ensure};
    use rink_core::{Accelerator, AcceleratorKind, Argument, AutoregressiveModel, KernelSource};
    use rink_inkling::{Inkling, KvCachePrecision, kernels::SOURCE};
    use rink_metal::{Metal, MetalBuffer, MetalKernel};
    use std::{path::Path, time::Instant};

    struct Choice {
        current: MetalKernel,
        previous: Option<MetalKernel>,
        kind: u8,
    }
    struct Device {
        inner: Metal,
        source: String,
        baseline: bool,
        candidate_mask: u8,
    }
    // SAFETY: storage, synchronization and ownership checks delegate to Metal.
    // Only the five named projection kernels can change. Their buffer/parameter
    // ABI and tile geometry are identical; the reference uses 256 threads.
    unsafe impl Accelerator for Device {
        type Buffer = MetalBuffer;
        type Kernel = Choice;
        fn kind(&self) -> AcceleratorKind {
            self.inner.kind()
        }
        fn name(&self) -> &str {
            self.inner.name()
        }
        fn allocate(&mut self, bytes: usize) -> Result<MetalBuffer> {
            self.inner.allocate(bytes)
        }
        fn upload(&mut self, bytes: &[u8]) -> Result<MetalBuffer> {
            self.inner.upload(bytes)
        }
        unsafe fn allocate_uninitialized(&mut self, bytes: usize) -> Result<MetalBuffer> {
            // SAFETY: forwards the caller's initialization obligation.
            unsafe { self.inner.allocate_uninitialized(bytes) }
        }
        fn write(&mut self, b: &MetalBuffer, offset: usize, bytes: &[u8]) -> Result<()> {
            self.inner.write(b, offset, bytes)
        }
        unsafe fn initialize_f32(
            &mut self,
            b: &MetalBuffer,
            count: usize,
            fill: impl FnOnce(&mut [std::mem::MaybeUninit<f32>]),
        ) -> Result<()> {
            // SAFETY: forwards the caller's complete-initialization obligation.
            unsafe { self.inner.initialize_f32(b, count, fill) }
        }
        fn compile(&mut self, source: &KernelSource<'_>, entry: &str) -> Result<Choice> {
            // Preserve this historical comparison's pre-TensorOps dispatch rules.
            if entry == "matmul_prefill_tensor" {
                anyhow::bail!("TensorOps is excluded from the a0e92fb comparison");
            }
            let current = self.inner.compile(source, entry)?;
            let (previous, kind) = if entry.starts_with("expert_prefill_") {
                let library = KernelSource {
                    name: "previous-prefill",
                    metal: Some(&self.source),
                    cuda: None,
                    subgroup_size: Some(32),
                };
                (
                    Some(self.inner.compile(&library, &format!("previous_{entry}"))?),
                    1,
                )
            } else if entry == "matmul_prefill_compact" {
                (Some(self.inner.compile(source, "matmul_prefill_m64")?), 2)
            } else {
                (None, 0)
            };
            Ok(Choice {
                current,
                previous,
                kind,
            })
        }
        unsafe fn dispatch(
            &mut self,
            k: &Choice,
            args: &[Argument<'_, MetalBuffer>],
            threads: usize,
            group: usize,
        ) -> Result<()> {
            if let Some(previous) = k
                .previous
                .as_ref()
                .filter(|_| self.baseline || self.candidate_mask & k.kind == 0)
            {
                ensure!(
                    group == 128 && threads % group == 0,
                    "unexpected candidate geometry"
                );
                // SAFETY: identical ABI and number of tiles; reference uses 256 threads.
                unsafe {
                    self.inner
                        .dispatch(previous, args, threads / group * 256, 256)
                }
            } else {
                // SAFETY: unchanged kernel, arguments and dispatch geometry.
                unsafe { self.inner.dispatch(&k.current, args, threads, group) }
            }
        }
        fn synchronize(&mut self) -> Result<()> {
            self.inner.synchronize()
        }
        fn read(&mut self, b: &MetalBuffer) -> Result<Vec<u8>> {
            self.inner.read(b)
        }
        fn read_prefix(&mut self, b: &MetalBuffer, bytes: usize) -> Result<Vec<u8>> {
            self.inner.read_prefix(b, bytes)
        }
        fn read_into(&mut self, b: &MetalBuffer, out: &mut [u8]) -> Result<()> {
            self.inner.read_into(b, out)
        }
    }

    pub fn run() -> Result<()> {
        let args: Vec<_> = std::env::args().collect();
        let rounds: usize = args.get(1).map(String::as_str).unwrap_or("6").parse()?;
        ensure!((1..=20).contains(&rounds), "rounds must be in 1..=20");
        let selection = args.get(2).map(String::as_str).unwrap_or("both");
        let candidate_mask = match selection {
            "experts" => 1,
            "dense" => 2,
            "both" => 3,
            _ => anyhow::bail!("select both, experts or dense"),
        };
        let directory = Path::new(
            args.get(3)
                .map(String::as_str)
                .unwrap_or("checkpoints/inkling-2-mini"),
        );
        let tokenizer = tokenizers::Tokenizer::from_file(directory.join("tokenizer.json"))
            .map_err(anyhow::Error::msg)?;
        let encoded = tokenizer
            .encode(
                "Explain how a computer processes information, using clear examples. ",
                false,
            )
            .map_err(anyhow::Error::msg)?;
        ensure!(!encoded.is_empty(), "empty benchmark prompt");
        let mut inner = Metal::new()?;
        inner.set_profiling(true);
        let device = Device {
            inner,
            source: format!(
                "{}\n{}",
                SOURCE.metal.unwrap(),
                include_str!("previous_prefill.metal")
            ),
            baseline: false,
            candidate_mask,
        };
        let mut model =
            Inkling::load_with_kv_cache(device, directory, 262144, KvCachePrecision::F16)?;
        model.set_prefill_batch_limit(1024)?;
        for &token in encoded.get_ids().iter().take(4) {
            model.step_greedy(token)?;
        }
        let mut records = Vec::new();
        for length in [512, 8192] {
            let prompt: Vec<_> = encoded
                .get_ids()
                .iter()
                .copied()
                .cycle()
                .take(length)
                .collect();
            let mut expected = None;
            for round in 0..rounds {
                for j in 0..2 {
                    let baseline = (round + j) % 2 == 0;
                    model.accelerator_mut().baseline = baseline;
                    model.reset()?;
                    model.accelerator_mut().inner.reset_stats();
                    let start = Instant::now();
                    let mut next = model.prefill_greedy(&prompt)?;
                    let seconds = start.elapsed().as_secs_f64();
                    let stats = model.accelerator().inner.stats();
                    let mut generated = Vec::new();
                    for _ in 0..32 {
                        generated.push(next);
                        next = model.step_greedy(next)?;
                    }
                    if let Some(expected) = &expected {
                        ensure!(&generated == expected, "generated sequence differs");
                    } else {
                        expected = Some(generated);
                    }
                    let record = serde_json::json!({"prompt_tokens":length,"round":round,"baseline":baseline,"selection":selection,"seconds":seconds,"tokens_per_second":length as f64/seconds,"gpu_seconds":stats.gpu_seconds,"allocations":stats.allocations,"peak_allocated_bytes":stats.peak_allocated_bytes,"generated_tokens_match":true});
                    eprintln!("{record}");
                    records.push(record);
                }
            }
        }
        println!(
            "{}",
            serde_json::to_string_pretty(
                &serde_json::json!({"device":model.device_name(),"context":262144,"prefill_batch":1024,"kv_cache":"f16","rounds":rounds,"candidate":selection,"baseline_revision":"a0e92fb","method":"Alternating old/new kernels in one loaded checkpoint; same tensors and reset between trials; no per-kernel profiling; four initial decode warmup tokens. First-use allocations included.","trials":records})
            )?
        );
        Ok(())
    }
}
#[cfg(target_os = "macos")]
fn main() -> anyhow::Result<()> {
    comparison::run()
}
#[cfg(not(target_os = "macos"))]
fn main() -> anyhow::Result<()> {
    anyhow::bail!("requires Metal on macOS")
}
