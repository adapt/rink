//! Audit the real decoder's allocation schedule through a full context without
//! executing GPU math. Buffers carry sizes and lifetimes, not tensor contents.
//! This is an allocation simulation, not a correctness or GPU-memory benchmark.
//! prefill_memory MODEL [CONTEXT] [BATCH_LIMIT]
use anyhow::{Result, ensure};
use rink_core::{Accelerator, AcceleratorKind, Argument, Buffer, KernelSource};
use rink_inkling::{Inkling, PrefillProgress};
use std::{cell::RefCell, path::Path, rc::Rc};

#[derive(Default)]
struct Stats {
    live: usize,
    peak: usize,
}
struct Allocation {
    bytes: usize,
    stats: Rc<RefCell<Stats>>,
}
impl Drop for Allocation {
    fn drop(&mut self) {
        self.stats.borrow_mut().live -= self.bytes;
    }
}
#[derive(Clone)]
struct SizedBuffer(Rc<Allocation>);
impl Buffer for SizedBuffer {
    fn byte_len(&self) -> usize {
        self.0.bytes
    }
}
#[derive(Default)]
struct Allocator {
    stats: Rc<RefCell<Stats>>,
    pending: Vec<SizedBuffer>,
}
// SAFETY: no tensor data exists and no memory is accessed. Pending buffers are
// retained until synchronize, just as an ordered accelerator must retain them.
unsafe impl Accelerator for Allocator {
    type Buffer = SizedBuffer;
    type Kernel = Rc<RefCell<Stats>>;
    fn kind(&self) -> AcceleratorKind {
        AcceleratorKind::Metal
    }
    fn name(&self) -> &str {
        "allocation simulation (no GPU math)"
    }
    fn allocate(&mut self, bytes: usize) -> Result<SizedBuffer> {
        let mut s = self.stats.borrow_mut();
        s.live += bytes;
        s.peak = s.peak.max(s.live);
        Ok(SizedBuffer(Rc::new(Allocation {
            bytes,
            stats: self.stats.clone(),
        })))
    }
    fn upload(&mut self, bytes: &[u8]) -> Result<SizedBuffer> {
        self.allocate(bytes.len())
    }
    fn compile(&mut self, _: &KernelSource<'_>, _: &str) -> Result<Self::Kernel> {
        Ok(self.stats.clone())
    }
    unsafe fn dispatch(
        &mut self,
        kernel: &Self::Kernel,
        args: &[Argument<'_, SizedBuffer>],
        _: usize,
        _: usize,
    ) -> Result<()> {
        ensure!(
            Rc::ptr_eq(kernel, &self.stats),
            "kernel belongs to another allocator"
        );
        for arg in args {
            if let Argument::Buffer(b) = arg {
                ensure!(
                    Rc::ptr_eq(&b.0.stats, &self.stats),
                    "buffer belongs to another allocator"
                );
                self.pending.push((*b).clone());
            }
        }
        Ok(())
    }
    fn synchronize(&mut self) -> Result<()> {
        self.pending.clear();
        Ok(())
    }
    fn read(&mut self, b: &SizedBuffer) -> Result<Vec<u8>> {
        self.read_prefix(b, b.byte_len())
    }
    fn read_prefix(&mut self, b: &SizedBuffer, bytes: usize) -> Result<Vec<u8>> {
        ensure!(
            Rc::ptr_eq(&b.0.stats, &self.stats),
            "buffer belongs to another allocator"
        );
        ensure!(bytes <= b.byte_len(), "read exceeds buffer");
        self.synchronize()?;
        Ok(vec![0; bytes])
    }
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    ensure!(
        (2..=4).contains(&args.len()),
        "prefill_memory MODEL [CONTEXT] [BATCH_LIMIT]"
    );
    let capacity = args
        .get(2)
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(262144);
    let limit = args.get(3).map(|s| s.parse()).transpose()?;
    let device = Allocator::default();
    let stats = device.stats.clone();
    let mut model = Inkling::load(device, Path::new(&args[1]), capacity)?;
    if let Some(limit) = limit {
        model.set_prefill_batch_limit(limit)?;
    }
    let limit = model.prefill_batch_limit();
    let loaded = stats.borrow().live;
    let mut checkpoints = Vec::new();
    let mut sizes = std::collections::BTreeMap::<usize, usize>::new();
    let mut previous = 0;
    let mut next_checkpoint = 128;
    let mut warm_peak = None;
    let mut stable = true;
    model.prefill_greedy_with_progress(&vec![0; capacity], |event| {
        let completed = match event {
            PrefillProgress::Prompt { completed_tokens, .. } => completed_tokens,
            PrefillProgress::Complete => capacity,
            _ => return,
        };
        if completed > previous {
            *sizes.entry(completed - previous).or_default() += 1;
            previous = completed;
        }
        if completed > 0 && completed < capacity {
            let peak = stats.borrow().peak;
            stable &= peak == *warm_peak.get_or_insert(peak);
        }
        if completed >= next_checkpoint || completed == capacity {
            let s = stats.borrow();
            checkpoints.push(serde_json::json!({"position": completed, "live_bytes": s.live, "peak_bytes": s.peak}));
            next_checkpoint = (completed / 16384 + 1) * 16384;
        }
    })?;
    ensure!(
        stable,
        "prefill peak allocation grew after the first full batch"
    );
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "method": "allocation simulation; no GPU execution, page rounding or driver overhead",
            "context": capacity, "batch_limit": limit, "kv_cache": "f16", "loaded_bytes": loaded,
            "live_bytes": stats.borrow().live, "peak_bytes": stats.borrow().peak,
            "stable_prefill_peak": stable,
            "batches": sizes, "checkpoints": checkpoints,
        }))?
    );
    Ok(())
}
