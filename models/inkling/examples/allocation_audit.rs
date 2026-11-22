//! Count Rust heap traffic and simulate device allocation lifetimes without GPU math.
//! allocation_audit MODEL_DIR [CONTEXT]
//! No tensor data is stored. Host fills and shader execution are skipped, so
//! timings and image CPU heap counts do not describe a real inference run.
use anyhow::Result;
use rink_core::{
    Accelerator, AcceleratorKind, Argument, AutoregressiveModel, Buffer, KernelSource, ensure,
};
use rink_inkling::{Image, Inkling};
use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::RefCell,
    path::Path,
    rc::Rc,
    sync::atomic::{AtomicUsize, Ordering::Relaxed},
    time::Instant,
};

struct CountHeap;
static CALLS: AtomicUsize = AtomicUsize::new(0);
static REALLOCS: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
fn add(n: usize) {
    BYTES.fetch_add(n, Relaxed);
    let live = LIVE.fetch_add(n, Relaxed) + n;
    PEAK.fetch_max(live, Relaxed);
}
unsafe impl GlobalAlloc for CountHeap {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(l) };
        if !p.is_null() {
            CALLS.fetch_add(1, Relaxed);
            add(l.size());
        }
        p
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        let p = unsafe { System.alloc_zeroed(l) };
        if !p.is_null() {
            CALLS.fetch_add(1, Relaxed);
            add(l.size());
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        LIVE.fetch_sub(l.size(), Relaxed);
        unsafe { System.dealloc(p, l) };
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        let q = unsafe { System.realloc(p, l, n) };
        if !q.is_null() {
            REALLOCS.fetch_add(1, Relaxed);
            LIVE.fetch_sub(l.size(), Relaxed);
            add(n);
        }
        q
    }
}
#[global_allocator]
static ALLOC: CountHeap = CountHeap;
fn phase<T>(name: &str, f: impl FnOnce() -> Result<T>) -> Result<T> {
    let calls = CALLS.load(Relaxed);
    let realloc = REALLOCS.load(Relaxed);
    let bytes = BYTES.load(Relaxed);
    let live = LIVE.load(Relaxed);
    PEAK.store(live, Relaxed);
    let start = Instant::now();
    let value = f()?;
    let seconds = start.elapsed().as_secs_f64();
    let allocations = CALLS.load(Relaxed) - calls;
    let reallocations = REALLOCS.load(Relaxed) - realloc;
    let requested = BYTES.load(Relaxed) - bytes;
    let peak = PEAK.load(Relaxed).saturating_sub(live);
    let delta = LIVE.load(Relaxed) as i64 - live as i64;
    let output = serde_json::json!({"phase":name,"seconds":seconds,"heap_allocations":allocations,"heap_reallocations":reallocations,"heap_bytes_requested":requested,"heap_peak_above_start":peak,"heap_live_delta":delta});
    println!("{output}");
    Ok(value)
}
#[derive(Default)]
struct Stats {
    live: usize,
    peak: usize,
    allocations: usize,
    allocated: usize,
    dispatches: usize,
}
struct Storage {
    bytes: usize,
    stats: Rc<RefCell<Stats>>,
}
impl Drop for Storage {
    fn drop(&mut self) {
        self.stats.borrow_mut().live -= self.bytes;
    }
}
#[derive(Clone)]
struct SizedBuffer(Rc<Storage>, usize);
impl Buffer for SizedBuffer {
    fn byte_len(&self) -> usize {
        self.1
    }
    fn view(&self, offset: usize, bytes: usize) -> Result<Self> {
        ensure!(
            offset % 256 == 0 && offset <= self.1 && bytes > 0 && bytes <= self.1 - offset,
            "invalid view"
        );
        Ok(Self(self.0.clone(), bytes))
    }
}
struct SizeOnly {
    stats: Rc<RefCell<Stats>>,
    pending: Vec<SizedBuffer>,
}
impl SizeOnly {
    fn new() -> Self {
        Self {
            stats: Rc::new(RefCell::new(Stats::default())),
            pending: Vec::with_capacity(262144),
        }
    }
    fn reset_stats(&mut self) {
        let live = self.stats.borrow().live;
        *self.stats.borrow_mut() = Stats {
            live,
            peak: live,
            ..Stats::default()
        };
    }
    fn report(&self, name: &str) {
        let s = self.stats.borrow();
        println!(
            "{}",
            serde_json::json!({"phase_device":name,"simulated_live_bytes":s.live,"simulated_peak_bytes":s.peak,"device_allocations":s.allocations,"device_allocated_bytes":s.allocated,"dispatches":s.dispatches})
        );
    }
}
// Size/lifetime accounting only. Pending dispatches retain handles, but no GPU
// tensor data is stored or computed. Reads return zero, not model results.
unsafe impl Accelerator for SizeOnly {
    type Buffer = SizedBuffer;
    type Kernel = ();
    fn kind(&self) -> AcceleratorKind {
        AcceleratorKind::Metal
    }
    fn name(&self) -> &str {
        "size-only"
    }
    fn allocate(&mut self, bytes: usize) -> Result<SizedBuffer> {
        let mut s = self.stats.borrow_mut();
        s.live += bytes;
        s.peak = s.peak.max(s.live);
        s.allocations += 1;
        s.allocated += bytes;
        Ok(SizedBuffer(
            Rc::new(Storage {
                bytes,
                stats: self.stats.clone(),
            }),
            bytes,
        ))
    }
    fn upload(&mut self, bytes: &[u8]) -> Result<SizedBuffer> {
        self.allocate(bytes.len())
    }
    fn write(&mut self, b: &SizedBuffer, offset: usize, data: &[u8]) -> Result<()> {
        ensure!(
            offset <= b.byte_len() && data.len() <= b.byte_len() - offset,
            "write exceeds buffer"
        );
        self.synchronize()
    }
    unsafe fn initialize_f32(
        &mut self,
        b: &SizedBuffer,
        count: usize,
        _: impl FnOnce(&mut [std::mem::MaybeUninit<f32>]),
    ) -> Result<()> {
        ensure!(count <= b.byte_len() / 4, "fill exceeds buffer");
        self.synchronize()
    }
    fn read_into(&mut self, b: &SizedBuffer, out: &mut [u8]) -> Result<()> {
        ensure!(out.len() <= b.byte_len(), "read exceeds buffer");
        self.synchronize()?;
        out.fill(0);
        Ok(())
    }
    fn compile(&mut self, _: &KernelSource<'_>, _: &str) -> Result<()> {
        Ok(())
    }
    unsafe fn dispatch(
        &mut self,
        _: &(),
        args: &[Argument<'_, SizedBuffer>],
        _: usize,
        _: usize,
    ) -> Result<()> {
        for a in args {
            if let Argument::Buffer(b) = a {
                self.pending.push((*b).clone());
            }
        }
        self.stats.borrow_mut().dispatches += 1;
        Ok(())
    }
    fn synchronize(&mut self) -> Result<()> {
        self.pending.clear();
        Ok(())
    }
    fn read(&mut self, b: &SizedBuffer) -> Result<Vec<u8>> {
        self.read_prefix(b, b.byte_len())
    }
    fn read_prefix(&mut self, _: &SizedBuffer, bytes: usize) -> Result<Vec<u8>> {
        self.synchronize()?;
        Ok(vec![0; bytes])
    }
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    ensure!(
        (2..=3).contains(&args.len()),
        "allocation_audit MODEL_DIR [CONTEXT]"
    );
    let directory = Path::new(&args[1]);
    let capacity = args
        .get(2)
        .map(|s| s.parse::<usize>())
        .transpose()?
        .unwrap_or(262144);
    ensure!(
        capacity >= 32768,
        "audit needs at least 32768 context positions"
    );
    let cp = rink_inkling::checkpoint::Checkpoint::open(directory)?;
    phase("1000_checkpoint_tensor_lookups", || {
        for _ in 0..1000 {
            std::hint::black_box(cp.tensor("model.llm.embed.weight")?);
        }
        Ok(())
    })?;
    drop(cp);
    let device = SizeOnly::new();
    let mut model = phase("load", || Inkling::load(device, directory, capacity))?;
    model.accelerator().report("load");
    phase("warmup_decode", || {
        model.step_greedy(0)?;
        Ok(())
    })?;
    model.accelerator_mut().reset_stats();
    phase("100_warm_decode_steps", || {
        for _ in 0..100 {
            model.step_greedy(0)?;
        }
        Ok(())
    })?;
    model.accelerator().report("100_warm_decode_steps");
    model.reset()?;
    let prompt = vec![0; 8192];
    model.accelerator_mut().reset_stats();
    phase("8192_prefill", || model.prefill_greedy(&prompt))?;
    model.accelerator().report("8192_prefill");
    model.reset()?;
    model.accelerator_mut().reset_stats();
    phase("8192_prefill_repeated", || model.prefill_greedy(&prompt))?;
    model.accelerator().report("8192_prefill_repeated");
    model.accelerator_mut().reset_stats();
    phase("100_decode_after_prefill", || {
        for _ in 0..100 {
            model.step_greedy(0)?;
        }
        Ok(())
    })?;
    model.accelerator().report("100_decode_after_prefill");
    let image = Image::from_rgba(5344, 3044, vec![255; 5344 * 3044 * 4])?;

    let tokens = vec![model.config().image_token_id; image.token_count()];
    model.accelerator_mut().reset_stats();
    phase("image_after_text", || {
        model.prefill_image_greedy_with_progress(&tokens, &image, |_| {})
    })?;
    model.accelerator().report("image_after_text");
    let mut previous_peak = None;
    for turn in 0..8 {
        model.reset()?;
        model.accelerator_mut().reset_stats();
        let name = format!("image_repeated_{turn}");
        phase(&name, || {
            model.prefill_image_greedy_with_progress(&tokens, &image, |_| {})
        })?;
        model.accelerator().report(&name);
        let stats = model.accelerator().stats.borrow();
        if turn >= 6 {
            ensure!(
                Some(stats.peak) == previous_peak && stats.allocations == 0,
                "image pool did not stabilize"
            );
        }
        previous_peak = Some(stats.peak);
    }
    drop(model);
    let mut fresh = Inkling::load(SizeOnly::new(), directory, capacity)?;
    fresh.accelerator_mut().reset_stats();
    phase("image_fresh_model", || {
        fresh.prefill_image_greedy_with_progress(&tokens, &image, |_| {})
    })?;
    fresh.accelerator().report("image_fresh_model");
    Ok(())
}
