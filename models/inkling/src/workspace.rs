//! Storage shared by serialized text and image work. A reset reuses addresses
//! only after their last GPU reads have been encoded on the ordered stream.
use anyhow::{Context, Result};
use rink_core::{Accelerator, Buffer};

pub(crate) struct Pool<B> {
    free: Vec<B>,
}
impl<B: Buffer> Pool<B> {
    pub fn new() -> Self {
        Self { free: Vec::new() }
    }
    pub fn put(&mut self, buffer: B) {
        self.free.push(buffer);
    }
    pub fn take<A: Accelerator<Buffer = B>>(&mut self, device: &mut A, bytes: usize) -> Result<B> {
        if let Some(i) = self
            .free
            .iter()
            .enumerate()
            .filter(|(_, b)| b.byte_len() >= bytes)
            .min_by_key(|(_, b)| b.byte_len())
            .map(|(i, _)| i)
        {
            return Ok(self.free.swap_remove(i));
        }
        // Release idle blocks before growing, so alternate image sizes or
        // prompt batches do not accumulate separate high-water reservations.
        self.free.clear();
        // SAFETY: the model initializes each logical tensor before reading it.
        // Padding in these allocations is never passed as logical tensor data.
        unsafe { device.allocate_uninitialized(bytes) }
    }
}

struct Block<B> {
    buffer: B,
    used: usize,
}
pub(crate) struct Arena<B> {
    blocks: Vec<Block<B>>,
}
impl<B: Buffer> Arena<B> {
    pub fn new() -> Self {
        Self { blocks: Vec::new() }
    }
    /// Caller must have encoded the last reads of every previous view.
    pub fn reset(&mut self) {
        for block in &mut self.blocks {
            block.used = 0;
        }
    }
    /// Caller must drop all views, or finish their reads before reusing the pool.
    pub fn release(&mut self, pool: &mut Pool<B>) {
        for block in self.blocks.drain(..) {
            pool.put(block.buffer);
        }
    }
    pub fn alloc<A: Accelerator<Buffer = B>>(
        &mut self,
        device: &mut A,
        pool: &mut Pool<B>,
        bytes: usize,
    ) -> Result<B> {
        let aligned = bytes.checked_add(255).context("workspace size overflow")? / 256 * 256;
        let index = match self
            .blocks
            .iter()
            .position(|b| aligned <= b.buffer.byte_len() - b.used)
        {
            Some(i) => i,
            None => {
                let buffer = pool.take(device, aligned.max(64 * 1024))?;
                self.blocks.push(Block { buffer, used: 0 });
                self.blocks.len() - 1
            }
        };
        let block = &mut self.blocks[index];
        let view = block.buffer.view(block.used, bytes)?;
        block.used += aligned;
        Ok(view)
    }
}
