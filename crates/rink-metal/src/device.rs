use foreign_types::ForeignType;
use metal::{CompileOptions, Device, MTLCommandBufferStatus, MTLResourceOptions, MTLSize};
use objc::{msg_send, sel, sel_impl};
use rink_core::{Accelerator, AcceleratorKind, Argument, Buffer, KernelSource, Result, ensure};
use std::{collections::HashMap, sync::Arc};

#[derive(Clone)]
pub struct MetalBuffer {
    raw: metal::Buffer,
    owner: Arc<()>,
}
impl Buffer for MetalBuffer {
    fn byte_len(&self) -> usize {
        self.raw.length() as usize
    }
}

pub struct MetalKernel {
    pipeline: metal::ComputePipelineState,
    owner: Arc<()>,
}

pub struct Metal {
    device: Device,
    queue: metal::CommandQueue,
    pending: Option<metal::CommandBuffer>,
    libraries: HashMap<String, (String, metal::Library)>,
    owner: Arc<()>,
    name: String,
}

impl Metal {
    pub fn new() -> Result<Self> {
        let device = Device::system_default()
            .ok_or_else(|| rink_core::Error::msg("no Metal GPU available"))?;
        let name = device.name().to_string();
        let queue = device.new_command_queue();
        Ok(Self {
            device,
            queue,
            pending: None,
            libraries: HashMap::new(),
            owner: Arc::new(()),
            name,
        })
    }
    pub fn recommended_memory(&self) -> u64 {
        self.device.recommended_max_working_set_size()
    }
}

// SAFETY: command buffers retain their resources. All host reads and Drop wait for completion.
// Ownership tags reject resources from other streams, avoiding cross-stream data races.
unsafe impl Accelerator for Metal {
    type Buffer = MetalBuffer;
    type Kernel = MetalKernel;
    fn kind(&self) -> AcceleratorKind {
        AcceleratorKind::Metal
    }
    fn name(&self) -> &str {
        &self.name
    }
    fn allocate(&mut self, bytes: usize) -> Result<MetalBuffer> {
        ensure!(
            bytes > 0 && bytes <= self.device.max_buffer_length() as usize,
            "invalid Metal allocation: {bytes} bytes"
        );
        ensure!(
            self.device
                .current_allocated_size()
                .saturating_add(bytes as u64)
                <= self.recommended_memory(),
            "allocation of {bytes} bytes exceeds Metal's recommended working set; reduce context capacity"
        );
        // SAFETY: valid Objective-C device and selector. Use a raw nullable result so
        // allocation failure becomes an error instead of a foreign-types null pointer.
        let device = &*self.device;
        #[allow(unexpected_cfgs)]
        // objc 0.2's selector macro probes its legacy cargo-clippy feature.
        let ptr: *mut metal::MTLBuffer = unsafe {
            msg_send![device,
            newBufferWithLength: bytes as u64 options: MTLResourceOptions::StorageModeShared]
        };
        ensure!(!ptr.is_null(), "Metal could not allocate {bytes} bytes");
        // SAFETY: newBuffer returned a non-null owned (+1) MTLBuffer.
        let raw = unsafe { metal::Buffer::from_ptr(ptr) };
        // SAFETY: new shared buffer, no pending GPU accesses, exactly its allocated length.
        unsafe {
            std::ptr::write_bytes(raw.contents().cast::<u8>(), 0, bytes);
        }
        Ok(MetalBuffer {
            raw,
            owner: self.owner.clone(),
        })
    }
    fn upload(&mut self, bytes: &[u8]) -> Result<MetalBuffer> {
        let buffer = self.allocate(bytes.len())?;
        // SAFETY: the newly allocated shared buffer does not alias the source.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                buffer.raw.contents().cast(),
                bytes.len(),
            );
        }
        Ok(buffer)
    }
    fn compile(&mut self, source: &KernelSource<'_>, entry: &str) -> Result<MetalKernel> {
        objc::rc::autoreleasepool(|| {
            let code = source.metal.ok_or_else(|| {
                rink_core::Error::msg(format!("{} has no Metal kernels", source.name))
            })?;
            if let Some((old, _)) = self.libraries.get(source.name) {
                ensure!(
                    old == code,
                    "kernel library name collision: {}",
                    source.name
                );
            } else {
                let options = CompileOptions::new();
                options.set_fast_math_enabled(false);
                let library = self
                    .device
                    .new_library_with_source(code, &options)
                    .map_err(rink_core::Error::msg)?;
                self.libraries
                    .insert(source.name.to_string(), (code.to_string(), library));
            }
            let function = self.libraries[source.name]
                .1
                .get_function(entry, None)
                .map_err(rink_core::Error::msg)?;
            let pipeline = self
                .device
                .new_compute_pipeline_state_with_function(&function)
                .map_err(rink_core::Error::msg)?;
            if let Some(required) = source.subgroup_size {
                ensure!(
                    pipeline.thread_execution_width() as usize == required,
                    "{} requires subgroup width {required}, GPU kernel provides {}",
                    source.name,
                    pipeline.thread_execution_width()
                );
            }
            Ok(MetalKernel {
                pipeline,
                owner: self.owner.clone(),
            })
        })
    }
    unsafe fn dispatch(
        &mut self,
        kernel: &MetalKernel,
        args: &[Argument<'_, MetalBuffer>],
        threads: usize,
        group_size: usize,
    ) -> Result<()> {
        ensure!(
            Arc::ptr_eq(&kernel.owner, &self.owner),
            "kernel belongs to a different device stream"
        );
        ensure!(
            threads > 0
                && group_size > 0
                && group_size <= kernel.pipeline.max_total_threads_per_threadgroup() as usize,
            "invalid dispatch dimensions"
        );
        for arg in args {
            match arg {
                Argument::Buffer(b) => ensure!(
                    Arc::ptr_eq(&b.owner, &self.owner),
                    "buffer belongs to a different device stream"
                ),
                Argument::Bytes(b) => ensure!(
                    !b.is_empty() && b.len() <= 4096,
                    "inline argument must contain 1..=4096 bytes"
                ),
            }
        }
        objc::rc::autoreleasepool(|| {
            let command = self
                .pending
                .get_or_insert_with(|| self.queue.new_command_buffer().to_owned());
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&kernel.pipeline);
            for (i, arg) in args.iter().enumerate() {
                match arg {
                    Argument::Buffer(b) => encoder.set_buffer(i as u64, Some(&b.raw), 0),
                    Argument::Bytes(b) => {
                        encoder.set_bytes(i as u64, b.len() as u64, b.as_ptr().cast())
                    }
                }
            }
            encoder.dispatch_thread_groups(
                MTLSize::new(threads.div_ceil(group_size) as u64, 1, 1),
                MTLSize::new(group_size as u64, 1, 1),
            );
            encoder.end_encoding();
        });
        Ok(())
    }
    fn synchronize(&mut self) -> Result<()> {
        if let Some(command) = self.pending.take() {
            command.commit();
            command.wait_until_completed();
            ensure!(
                command.status() == MTLCommandBufferStatus::Completed,
                "Metal command failed: {:?}",
                command.status()
            );
        }
        Ok(())
    }
    fn read(&mut self, buffer: &MetalBuffer) -> Result<Vec<u8>> {
        ensure!(
            Arc::ptr_eq(&buffer.owner, &self.owner),
            "buffer belongs to a different device stream"
        );
        self.synchronize()?;
        // SAFETY: shared storage; all work is complete, and the slice fits the buffer.
        Ok(unsafe {
            std::slice::from_raw_parts(buffer.raw.contents().cast::<u8>(), buffer.byte_len())
        }
        .to_vec())
    }
}
impl Drop for Metal {
    fn drop(&mut self) {
        let _ = self.synchronize();
    }
}
