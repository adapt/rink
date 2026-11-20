//! Opt-in stage-boundary timestamps. Each dispatch gets its own compute pass;
//! these diagnostic timings must not be used as normal throughput measurements.
use metal::{ComputeCommandEncoder, DeviceRef, MTLStorageMode, NSRange};
use objc::{msg_send, sel, sel_impl};
use rink_core::{Result, ensure};
use std::{collections::BTreeMap, sync::Arc};

// Two 8-byte timestamps per dispatch, within Apple's 32 KiB sample-buffer limit.
const MAX_DISPATCHES: usize = 2048;

#[derive(Clone, Copy, Debug, Default)]
pub struct MetalKernelStats {
    pub dispatches: u64,
    pub gpu_seconds: f64,
}

pub(super) struct KernelProfile {
    samples: metal::CounterSampleBuffer,
    pending: Vec<Arc<str>>,
    start_cpu: u64,
    start_gpu: u64,
    pub stats: BTreeMap<String, MetalKernelStats>,
}

impl KernelProfile {
    pub fn new(device: &DeviceRef) -> Result<Self> {
        ensure!(
            device.supports_counter_sampling(metal::MTLCounterSamplingPoint::AtStageBoundary),
            "kernel profiling requires Metal stage-boundary counter sampling"
        );
        let timestamp = device
            .counter_sets()
            .into_iter()
            .find(|set| set.name().eq_ignore_ascii_case("timestamp"))
            .ok_or_else(|| rink_core::Error::msg("Metal timestamp counter set is unavailable"))?;
        let descriptor = metal::CounterSampleBufferDescriptor::new();
        descriptor.set_counter_set(&timestamp);
        descriptor.set_storage_mode(MTLStorageMode::Shared);
        descriptor.set_sample_count((MAX_DISPATCHES * 2) as u64);
        descriptor.set_label("Rink kernel timestamps");
        let samples = device
            .new_counter_sample_buffer_with_descriptor(&descriptor)
            .map_err(rink_core::Error::msg)?;
        Ok(Self {
            samples,
            pending: Vec::new(),
            start_cpu: 0,
            start_gpu: 0,
            stats: BTreeMap::new(),
        })
    }

    pub fn full(&self) -> bool {
        self.pending.len() == MAX_DISPATCHES
    }

    pub fn encoder(
        &mut self,
        device: &DeviceRef,
        command: &metal::CommandBufferRef,
        name: &Arc<str>,
    ) -> ComputeCommandEncoder {
        if self.pending.is_empty() {
            device.sample_timestamps(&mut self.start_cpu, &mut self.start_gpu);
        }
        let descriptor = metal::ComputePassDescriptor::new();
        let attachment = metal::ComputePassSampleBufferAttachmentDescriptor::new();
        attachment.set_sample_buffer(&self.samples);
        attachment.set_start_of_encoder_sample_index((self.pending.len() * 2) as u64);
        attachment.set_end_of_encoder_sample_index((self.pending.len() * 2 + 1) as u64);
        descriptor
            .sample_buffer_attachments()
            .set_object_at(0, Some(&attachment));
        self.pending.push(name.clone());
        command
            .compute_command_encoder_with_descriptor(descriptor)
            .to_owned()
    }

    pub fn resolve(&mut self, device: &DeviceRef) -> Result<()> {
        if self.pending.is_empty() {
            return Ok(());
        }
        let (mut cpu, mut gpu) = (0, 0);
        device.sample_timestamps(&mut cpu, &mut gpu);
        let names = std::mem::take(&mut self.pending);
        ensure!(
            cpu > self.start_cpu && gpu > self.start_gpu,
            "invalid Metal clock calibration"
        );
        // sampleTimestamps reports CPU time in nanoseconds. Calibrate the GPU's
        // clock before/after execution instead of assuming GPU timestamp units.
        let seconds_per_tick =
            (cpu - self.start_cpu) as f64 / (gpu - self.start_gpu) as f64 * 1.0e-9;
        let timestamps = self.timestamps(names.len() * 2)?;
        for (name, pair) in names.iter().zip(timestamps.chunks_exact(2)) {
            ensure!(
                pair[0] != u64::MAX && pair[1] != u64::MAX && pair[1] >= pair[0],
                "invalid Metal kernel timestamp"
            );
            let stat = self.stats.entry(name.to_string()).or_default();
            stat.dispatches += 1;
            stat.gpu_seconds += (pair[1] - pair[0]) as f64 * seconds_per_tick;
        }
        Ok(())
    }

    fn timestamps(&self, count: usize) -> Result<Vec<u64>> {
        objc::rc::autoreleasepool(|| {
            // Resolve directly: metal 0.32's resolve_counter_range computes its
            // copy length from an empty Vec and returns uninitialized elements.
            #[allow(unexpected_cfgs)]
            let data: *mut objc::runtime::Object = unsafe {
                // SAFETY: completed sample buffer; range contains recorded samples.
                msg_send![&*self.samples, resolveCounterRange: NSRange::new(0, count as u64)]
            };
            ensure!(!data.is_null(), "Metal failed to resolve timestamp samples");
            #[allow(unexpected_cfgs)]
            let (bytes, length): (*const u8, usize) = unsafe {
                // SAFETY: non-null NSData, alive within this autorelease pool.
                (msg_send![data, bytes], msg_send![data, length])
            };
            ensure!(
                !bytes.is_null() && length == count * 8,
                "unexpected Metal timestamp data size"
            );
            let mut timestamps = vec![0u64; count];
            // SAFETY: both buffers hold count * 8 bytes and do not overlap. The
            // byte copy does not require NSData's pointer to be u64-aligned.
            unsafe { std::ptr::copy_nonoverlapping(bytes, timestamps.as_mut_ptr().cast(), length) };
            Ok(timestamps)
        })
    }
}
