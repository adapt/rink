#![cfg(target_os = "macos")]
use rink_core::{Accelerator, Argument, KernelSource};
use rink_metal::Metal;

#[test]
#[ignore = "requires a Metal GPU"]
fn ordered_dispatch_keeps_resources_alive_and_synchronizes_reads() {
    let mut device = Metal::new().unwrap();
    let source = KernelSource {
        name: "device-test",
        metal: Some(
            "#include <metal_stdlib>\nusing namespace metal; kernel void add_one(device uint* y [[buffer(0)]], device const uint* x [[buffer(1)]], uint i [[thread_position_in_grid]]) { if (i<1) y[0]=x[0]+1; }",
        ),
        cuda: None,
        subgroup_size: None,
    };
    let kernel = device.compile(&source, "add_one").unwrap();
    let input = device.upload(&41u32.to_le_bytes()).unwrap();
    let intermediate = device.allocate(4).unwrap();
    let output = device.allocate(4).unwrap();
    // SAFETY: each buffer contains one u32 and the kernel guards extra threads.
    unsafe {
        device
            .dispatch(
                &kernel,
                &[Argument::Buffer(&intermediate), Argument::Buffer(&input)],
                1,
                32,
            )
            .unwrap();
        device
            .dispatch(
                &kernel,
                &[Argument::Buffer(&output), Argument::Buffer(&intermediate)],
                1,
                32,
            )
            .unwrap();
    }
    drop(input);
    drop(intermediate); // Command buffer must retain both resources.
    assert_eq!(device.read(&output).unwrap(), 43u32.to_le_bytes());
    assert_eq!(device.stats().encoders, 1);
    assert_eq!(device.stats().command_buffers, 1);
    assert_eq!(device.read_prefix(&output, 2).unwrap(), [43, 0]);
    assert!(device.read_prefix(&output, 5).is_err());
    let mut other = Metal::new().unwrap();
    assert!(other.read(&output).is_err());
    assert!(device.allocate(0).is_err());
    assert!(device.allocate(usize::MAX).is_err());
}

#[test]
#[ignore = "requires a Metal GPU with timestamp counters"]
fn kernel_timestamps_preserve_order_and_reset_between_phases() {
    let mut device = Metal::new().unwrap();
    device.set_profiling(true);
    device.set_kernel_profiling(true).unwrap();
    let source = KernelSource {
        name: "timestamp-test",
        metal: Some(
            "#include <metal_stdlib>\nusing namespace metal; kernel void increment(device uint* x [[buffer(0)]], uint i [[thread_position_in_grid]]) { x[i]+=1; }",
        ),
        cuda: None,
        subgroup_size: None,
    };
    let kernel = device.compile(&source, "increment").unwrap();
    let buffer = device.allocate(4096).unwrap();
    // Also cross the sample-buffer capacity to verify rollover and aggregation.
    for _ in 0..2051 {
        // SAFETY: 1024 threads each update one u32 in a 4096-byte buffer.
        unsafe {
            device
                .dispatch(&kernel, &[Argument::Buffer(&buffer)], 1024, 128)
                .unwrap()
        };
    }
    assert!(
        device
            .read(&buffer)
            .unwrap()
            .chunks_exact(4)
            .all(|v| v == 2051u32.to_le_bytes())
    );
    let stat = device.kernel_stats().unwrap()["timestamp-test::increment"];
    assert_eq!(stat.dispatches, 2051);
    assert!(stat.gpu_seconds > 0. && stat.gpu_seconds.is_finite());
    assert!(stat.gpu_seconds <= device.stats().gpu_seconds * 1.1);
    assert_eq!(device.stats().encoders, 2051);
    assert_eq!(device.stats().command_buffers, 2);
    device.reset_stats();
    assert!(device.kernel_stats().unwrap().is_empty());
    // SAFETY: same buffer and dispatch geometry as above.
    unsafe {
        device
            .dispatch(&kernel, &[Argument::Buffer(&buffer)], 1024, 128)
            .unwrap()
    };
    device.synchronize().unwrap();
    assert_eq!(
        device.kernel_stats().unwrap()["timestamp-test::increment"].dispatches,
        1
    );
    device.set_kernel_profiling(false).unwrap();
    assert!(device.kernel_stats().is_none());
    device.reset_stats();
    // SAFETY: same buffer and dispatch geometry as above.
    unsafe {
        device
            .dispatch(&kernel, &[Argument::Buffer(&buffer)], 1024, 128)
            .unwrap()
    };
    assert_eq!(
        device.read_prefix(&buffer, 4).unwrap(),
        2053u32.to_le_bytes()
    );
    assert_eq!(device.stats().encoders, 1);
}
