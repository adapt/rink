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
    let mut other = Metal::new().unwrap();
    assert!(other.read(&output).is_err());
    assert!(device.allocate(0).is_err());
    assert!(device.allocate(usize::MAX).is_err());
}
