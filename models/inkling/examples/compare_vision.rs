//! Compare all image features with scripts/vision_reference.py's F32 CPU oracle.
//! compare_vision MODEL_DIR REFERENCE_DIR
#[cfg(target_os = "macos")]
fn main() -> anyhow::Result<()> {
    use anyhow::ensure;
    use rink_core::Accelerator;
    use rink_inkling::{
        Config, Image, checkpoint::Checkpoint, kernels::Kernels, vision::VisionEncoder,
    };
    use rink_metal::Metal;
    use std::{path::Path, time::Instant};

    let args: Vec<_> = std::env::args().collect();
    ensure!(
        args.len() == 3,
        "usage: compare_vision MODEL_DIR REFERENCE_DIR"
    );
    let directory = Path::new(&args[1]);
    let config = Config::load(directory.join("config.json"))?;
    let checkpoint = Checkpoint::open(directory)?;
    let mut device = Metal::new()?;
    let kernels = Kernels::compile(&mut device)?;
    let encoder = VisionEncoder::load(
        &mut device,
        &checkpoint,
        config.vision_config.as_ref().unwrap(),
    )?;
    let mut cases: Vec<_> = std::fs::read_dir(&args[2])?.collect::<Result<_, _>>()?;
    cases.sort_by_key(|e| e.path());
    let mut tested = 0;
    for case in cases {
        let path = case.path();
        if path.extension().is_none_or(|e| e != "png") {
            continue;
        }
        let image = Image::open(&path)?;
        let bytes = std::fs::read(path.with_extension("safetensors"))?;
        let reference = safetensors::SafeTensors::deserialize(&bytes)?;
        let values = |name| -> anyhow::Result<Vec<f32>> {
            let t = reference.tensor(name)?;
            ensure!(
                t.dtype() == safetensors::Dtype::F32,
                "expected F32 reference"
            );
            Ok(t.data()
                .chunks_exact(4)
                .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
                .collect())
        };
        ensure!(
            image.patches() == values("patches")?,
            "{}: preprocessing differs",
            path.display()
        );
        let start = Instant::now();
        let encoded = encoder.encode(&mut device, &kernels, &image)?;
        let actual: Vec<_> = device
            .read(&encoded)?
            .chunks_exact(4)
            .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
            .collect();
        let elapsed = start.elapsed().as_secs_f64();
        let expected = values("features")?;
        ensure!(actual.len() == expected.len(), "feature shape mismatch");
        let mut max_error = 0f64;
        let mut squared_error = 0f64;
        let mut squared_ref = 0f64;
        let mut violations = 0;
        for (&a, &b) in actual.iter().zip(&expected) {
            ensure!(a.is_finite() && b.is_finite(), "nonfinite vision feature");
            let error = (f64::from(a) - f64::from(b)).abs();
            max_error = max_error.max(error);
            squared_error += error * error;
            squared_ref += f64::from(b).powi(2);
            violations += usize::from(error > 0.00002 + 0.00002 * f64::from(b).abs());
        }
        println!(
            "{}",
            serde_json::json!({"image": path.file_name().unwrap().to_string_lossy(),
            "features": actual.len(), "seconds": elapsed, "max_abs_error": max_error,
            "rmse": (squared_error / actual.len() as f64).sqrt(),
            "relative_rms_error": (squared_error / squared_ref).sqrt(), "tolerance_violations": violations})
        );
        ensure!(violations == 0, "vision feature tolerance exceeded");
        tested += 1;
    }
    ensure!(tested > 0, "no PNG references found");
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("requires macOS with Metal");
}
