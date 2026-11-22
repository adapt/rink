use rink_inkling::{
    Image,
    image::{IMAGE_TOKEN, image_prompt, image_user_message},
};
use std::path::Path;

fn fixture() -> &'static Path {
    Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/vision"
    ))
}

#[test]
fn preprocessing_matches_reference_and_validates_inputs() {
    let image = Image::open(fixture().join("input.png")).unwrap();
    assert_eq!(
        (image.width(), image.height(), image.token_count()),
        (241, 81, 21)
    );
    let bytes = std::fs::read(fixture().join("expected.safetensors")).unwrap();
    let reference = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    assert_eq!(
        bytemuck::cast_slice::<f32, u8>(&image.patches()),
        reference.tensor("patches").unwrap().data()
    );
    assert!(Image::from_rgba(0, 1, vec![]).is_err());
    assert!(Image::from_rgba(2, 2, vec![0; 15]).is_err());
    assert!(Image::open(fixture().join("config.json")).is_err());
    assert!(Image::open(fixture().join("unsupported-16bit.png")).is_err());
}

#[test]
fn image_dimension_limit_applies_on_each_axis() {
    // Narrow images exercise both boundaries without allocating a 16K square.
    for (width, height) in [(16_384, 1), (1, 16_384), (16_385, 1), (1, 16_385)] {
        let accepted = width <= 16_384 && height <= 16_384;
        let rgba = Image::from_rgba(width, height, vec![255; (width * height * 4) as usize]);
        assert_eq!(rgba.is_ok(), accepted);
        if !accepted {
            assert!(rgba.unwrap_err().to_string().contains("1..=16384"));
        }
        for extension in ["png", "jpg"] {
            let path = std::env::temp_dir().join(format!(
                "rink-image-limit-{}-{width}-{height}.{extension}",
                std::process::id()
            ));
            image::RgbImage::new(width, height).save(&path).unwrap();
            let decoded = Image::open(&path);
            std::fs::remove_file(path).unwrap();
            if accepted {
                let decoded = decoded.unwrap();
                assert_eq!(
                    (decoded.width(), decoded.height()),
                    (width as usize, height as usize)
                );
                assert_eq!(decoded.token_count(), 410);
            } else {
                assert!(
                    decoded
                        .unwrap_err()
                        .to_string()
                        .contains("maximum dimension is 16384")
                );
            }
        }
    }
}

#[test]
fn padding_is_negative_one_before_normalization_and_alpha_uses_checkerboard() {
    let image = Image::from_rgba(9, 1, vec![0; 9 * 4]).unwrap();
    let patches = image.patches();
    assert_eq!(patches.len(), 4800);
    assert_eq!(patches[0], (204. - 127.5) / 127.5);
    assert_eq!(patches[8 * 3], (153. - 127.5) / 127.5);
    assert_eq!(patches[9 * 3], (-1. - 127.5) / 127.5);
    assert_eq!(patches[1200], (-1. - 127.5) / 127.5);
}

#[test]
fn jpeg_loader_applies_exif_orientation() {
    let jpeg = Image::open(fixture().join("oriented.jpg")).unwrap();
    let expected = Image::open(fixture().join("oriented-expected.png")).unwrap();
    assert_eq!((jpeg.width(), jpeg.height()), (7, 11));
    // JPEG decoders may differ by a quantization step. Compare against Pillow's
    // independently decoded and oriented PNG, including the padded edges.
    for (a, b) in jpeg.patches().iter().zip(expected.patches()) {
        assert!((a - b).abs() <= 2. / 127.5);
    }
}

#[test]
fn image_template_uses_separate_user_parts() {
    let message = image_user_message("What is this?", 2);
    assert_eq!(
        message,
        format!(
            "<|message_user|><|content_image|>{IMAGE_TOKEN}{IMAGE_TOKEN}<|end_message|><|message_user|><|content_text|>What is this?<|end_message|>"
        )
    );
    let prompt = image_prompt("What is this?", 2, 0.9).unwrap();
    assert!(prompt.ends_with(&message));
    assert!(prompt.starts_with(
        "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>"
    ));
    assert!(image_prompt("", 1, f32::NAN).is_err());
}

#[test]
fn rejects_unsupported_vision_configurations() {
    use rink_inkling::{Config, config::VisionConfig};
    let config = Config::load(fixture().join("config.json")).unwrap();
    for mutate in [
        |c: &mut VisionConfig| c.num_attention_heads = 0,
        |c: &mut VisionConfig| c.hidden_size = 33,
        |c: &mut VisionConfig| c.projector_bias = true,
        |c: &mut VisionConfig| c.hidden_act = "gelu_new".into(),
        |c: &mut VisionConfig| {
            c.rope_parameters =
                Some(serde_json::json!({"rope_type": "axial", "rope_theta": 500000}))
        },
        |c: &mut VisionConfig| c.intermediate_size = Some(65),
        |c: &mut VisionConfig| c.num_channels = 4,
        |c: &mut VisionConfig| c.layer_norm_eps = f32::NAN,
        |c: &mut VisionConfig| c.native_patch_size = 16,
    ] {
        let mut c = config.vision_config.clone().unwrap();
        mutate(&mut c);
        assert!(c.validate().is_err());
    }
}

#[cfg(target_os = "macos")]
mod gpu {
    use super::*;
    use rink_core::{Accelerator, AutoregressiveModel};
    use rink_inkling::{
        Config, Inkling, KvCachePrecision, PrefillProgress,
        checkpoint::Checkpoint,
        kernels::{Kernels, Params},
        vision::VisionEncoder,
    };
    use rink_metal::Metal;

    fn close(a: &[f32], b: &[f32], tolerance: f32) {
        assert_eq!(a.len(), b.len());
        for (i, (&a, &b)) in a.iter().zip(b).enumerate() {
            assert!(
                a.is_finite() && b.is_finite() && (a - b).abs() < tolerance,
                "index {i}: {a} != {b} (tolerance {tolerance})"
            );
        }
    }
    fn floats(data: &[u8]) -> Vec<f32> {
        data.chunks_exact(4)
            .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
            .collect()
    }

    #[test]
    #[ignore = "requires a Metal GPU"]
    fn complete_encoder_matches_cpu_reference() {
        let config = Config::load(fixture().join("config.json")).unwrap();
        let checkpoint = Checkpoint::open(fixture()).unwrap();
        let mut d = Metal::new().unwrap();
        let kernels = Kernels::compile(&mut d).unwrap();
        let encoder =
            VisionEncoder::load(&mut d, &checkpoint, config.vision_config.as_ref().unwrap())
                .unwrap();
        let image = Image::open(fixture().join("input.png")).unwrap();
        let actual = encoder.encode(&mut d, &kernels, &image).unwrap();
        let bytes = std::fs::read(fixture().join("expected.safetensors")).unwrap();
        let reference = safetensors::SafeTensors::deserialize(&bytes).unwrap();
        close(
            &floats(&d.read(&actual).unwrap()),
            &floats(reference.tensor("features").unwrap().data()),
            0.00001,
        );
    }

    #[derive(serde::Deserialize)]
    struct TextReference {
        tokens: Vec<u32>,
        prefill: Vec<f32>,
        continuation: Vec<Vec<f32>>,
        turn2: Vec<u32>,
        second: Vec<f32>,
        after: Vec<Vec<f32>>,
    }

    #[test]
    #[ignore = "requires a Metal GPU"]
    fn mixed_prefill_crosses_chunks_preserves_history_and_resets() {
        for precision in [KvCachePrecision::F16, KvCachePrecision::F32] {
            let reference: TextReference = serde_json::from_slice(
                &std::fs::read(fixture().join(format!("text-{precision}.json"))).unwrap(),
            )
            .unwrap();
            let image = Image::open(fixture().join("input.png")).unwrap();
            let mut model =
                Inkling::load_with_kv_cache(Metal::new().unwrap(), fixture(), 128, precision)
                    .unwrap();
            // Rejected placeholders and budgets leave the model usable.
            for bad in [
                vec![1, 2],
                vec![13; 20],
                vec![13; 22],
                vec![13; 129],
                vec![13, 1, 13],
            ] {
                assert!(model.prefill_image(&bad, &image).is_err());
                assert_eq!(model.position(), 0);
            }
            for _ in 0..2 {
                close(
                    &model.prefill_image(&reference.tokens, &image).unwrap(),
                    &reference.prefill,
                    0.00004,
                );
                assert_eq!(model.position(), reference.tokens.len());
                for (token, expected) in [4, 5, 6].into_iter().zip(&reference.continuation) {
                    close(&model.step(token).unwrap(), expected, 0.00004);
                }
                close(
                    &model.prefill_image(&reference.turn2, &image).unwrap(),
                    &reference.second,
                    0.00004,
                );
                for (token, expected) in [7, 8].into_iter().zip(&reference.after) {
                    close(&model.step(token).unwrap(), expected, 0.00004);
                }
                model.reset().unwrap();
            }
            let expected = rink_core::greedy(&reference.prefill).unwrap();
            assert_eq!(
                model
                    .prefill_image_greedy(&reference.tokens, &image)
                    .unwrap(),
                expected
            );
        }
    }

    #[test]
    #[ignore = "requires a Metal GPU"]
    fn progress_preserves_results_and_gpu_work() {
        use PrefillProgress::*;
        let work = |model: &Inkling<Metal>| {
            let s = model.accelerator().stats();
            (
                s.dispatches,
                s.encoders,
                s.command_buffers,
                s.readback_bytes,
            )
        };
        for precision in [KvCachePrecision::F16, KvCachePrecision::F32] {
            let reference: TextReference = serde_json::from_slice(
                &std::fs::read(fixture().join(format!("text-{precision}.json"))).unwrap(),
            )
            .unwrap();
            let image = Image::open(fixture().join("input.png")).unwrap();
            let mut model =
                Inkling::load_with_kv_cache(Metal::new().unwrap(), fixture(), 128, precision)
                    .unwrap();
            let mut events = Vec::new();
            // Rejected input must not report started or completed work.
            assert!(
                model
                    .prefill_greedy_with_progress(&[], |e| events.push(e))
                    .is_err()
            );
            assert!(
                model
                    .prefill_image_greedy_with_progress(&[1], &image, |e| events.push(e))
                    .is_err()
            );
            assert!(events.is_empty());
            assert_eq!(model.position(), 0);

            model.accelerator_mut().reset_stats();
            let first = model
                .prefill_image_greedy_with_progress(&reference.tokens, &image, |e| events.push(e))
                .unwrap();
            let observed_work = work(&model);
            assert_eq!(first, rink_core::greedy(&reference.prefill).unwrap());
            let total_tokens = reference.tokens.len();
            assert_eq!(
                events,
                [
                    LoadingImageEncoder,
                    ImageEncoderLoaded,
                    ImageEncoding {
                        completed_layers: 0,
                        total_layers: 2
                    },
                    ImageEncoding {
                        completed_layers: 1,
                        total_layers: 2
                    },
                    ImageEncoding {
                        completed_layers: 2,
                        total_layers: 2
                    },
                    ImageEncoded,
                    Prompt {
                        completed_tokens: 0,
                        total_tokens
                    },
                    Prompt {
                        completed_tokens: 32,
                        total_tokens
                    },
                    Complete,
                ]
            );
            let observed_logits = model.step(4).unwrap();
            model.reset().unwrap();
            model.accelerator_mut().reset_stats();
            assert_eq!(
                model
                    .prefill_image_greedy(&reference.tokens, &image)
                    .unwrap(),
                first
            );
            assert_eq!(work(&model), observed_work);
            assert_eq!(model.step(4).unwrap(), observed_logits);

            // Reusing the encoder skips loading. A short prompt completes only
            // after selection; it has no intermediate synchronized chunk.
            events.clear();
            model.reset().unwrap();
            model
                .prefill_image_greedy_with_progress(&reference.turn2, &image, |e| events.push(e))
                .unwrap();
            assert!(matches!(
                events[0],
                ImageEncoding {
                    completed_layers: 0,
                    ..
                }
            ));
            let prompt_events: Vec<_> = events
                .iter()
                .copied()
                .filter(|e| matches!(e, Prompt { .. } | Complete))
                .collect();
            assert_eq!(
                prompt_events,
                [
                    Prompt {
                        completed_tokens: 0,
                        total_tokens: reference.turn2.len()
                    },
                    Complete
                ]
            );

            let tokens = vec![1; 65];
            events.clear();
            model.reset().unwrap();
            model.accelerator_mut().reset_stats();
            let first = model
                .prefill_greedy_with_progress(&tokens, |e| events.push(e))
                .unwrap();
            let observed_work = work(&model);
            assert_eq!(
                events,
                [
                    Prompt {
                        completed_tokens: 0,
                        total_tokens: 65
                    },
                    Prompt {
                        completed_tokens: 32,
                        total_tokens: 65
                    },
                    Prompt {
                        completed_tokens: 64,
                        total_tokens: 65
                    },
                    Complete,
                ]
            );
            let observed_logits = model.step(4).unwrap();
            model.reset().unwrap();
            model.accelerator_mut().reset_stats();
            assert_eq!(model.prefill_greedy(&tokens).unwrap(), first);
            assert_eq!(work(&model), observed_work);
            assert_eq!(model.step(4).unwrap(), observed_logits);
        }
    }

    #[test]
    #[ignore = "requires a Metal GPU"]
    fn noncausal_attention_matches_f64_for_tail_tiles_and_all_head_widths() {
        let mut d = Metal::new().unwrap();
        let kernels = Kernels::compile(&mut d).unwrap();
        for n in [1usize, 5, 17, 33, 65] {
            for dim in [8usize, 16, 64, 128] {
                let heads = 2;
                let length = n * heads * dim;
                let values = |shift: usize| -> Vec<f32> {
                    (0..length)
                        .map(|i| (((i * 19 + shift) % 101) as f32 - 50.) / 17.)
                        .collect()
                };
                let q = values(1);
                let k = values(13);
                let v = values(29);
                let qb = d.upload(bytemuck::cast_slice(&q)).unwrap();
                let kb = d.upload(bytemuck::cast_slice(&k)).unwrap();
                let vb = d.upload(bytemuck::cast_slice(&v)).unwrap();
                let out = d
                    .upload(bytemuck::cast_slice(&vec![f32::NAN; length + 32]))
                    .unwrap();
                // SAFETY: F32 inputs have n*heads*dim elements, output has guard space.
                unsafe {
                    kernels
                        .run(
                            &mut d,
                            "vision_attention",
                            &[&out, &qb, &kb, &vb],
                            Params::new(&[n, heads, dim], &[]),
                            n.div_ceil(8) * heads * 32,
                            32,
                        )
                        .unwrap();
                }
                let actual = floats(&d.read(&out).unwrap());
                assert!(actual[length..].iter().all(|v| v.is_nan()));
                let mut expected = vec![0f32; length];
                for i in 0..n {
                    for head in 0..heads {
                        let scores: Vec<f64> = (0..n)
                            .map(|j| {
                                (0..dim)
                                    .map(|c| {
                                        f64::from(q[(i * heads + head) * dim + c])
                                            * f64::from(k[(j * heads + head) * dim + c])
                                    })
                                    .sum::<f64>()
                                    / (dim as f64).sqrt()
                            })
                            .collect();
                        let peak = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                        let probs: Vec<_> = scores.iter().map(|s| (s - peak).exp()).collect();
                        let den: f64 = probs.iter().sum();
                        for c in 0..dim {
                            expected[(i * heads + head) * dim + c] = (probs
                                .iter()
                                .enumerate()
                                .map(|(j, p)| p * f64::from(v[(j * heads + head) * dim + c]))
                                .sum::<f64>()
                                / den)
                                as f32;
                        }
                    }
                }
                // Large test scores amplify ordinary F32 dot-product rounding
                // in the softmax. Keep a separate F64 oracle and an absolute
                // bound of 4e-5 for input values up to about 3.
                close(&actual[..length], &expected, 0.00004);
                if dim == 64 {
                    // SAFETY: the native specialization uses the same buffers,
                    // with eight SIMD groups and exactly 64 channels per head.
                    unsafe {
                        kernels
                            .run(
                                &mut d,
                                "vision_attention_64x16",
                                &[&out, &qb, &kb, &vb],
                                Params::new(&[n, heads, dim], &[]),
                                n.div_ceil(16) * heads * 256,
                                256,
                            )
                            .unwrap();
                    }
                    let candidate = floats(&d.read(&out).unwrap());
                    assert!(candidate[length..].iter().all(|v| v.is_nan()));
                    assert_eq!(
                        bytemuck::cast_slice::<_, u8>(&candidate[..length]),
                        bytemuck::cast_slice::<_, u8>(&actual[..length])
                    );
                }
            }
        }
    }

    #[test]
    #[ignore = "requires a Metal GPU"]
    fn gelu_matches_f64_error_function_including_tails() {
        // macOS libm supplies the independent double precision oracle.
        unsafe extern "C" {
            fn erf(x: f64) -> f64;
        }
        let mut d = Metal::new().unwrap();
        let kernels = Kernels::compile(&mut d).unwrap();
        let x: Vec<f32> = (0..100_001).map(|i| -10. + i as f32 / 5000.).collect();
        let out = d.upload(bytemuck::cast_slice(&x)).unwrap();
        // SAFETY: output/bias alias is allowed when bias-present is zero.
        unsafe {
            kernels
                .run(
                    &mut d,
                    "vision_bias",
                    &[&out, &out],
                    Params::new(&[x.len(), 1, 1, 0], &[]),
                    x.len(),
                    128,
                )
                .unwrap();
        }
        let expected: Vec<f32> = x
            .iter()
            .map(|&v| {
                // SAFETY: libm erf accepts every finite f64 argument.
                (0.5 * f64::from(v) * (1. + unsafe { erf(f64::from(v) / 2f64.sqrt()) })) as f32
            })
            .collect();
        close(&floats(&d.read(&out).unwrap()), &expected, 0.000002);
    }
}
