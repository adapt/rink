use rink_bonsai::Config;

fn fixture() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny")
}

#[test]
fn validates_the_complete_artifact_contract() {
    let c = Config::load(fixture().join("config.json")).unwrap();
    assert_eq!(c.projections().len(), 15);
    assert_eq!(
        c.text_config.cache_bytes(512).unwrap(),
        512 * 128 * 4 + 384 * 128 * 4 + 640 * 3 * 2
    );
    assert!(c.text_config.cache_bytes(0).is_err());
    assert!(c.text_config.cache_bytes(513).is_err());
    for change in 0..10 {
        let mut c = c.clone();
        match change {
            0 => c.quantization.bits = 4,
            1 => c.gdn_activation_layout = "ungrouped".into(),
            2 => c.modules.push(c.modules[0].clone()),
            3 => c.modules[0].block = 2048,
            4 => c.modules[0].embedding = !c.modules[0].embedding,
            5 => c.text_config.layer_types.reverse(),
            6 => c.text_config.linear_num_key_heads = 2,
            7 => c.text_config.rope_parameters.rope_type = "yarn".into(),
            8 => c.text_config.rms_norm_eps = f32::NAN,
            _ => c.text_config.full_attention_interval = 0,
        }
        assert!(c.validate().is_err(), "invalid variant {change} accepted");
    }
}
