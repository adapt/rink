use rink_inkling::Config;

fn official() -> Config {
    serde_json::from_str(include_str!("fixtures/official-config.json")).unwrap()
}

#[test]
fn released_configuration_has_48_independent_cache_visits() {
    let config = official();
    config.validate().unwrap();
    let schedule = config.text_config.schedule();
    assert_eq!(schedule.len(), 48);
    assert_eq!(
        schedule.iter().map(|v| v.layer).collect::<Vec<_>>(),
        (0..6)
            .chain(6..18)
            .chain(6..18)
            .chain(6..18)
            .chain(18..24)
            .collect::<Vec<_>>()
    );
    assert!(schedule.iter().enumerate().all(|(i, v)| i == v.cache));
    assert_eq!(schedule[18].pass, 1);
    assert_eq!(schedule[30].pass, 2);
}

#[test]
fn rejects_incompatible_architectures_and_invalid_dimensions() {
    for mutate in [
        |c: &mut Config| c.text_config.loop_config.core_start = 7,
        |c: &mut Config| c.text_config.num_key_value_heads = 0,
        |c: &mut Config| c.text_config.num_experts_per_tok = 129,
        |c: &mut Config| c.text_config.rms_norm_eps = f32::NAN,
        |c: &mut Config| c.text_config.n_layers_per_attn_res_block = 0,
        |c: &mut Config| c.model_type = "llama".into(),
        |c: &mut Config| c.eos_token_id = 900000,
        |c: &mut Config| c.text_config.attn_res_normalize_values = false,
    ] {
        let mut c = official();
        mutate(&mut c);
        assert!(c.validate().is_err());
    }
}

#[test]
fn chat_matches_single_turn_release_template() {
    assert_eq!(
        rink_inkling::chat_prompt("Hello", 0.9).unwrap(),
        "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|><|message_user|><|content_text|>Hello<|end_message|>"
    );
    assert!(rink_inkling::chat_prompt("Hello", f32::NAN).is_err());
}
