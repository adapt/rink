//! Small byte tokenizer for protocol tests; no checkpoint download is needed.
use tokenizers::{AddedToken, Tokenizer, models::bpe::BPE, pre_tokenizers::byte_level::ByteLevel};

pub fn tokenizer() -> Tokenizer {
    let mut alphabet: Vec<_> = ByteLevel::alphabet().into_iter().collect();
    alphabet.sort_unstable();
    let vocab: [(String, u32); 256] = alphabet
        .into_iter()
        .enumerate()
        .map(|(i, c)| (c.to_string(), i as u32))
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();
    let mut tokenizer = Tokenizer::new(
        BPE::builder()
            .vocab_and_merges(vocab, vec![])
            .build()
            .unwrap(),
    );
    tokenizer.with_pre_tokenizer(Some(ByteLevel::new(false, false, false)));
    tokenizer.with_decoder(Some(ByteLevel::new(false, false, false)));
    tokenizer.add_special_tokens(
        &[
            "message_user",
            "message_model",
            "message_system",
            "message_tool",
            "content_text",
            "content_thinking",
            "content_xml",
            "content_invoke_tool_json",
            "content_invoke_tool_text",
            "content_image",
            "unused_200054",
            "end_message",
            "content_model_end_sampling",
        ]
        .map(|s| AddedToken::from(format!("<|{s}|>"), true)),
    );
    tokenizer
}

pub fn wire(tokenizer: &Tokenizer, text: &str) -> Vec<u32> {
    let mut tokenizer = tokenizer.clone();
    tokenizer.set_encode_special_tokens(false);
    tokenizer.encode(text, false).unwrap().get_ids().to_vec()
}
