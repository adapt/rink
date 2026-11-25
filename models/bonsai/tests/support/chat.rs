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
    let mut t = Tokenizer::new(
        BPE::builder()
            .vocab_and_merges(vocab, vec![])
            .build()
            .unwrap(),
    );
    t.with_pre_tokenizer(Some(ByteLevel::new(false, false, false)));
    t.with_decoder(Some(ByteLevel::new(false, false, false)));
    t.add_special_tokens(
        &["<|im_start|>", "<|im_end|>", "<|endoftext|>"].map(|s| AddedToken::from(s, true)),
    );
    t.add_tokens(
        &[
            "<think>",
            "</think>",
            "<tool_call>",
            "</tool_call>",
            "<tool_response>",
            "</tool_response>",
        ]
        .map(|s| AddedToken::from(s, false)),
    );
    t
}
pub fn wire(t: &Tokenizer, s: &str) -> Vec<u32> {
    t.encode(s, false).unwrap().get_ids().to_vec()
}
