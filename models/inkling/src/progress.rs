/// Progress through a prompt, including optional image processing.
///
/// Updates report completed GPU work at existing synchronization points. The
/// final prompt chunk and output projection finish together at `Complete`.
/// Observers run synchronously and should return promptly; they do not change
/// kernel scheduling or add GPU synchronization.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrefillProgress {
    LoadingImageEncoder,
    ImageEncoderLoaded,
    ImageEncoding {
        completed_layers: usize,
        total_layers: usize,
    },
    ImageEncoded,
    Prompt {
        completed_tokens: usize,
        total_tokens: usize,
    },
    Complete,
}
