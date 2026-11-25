//! Bonsai 2.5: packed affine weights, signed Hadamard transforms, and hybrid attention.
pub mod chat;
pub mod config;
pub mod kernels;
pub mod model;
pub use config::Config;
pub use model::{Bonsai, DEFAULT_PREFILL_BATCH};
