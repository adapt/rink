//! Metal command-stream implementation. No model math lives in this crate.
#[cfg(target_os = "macos")]
mod device;
#[cfg(target_os = "macos")]
pub use device::*;

#[cfg(not(target_os = "macos"))]
pub fn unavailable() -> rink_core::Result<()> {
    rink_core::bail!("Metal requires macOS and a Metal-capable GPU")
}
