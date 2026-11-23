use anyhow::{Context, Result, ensure};
use std::sync::atomic::{AtomicBool, Ordering};

static ACTIVE: AtomicBool = AtomicBool::new(false);
static CANCELLED: AtomicBool = AtomicBool::new(false);

extern "C" fn interrupt(_: libc::c_int) {
    CANCELLED.store(true, Ordering::Relaxed);
}

/// Temporarily catch Ctrl-C while tools run. The CLI's prior handler is restored
/// before returning to input or generation. Only one guard can exist at a time.
pub struct Interrupt {
    previous: libc::sigaction,
}

impl Interrupt {
    pub fn install() -> Result<Self> {
        ensure!(
            !ACTIVE.swap(true, Ordering::SeqCst),
            "an interrupt handler is already active"
        );
        CANCELLED.store(false, Ordering::Relaxed);
        // SAFETY: zeroed sigaction values are initialized below and passed to
        // libc with valid pointers. The handler only stores to a lock-free atomic.
        let result = unsafe {
            let mut action: libc::sigaction = std::mem::zeroed();
            let mut previous: libc::sigaction = std::mem::zeroed();
            action.sa_sigaction = interrupt as *const () as libc::sighandler_t;
            libc::sigemptyset(&mut action.sa_mask);
            if libc::sigaction(libc::SIGINT, &action, &mut previous) != 0 {
                Err(std::io::Error::last_os_error())
            } else {
                Ok(Self { previous })
            }
        };
        if result.is_err() {
            ACTIVE.store(false, Ordering::SeqCst);
        }
        result.context("installing tool interrupt handler")
    }

    pub fn flag(&self) -> &'static AtomicBool {
        &CANCELLED
    }
}

impl Drop for Interrupt {
    fn drop(&mut self) {
        // SAFETY: restore the exact action saved by this exclusive guard.
        unsafe { libc::sigaction(libc::SIGINT, &self.previous, std::ptr::null_mut()) };
        ACTIVE.store(false, Ordering::SeqCst);
    }
}
