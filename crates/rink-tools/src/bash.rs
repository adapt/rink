use crate::{ContentPart, Tool, ToolResult};
use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    io::{self, Read},
    os::{fd::AsRawFd, unix::process::CommandExt},
    path::{Path, PathBuf},
    process::{Child, Command, ExitStatus, Stdio},
    sync::atomic::{AtomicBool, Ordering},
    time::{Duration, Instant},
};

pub struct Bash {
    cwd: PathBuf,
    timeout: Duration,
    output_bytes: usize,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Arguments {
    command: String,
    timeout_ms: Option<u64>,
}

impl Bash {
    pub fn new(cwd: impl AsRef<Path>, timeout: Duration, output_bytes: usize) -> Result<Self> {
        let cwd = cwd
            .as_ref()
            .canonicalize()
            .context("resolving tool directory")?;
        ensure!(cwd.is_dir(), "tool working directory must be a directory");
        ensure!(!timeout.is_zero(), "tool timeout must be positive");
        ensure!(
            (256..=1_048_576).contains(&output_bytes),
            "tool output limit must be 256..=1048576 bytes"
        );
        Ok(Self {
            cwd,
            timeout,
            output_bytes,
        })
    }

    pub fn cwd(&self) -> &Path {
        &self.cwd
    }
}

impl Tool for Bash {
    fn name(&self) -> &'static str {
        "bash"
    }

    fn definition(&self) -> Value {
        json!({
            "name":"bash", "type":"function",
            "description":format!("Run a non-interactive Bash command with direct host access in {}. Each call starts a fresh shell; cd and environment changes do not persist. Background children stop when the call ends. Returns stdout, stderr, exit status and timeout information. Output is limited to {} bytes; timeout is at most {} ms.", self.cwd.display(), self.output_bytes, self.timeout.as_millis()),
            "parameters":{
                "type":"object", "properties":{
                    "command":{"type":"string", "description":"The Bash script to execute."},
                    "timeout_ms":{"type":"integer", "minimum":1, "maximum":self.timeout.as_millis() as u64, "description":"Optional timeout in milliseconds."}
                }, "required":["command"], "additionalProperties":false
            }
        })
    }

    fn execute(&self, args: &Value, cancelled: &AtomicBool) -> Result<ToolResult> {
        let args: Arguments =
            serde_json::from_value(args.clone()).context("invalid bash arguments")?;
        ensure!(
            !args.command.trim().is_empty()
                && args.command.len() <= 65_536
                && !args.command.contains('\0'),
            "command must contain 1..=65536 bytes and no NUL"
        );
        let timeout = args
            .timeout_ms
            .map(Duration::from_millis)
            .unwrap_or(self.timeout);
        ensure!(
            !timeout.is_zero() && timeout <= self.timeout,
            "timeout_ms exceeds the configured limit or is zero"
        );
        if cancelled.load(Ordering::Relaxed) {
            return Ok(ToolResult {
                content: vec![ContentPart::Text(
                    json!({"cancelled":true,"error":"command was not started"}).to_string(),
                )],
                cancelled: true,
            });
        }
        let start = Instant::now();
        let child = Command::new("/bin/bash")
            .args(["--noprofile", "--norc", "-c", &args.command])
            .env_remove("BASH_ENV")
            .current_dir(&self.cwd)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .process_group(0)
            .spawn()
            .context("starting bash")?;
        let mut process = Process {
            child,
            stopped: false,
        };
        let mut stdout = process.child.stdout.take().context("missing stdout pipe")?;
        let mut stderr = process.child.stderr.take().context("missing stderr pipe")?;
        nonblocking(&stdout)?;
        nonblocking(&stderr)?;
        let mut out = Capture::new(self.output_bytes / 2);
        let mut err = Capture::new(self.output_bytes - self.output_bytes / 2);
        let mut timed_out = false;
        let mut interrupted = false;
        let status = loop {
            let activity = out.drain(&mut stdout)? | err.drain(&mut stderr)?;
            if let Some(status) = process.child.try_wait()? {
                process.stop()?;
                break status;
            }
            interrupted = cancelled.load(Ordering::Relaxed);
            timed_out = start.elapsed() >= timeout;
            if interrupted || timed_out {
                break process.stop()?;
            }
            if !activity {
                std::thread::sleep(Duration::from_millis(5));
            }
        };
        // Drain already buffered bytes, without waiting on descendants that may
        // have detached themselves from this process group or inherited pipes.
        let drain_start = Instant::now();
        while out.drain(&mut stdout)? | err.drain(&mut stderr)? {
            if drain_start.elapsed() >= Duration::from_millis(100) {
                out.truncated = true;
                err.truncated = true;
                break;
            }
        }
        use std::os::unix::process::ExitStatusExt;
        Ok(ToolResult {
            content: vec![ContentPart::Text(
                json!({
                    "stdout":String::from_utf8_lossy(&out.bytes),
                    "stderr":String::from_utf8_lossy(&err.bytes),
                    "exit_code":status.code(), "signal":status.signal(),
                    "timed_out":timed_out, "cancelled":interrupted,
                    "stdout_truncated":out.truncated, "stderr_truncated":err.truncated,
                    "elapsed_ms":start.elapsed().as_millis()
                })
                .to_string(),
            )],
            cancelled: interrupted,
        })
    }
}

fn nonblocking(pipe: &impl AsRawFd) -> Result<()> {
    // SAFETY: the pipe owns a live descriptor for the duration of both calls.
    unsafe {
        let flags = libc::fcntl(pipe.as_raw_fd(), libc::F_GETFL);
        if flags < 0 || libc::fcntl(pipe.as_raw_fd(), libc::F_SETFL, flags | libc::O_NONBLOCK) < 0 {
            return Err(io::Error::last_os_error()).context("setting nonblocking tool output");
        }
    }
    Ok(())
}

struct Capture {
    bytes: Vec<u8>,
    limit: usize,
    truncated: bool,
}
impl Capture {
    fn new(limit: usize) -> Self {
        Self {
            bytes: Vec::new(),
            limit,
            truncated: false,
        }
    }
    fn drain(&mut self, input: &mut impl Read) -> io::Result<bool> {
        let mut buffer = [0; 8192];
        let mut activity = false;
        // A continuously writing process must not postpone timeout/cancel checks.
        for _ in 0..32 {
            match input.read(&mut buffer) {
                Ok(0) => break,
                Ok(n) => {
                    activity = true;
                    let keep = n.min(self.limit - self.bytes.len());
                    self.bytes.extend_from_slice(&buffer[..keep]);
                    self.truncated |= keep < n;
                }
                Err(e) if e.kind() == io::ErrorKind::WouldBlock => break,
                Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(e),
            }
        }
        Ok(activity)
    }
}

struct Process {
    child: Child,
    stopped: bool,
}
impl Process {
    fn stop(&mut self) -> io::Result<ExitStatus> {
        // SAFETY: process_group(0) creates a separate group with this child PID.
        // Kill the group, including ordinary background children, then reap it.
        unsafe { libc::kill(-(self.child.id() as libc::pid_t), libc::SIGKILL) };
        let status = self.child.wait()?;
        self.stopped = true;
        Ok(status)
    }
}
impl Drop for Process {
    fn drop(&mut self) {
        if !self.stopped {
            let _ = self.stop();
        }
    }
}
