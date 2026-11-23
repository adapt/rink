use crate::{ContentPart, ImageFile, Tool, ToolResult};
use anyhow::{Context, Result, bail, ensure};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    fs::OpenOptions,
    io::{BufRead, BufReader, Read as _, Seek, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicBool, Ordering},
    time::{Duration, Instant},
};

const DEFAULT_LINES: usize = 200;
const MAX_LINES: usize = 2000;
const MAX_IMAGE_BYTES: u64 = 2 * 1024 * 1024 * 1024;

/// Read regular files using bounded buffers. No shell expansion or execution.
pub struct Read {
    cwd: PathBuf,
    home: Option<PathBuf>,
    output_bytes: usize,
    timeout: Duration,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Arguments {
    path: String,
    start_line: Option<usize>,
    max_lines: Option<usize>,
}

impl Read {
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
            home: std::env::var_os("HOME").map(PathBuf::from),
            output_bytes,
            timeout,
        })
    }

    fn path(&self, path: &str) -> Result<PathBuf> {
        ensure!(
            !path.is_empty() && path.len() <= 65_536 && !path.contains('\0'),
            "invalid file path"
        );
        if path == "~" || path.starts_with("~/") {
            Ok(self
                .home
                .as_ref()
                .context("home directory is unavailable")?
                .join(path.strip_prefix("~/").unwrap_or("")))
        } else {
            ensure!(
                !path.starts_with('~'),
                "use ~/ for your home directory; ~user is not supported"
            );
            Ok(self.cwd.join(path))
        }
    }

    fn read(&self, args: Arguments, cancelled: &AtomicBool) -> Result<ToolResult> {
        let started = Instant::now();
        let check = || -> Result<()> {
            ensure!(!cancelled.load(Ordering::Relaxed), "read cancelled");
            ensure!(started.elapsed() < self.timeout, "read timed out");
            Ok(())
        };
        check()?;
        let start_line = args.start_line.unwrap_or(1);
        let max_lines = args.max_lines.unwrap_or(DEFAULT_LINES);
        ensure!(start_line > 0, "start_line must be at least 1");
        ensure!(
            (1..=MAX_LINES).contains(&max_lines),
            "max_lines must be 1..=2000"
        );
        let path = self.path(&args.path)?;
        let mut options = OpenOptions::new();
        options.read(true);
        // Avoid blocking when a path refers (or changes) to a FIFO/device.
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.custom_flags(libc::O_NONBLOCK | libc::O_NOCTTY);
        }
        let file = options
            .open(&path)
            .with_context(|| format!("opening {}", path.display()))?;
        let metadata = file.metadata()?;
        ensure!(metadata.is_file(), "read supports regular files only");
        let mut reader = BufReader::new(file);
        let signature = reader.fill_buf()?;
        let image =
            signature.starts_with(b"\x89PNG\r\n\x1a\n") || signature.starts_with(b"\xff\xd8\xff");
        if image {
            ensure!(
                args.start_line.is_none() && args.max_lines.is_none(),
                "line ranges apply to text files, not images"
            );
            ensure!(
                metadata.len() <= MAX_IMAGE_BYTES,
                "encoded image exceeds 2 GiB"
            );
            let mut snapshot = tempfile::tempfile().context("creating image snapshot")?;
            let mut buffer = [0; 65_536];
            let mut size = 0u64;
            loop {
                check()?;
                let n = reader.read(&mut buffer)?;
                if n == 0 {
                    break;
                }
                size += n as u64;
                ensure!(size <= MAX_IMAGE_BYTES, "encoded image exceeds 2 GiB");
                snapshot.write_all(&buffer[..n])?;
            }
            snapshot.rewind()?;
            return Ok(ToolResult {
                content: vec![
                    ContentPart::Text(format!("Image file: {}", path.display())),
                    ContentPart::Image(ImageFile {
                        path,
                        file: snapshot,
                    }),
                ],
                cancelled: false,
            });
        }
        ensure!(
            !signature.starts_with(b"%PDF-"),
            "PDF reading is not supported; render the selected pages to PNG or JPEG first"
        );
        let page = text_page(&mut reader, start_line, max_lines, self.output_bytes, check)?;
        let mut text = format!("File: {}\n", path.display());
        if page.bytes.is_empty() {
            text.push_str("[No text at the requested line; end of file.]\n");
        } else {
            let mut bytes = page.bytes;
            // A byte limit can split the last UTF-8 character; retain only the
            // valid prefix in that case. Other invalid sequences are errors.
            if let Err(error) = std::str::from_utf8(&bytes) {
                if page.partial_line && error.error_len().is_none() {
                    bytes.truncate(error.valid_up_to());
                } else {
                    bail!("file is not UTF-8 text or a supported PNG/JPEG image");
                }
            }
            let body = std::str::from_utf8(&bytes)?;
            ensure!(
                !body
                    .chars()
                    .any(|c| c.is_control() && !matches!(c, '\n' | '\r' | '\t' | '\x0c')),
                "file contains binary data; only UTF-8 text and PNG/JPEG images are supported"
            );
            for (index, line) in body.split_inclusive('\n').enumerate() {
                text.push_str(&format!("{}: {}", start_line + index, line));
            }
            if !text.ends_with('\n') {
                text.push('\n');
            }
        }
        if page.partial_line {
            text.push_str(&format!("[Output truncated within line {} by the {}-byte text limit. Read from that line with a larger --tool-output-bytes setting to get the whole line.]", page.next_line, self.output_bytes));
        } else if page.more {
            text.push_str(&format!(
                "[More text. Continue with start_line={}.]",
                page.next_line
            ));
        } else {
            text.push_str("[End of file.]");
        }
        Ok(ToolResult::text(text))
    }
}

impl Tool for Read {
    fn name(&self) -> &'static str {
        "read"
    }

    fn definition(&self) -> Value {
        json!({
            "name":"read", "type":"function",
            "description":format!("Read a local UTF-8 text file or view a PNG/JPEG image with direct host access. Relative paths use {}; ~/ expands to your home directory. Images are placed directly in your visual context. For text, start_line is 1-based and max_lines defaults to 200; at most {} text bytes are returned. Output has line numbers and continuation information. Line ranges do not apply to images. Directories, PDFs and other binary formats are unsupported.", self.cwd.display(), self.output_bytes),
            "parameters":{
                "type":"object", "properties":{
                    "path":{"type":"string", "description":"Local file path, absolute or relative to the tool working directory."},
                    "start_line":{"type":"integer", "minimum":1, "description":"First text line to read (1-based). Omit for images."},
                    "max_lines":{"type":"integer", "minimum":1, "maximum":MAX_LINES, "description":"Maximum text lines, default 200. Omit for images."}
                }, "required":["path"], "additionalProperties":false
            }
        })
    }

    fn execute(&self, args: &Value, cancelled: &AtomicBool) -> Result<ToolResult> {
        let args = serde_json::from_value(args.clone()).context("invalid read arguments")?;
        let result = self.read(args, cancelled);
        if cancelled.load(Ordering::Relaxed) {
            let mut result = ToolResult::error("read cancelled");
            result.cancelled = true;
            return Ok(result);
        }
        result
    }
}

struct Page {
    bytes: Vec<u8>,
    next_line: usize,
    partial_line: bool,
    more: bool,
}

fn text_page(
    reader: &mut impl BufRead,
    start: usize,
    lines: usize,
    limit: usize,
    check: impl Fn() -> Result<()>,
) -> Result<Page> {
    let mut bytes = Vec::new();
    let mut line = 1usize;
    let mut completed = 0;
    loop {
        check()?;
        let available = reader.fill_buf()?;
        if available.is_empty() {
            return Ok(Page {
                bytes,
                next_line: line,
                partial_line: false,
                more: false,
            });
        }
        if line >= start && (completed == lines || bytes.len() == limit) {
            let partial_line = !bytes.is_empty() && !bytes.ends_with(b"\n");
            return Ok(Page {
                bytes,
                next_line: line,
                partial_line,
                more: true,
            });
        }
        let mut n = available
            .iter()
            .position(|&b| b == b'\n')
            .map_or(available.len(), |i| i + 1);
        if line >= start {
            n = n.min(limit - bytes.len());
            bytes.extend_from_slice(&available[..n]);
        }
        if available[n - 1] == b'\n' {
            if line >= start {
                completed += 1;
            }
            line = line.checked_add(1).context("line number overflow")?;
        }
        reader.consume(n);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expands_home_without_shell_expansion_and_checks_cancellation_while_scanning() {
        let dir = tempfile::tempdir().unwrap();
        let mut tool = Read::new(dir.path(), Duration::from_secs(2), 256).unwrap();
        tool.home = Some(dir.path().to_owned());
        assert_eq!(
            tool.path("~/picture.png").unwrap(),
            dir.path().join("picture.png")
        );
        assert_eq!(tool.path("~").unwrap(), dir.path());
        assert!(tool.path("~another/picture.png").is_err());
        let mut bytes = std::io::Cursor::new(b"skip\nnext\nlast");
        let calls = std::cell::Cell::new(0);
        let result = text_page(&mut bytes, 10, 1, 256, || {
            calls.set(calls.get() + 1);
            ensure!(calls.get() < 3, "cancelled");
            Ok(())
        });
        assert!(result.is_err());
        assert_eq!(calls.get(), 3);
    }
}
