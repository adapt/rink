use rink_tools::{ContentPart, Read, Tool};
use serde_json::{Value, json};
use std::{
    io::{Read as _, Seek},
    sync::atomic::AtomicBool,
    time::Duration,
};

fn text(tool: &Read, args: Value) -> String {
    tool.execute(&args, &AtomicBool::new(false))
        .unwrap()
        .as_text()
        .unwrap()
        .into()
}

#[test]
fn paginates_utf8_crlf_and_unterminated_lines_with_explicit_continuation() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("notes.txt"),
        "first\r\n世界 🌍\nthird\nlast",
    )
    .unwrap();
    let tool = Read::new(dir.path(), Duration::from_secs(2), 4096).unwrap();
    let page = text(
        &tool,
        json!({"path":"notes.txt","start_line":2,"max_lines":2}),
    );
    assert!(page.contains("2: 世界 🌍\n3: third\n"));
    assert!(page.contains("start_line=4"));
    assert!(!page.contains("1: first"));
    let last = text(&tool, json!({"path":"notes.txt","start_line":4}));
    assert!(last.contains("4: last\n[End of file.]"));
    assert!(text(&tool, json!({"path":"notes.txt","start_line":5})).contains("No text"));
    std::fs::write(dir.path().join("empty"), "").unwrap();
    assert!(text(&tool, json!({"path":"empty"})).contains("No text"));
    // Absolute paths are accepted without using a shell or interpreting $().
    let literal = dir.path().join("$(touch should-not-run).txt");
    std::fs::write(&literal, "literal").unwrap();
    assert!(text(&tool, json!({"path":literal})).contains("1: literal"));
    assert!(!dir.path().join("should-not-run").exists());
}

#[test]
fn text_limits_handle_large_lines_and_utf8_boundaries_without_losing_line_numbers() {
    let dir = tempfile::tempdir().unwrap();
    let tool = Read::new(dir.path(), Duration::from_secs(2), 256).unwrap();
    let file = dir.path().join("large.txt");
    std::fs::write(&file, format!("{}🌍tail\nnext\n", "x".repeat(255))).unwrap();
    let page = text(&tool, json!({"path":"large.txt"}));
    assert!(page.contains(&format!("1: {}\n", "x".repeat(255))));
    assert!(page.contains("truncated within line 1"));
    assert!(!page.contains('\u{fffd}'));
    assert!(text(&tool, json!({"path":"large.txt","start_line":2})).contains("2: next"));
    // Skipping a very long line uses the same small streaming buffer.
    std::fs::write(&file, format!("{}\nsecond\n", "x".repeat(1_000_000))).unwrap();
    assert!(text(&tool, json!({"path":"large.txt","start_line":2})).contains("2: second"));
    std::fs::write(&file, format!("{}\nnext", "x".repeat(255))).unwrap();
    let page = text(&tool, json!({"path":"large.txt"}));
    assert!(page.contains("start_line=2"));
    assert!(!page.contains("within line"));
}

#[test]
fn images_are_detected_by_bytes_and_snapshots_survive_source_changes() {
    let dir = tempfile::tempdir().unwrap();
    let tool = Read::new(dir.path(), Duration::from_secs(2), 256).unwrap();
    for bytes in [
        b"\x89PNG\r\n\x1a\nencoded-png".as_slice(),
        b"\xff\xd8\xffencoded-jpeg".as_slice(),
    ] {
        let path = dir.path().join("image.wrong-extension");
        std::fs::write(&path, bytes).unwrap();
        let mut result = tool
            .execute(&json!({"path":path}), &AtomicBool::new(false))
            .unwrap();
        std::fs::write(&path, "overwritten").unwrap();
        let ContentPart::Image(image) = &mut result.content[1] else {
            panic!("expected an image")
        };
        image.file.rewind().unwrap();
        let mut actual = Vec::new();
        image.file.read_to_end(&mut actual).unwrap();
        assert_eq!(actual, bytes);
        assert_eq!(image.path, path);
        std::fs::write(&path, bytes).unwrap();
        assert!(
            tool.execute(
                &json!({"path":path,"start_line":1}),
                &AtomicBool::new(false)
            )
            .is_err()
        );
    }
}

#[test]
fn rejects_unsupported_files_and_invalid_requests_and_honors_cancellation() {
    let dir = tempfile::tempdir().unwrap();
    let tool = Read::new(dir.path(), Duration::from_secs(2), 256).unwrap();
    for (name, bytes) in [
        ("binary", b"abc\0def".as_slice()),
        ("invalid-utf8", b"abc\xffdef".as_slice()),
        ("document.pdf", b"%PDF-1.7\n".as_slice()),
    ] {
        std::fs::write(dir.path().join(name), bytes).unwrap();
        assert!(
            tool.execute(&json!({"path":name}), &AtomicBool::new(false))
                .is_err()
        );
    }
    for args in [
        json!({"path":"."}),
        json!({"path":"missing"}),
        json!({"path":"","start_line":1}),
        json!({"path":"x","start_line":0}),
        json!({"path":"x","max_lines":0}),
        json!({"path":"x","max_lines":2001}),
        json!({"path":"x","unknown":1}),
    ] {
        assert!(tool.execute(&args, &AtomicBool::new(false)).is_err());
    }
    let cancelled = tool
        .execute(&json!({"path":"missing"}), &AtomicBool::new(true))
        .unwrap();
    assert!(cancelled.cancelled);
    assert!(cancelled.as_text().unwrap().contains("cancelled"));
}

#[cfg(unix)]
#[test]
fn rejects_a_fifo_without_waiting_for_a_writer_and_follows_regular_file_symlinks() {
    use std::{ffi::CString, os::unix::fs::symlink, time::Instant};
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("pipe");
    let c_path = CString::new(path.as_os_str().as_encoded_bytes()).unwrap();
    // SAFETY: c_path is a live, NUL-terminated path inside this test's directory.
    assert_eq!(unsafe { libc::mkfifo(c_path.as_ptr(), 0o600) }, 0);
    let tool = Read::new(dir.path(), Duration::from_secs(2), 256).unwrap();
    let start = Instant::now();
    assert!(
        tool.execute(&json!({"path":path}), &AtomicBool::new(false))
            .is_err()
    );
    assert!(start.elapsed() < Duration::from_secs(1));
    std::fs::write(dir.path().join("target"), "linked text").unwrap();
    symlink("target", dir.path().join("link")).unwrap();
    assert!(text(&tool, json!({"path":"link"})).contains("linked text"));
}

#[test]
fn text_only_models_reject_images_and_advertise_only_text_reading() {
    let dir = tempfile::tempdir().unwrap();
    // Only a signature is needed: rejection precedes image validation/snapshotting.
    std::fs::write(dir.path().join("image.png"), b"\x89PNG\r\n\x1a\n").unwrap();
    std::fs::write(dir.path().join("image.jpg"), b"\xff\xd8\xff").unwrap();
    std::fs::write(dir.path().join("text"), "still readable").unwrap();
    let tool = Read::new(dir.path(), Duration::from_secs(2), 4096)
        .unwrap()
        .with_images(false);
    for path in ["image.png", "image.jpg"] {
        let error = tool
            .execute(&json!({"path":path}), &AtomicBool::new(false))
            .err()
            .unwrap();
        assert!(error.to_string().contains("does not support image input"));
    }
    assert!(text(&tool, json!({"path":"text"})).contains("1: still readable"));
    let definition = tool.definition();
    let description = definition["description"].as_str().unwrap();
    assert!(
        description.contains("Images, directories, PDFs and other binary formats are unsupported")
    );
    assert!(!description.contains("visual context"));
}
