#![cfg(unix)]
use rink_tools::{Bash, Registry, Tool};
use serde_json::{Value, json};
use std::{
    path::PathBuf,
    sync::atomic::{AtomicBool, AtomicUsize, Ordering},
    time::{Duration, Instant, SystemTime},
};

struct Directory(PathBuf);
static DIRECTORY_ID: AtomicUsize = AtomicUsize::new(0);
impl Directory {
    fn new() -> Self {
        let p = std::env::temp_dir().join(format!(
            "rink-tools-{}-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            DIRECTORY_ID.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&p).unwrap();
        Self(p)
    }
    fn bash(&self, ms: u64, bytes: usize) -> Bash {
        Bash::new(&self.0, Duration::from_millis(ms), bytes).unwrap()
    }
}
impl Drop for Directory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn run(bash: &Bash, command: &str) -> Value {
    serde_json::from_str(
        bash.execute(&json!({"command":command}), &AtomicBool::new(false))
            .unwrap()
            .as_text()
            .unwrap(),
    )
    .unwrap()
}

#[test]
fn captures_status_streams_cwd_and_uses_fresh_shells() {
    let dir = Directory::new();
    let bash = dir.bash(2000, 4096);
    let result = run(&bash, "printf '%s' \"$PWD\"; printf 'problem' >&2; exit 7");
    assert_eq!(result["exit_code"], 7);
    assert_eq!(
        result["stdout"],
        dir.0.canonicalize().unwrap().to_str().unwrap()
    );
    assert_eq!(result["stderr"], "problem");
    assert_eq!(result["timed_out"], false);
    run(&bash, "export RINK_TOOL_TEST=changed; cd /");
    assert_eq!(
        run(&bash, "printf '%s' \"${RINK_TOOL_TEST-unset}\"")["stdout"],
        "unset"
    );
    assert_eq!(run(&bash, "read x; printf '%s' \"$?\"")["stdout"], "1");
}

#[test]
fn drains_both_streams_with_bounded_capture_and_handles_invalid_utf8() {
    let dir = Directory::new();
    let result = run(
        &dir.bash(5000, 1024),
        "for ((i=0;i<20000;i++)); do printf abcdefgh; printf ijklmnop >&2; done",
    );
    assert_eq!(result["exit_code"], 0);
    assert_eq!(result["stdout"].as_str().unwrap().len(), 512);
    assert_eq!(result["stderr"].as_str().unwrap().len(), 512);
    assert_eq!(result["stdout_truncated"], true);
    assert_eq!(result["stderr_truncated"], true);
    assert_eq!(
        run(&dir.bash(2000, 1024), "printf '\\377'")["stdout"],
        "\u{fffd}"
    );
}

#[test]
fn timeout_and_normal_exit_stop_background_children() {
    let dir = Directory::new();
    let start = Instant::now();
    let result = run(
        &dir.bash(60, 1024),
        "(sleep 0.3; touch timeout-survivor) & wait",
    );
    assert_eq!(result["timed_out"], true);
    assert!(start.elapsed() < Duration::from_secs(2));
    let start = Instant::now();
    let result = run(
        &dir.bash(2000, 1024),
        "(sleep 0.3; touch exit-survivor) & exit 0",
    );
    assert_eq!(result["exit_code"], 0);
    assert!(start.elapsed() < Duration::from_secs(2));
    std::thread::sleep(Duration::from_millis(400));
    assert!(!dir.0.join("timeout-survivor").exists());
    assert!(!dir.0.join("exit-survivor").exists());
}

#[test]
fn cancellation_and_invalid_arguments_do_not_leave_commands_running() {
    let dir = Directory::new();
    let bash = dir.bash(3000, 1024);
    let flag = AtomicBool::new(false);
    std::thread::scope(|s| {
        s.spawn(|| {
            std::thread::sleep(Duration::from_millis(60));
            flag.store(true, Ordering::Relaxed);
        });
        let result = bash
            .execute(
                &json!({"command":"(sleep 0.3; touch cancelled-survivor) & wait"}),
                &flag,
            )
            .unwrap();
        assert!(result.cancelled);
    });
    let result = bash
        .execute(&json!({"command":"touch must-not-start"}), &flag)
        .unwrap();
    assert!(result.cancelled);
    let flag = AtomicBool::new(false);
    for args in [
        json!({"command":"touch invalid", "timeout_ms":0}),
        json!({"command":"touch invalid", "timeout_ms":4000}),
        json!({"command":"touch invalid", "extra":true}),
        json!({"command":42}),
        json!({"command":"x\u{0}"}),
    ] {
        assert!(bash.execute(&args, &flag).is_err());
    }
    let mut registry = Registry::default();
    registry.register(bash).unwrap();
    assert!(
        registry
            .execute("not_enabled", &json!({"command":"touch invalid"}), &flag)
            .as_text()
            .unwrap()
            .contains("not enabled")
    );
    std::thread::sleep(Duration::from_millis(400));
    for file in ["cancelled-survivor", "must-not-start", "invalid"] {
        assert!(!dir.0.join(file).exists());
    }
}
