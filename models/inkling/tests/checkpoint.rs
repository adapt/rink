use rink_inkling::checkpoint::Checkpoint;
use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};

struct Scratch(PathBuf);
impl Scratch {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "rink-checkpoint-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn reads_single_file_and_indexed_checkpoint() {
    let dir = Scratch::new();
    let bytes = include_bytes!("fixtures/tiny/model.safetensors");
    fs::write(dir.0.join("model.safetensors"), bytes).unwrap();
    let cp = Checkpoint::open(&dir.0).unwrap();
    assert_eq!(
        cp.tensor("model.llm.embed.weight").unwrap().shape(),
        &[16, 8]
    );
    assert!(cp.tensor("missing").is_err());
    drop(cp);
    fs::rename(
        dir.0.join("model.safetensors"),
        dir.0.join("shard.safetensors"),
    )
    .unwrap();
    fs::write(
        dir.0.join("model.safetensors.index.json"),
        r#"{"weight_map":{"model.llm.embed.weight":"shard.safetensors"}}"#,
    )
    .unwrap();
    let cp = Checkpoint::open(&dir.0).unwrap();
    assert_eq!(
        cp.tensor("model.llm.embed.weight").unwrap().shape(),
        &[16, 8]
    );
}

#[test]
fn rejects_truncated_data_and_index_path_traversal() {
    let dir = Scratch::new();
    fs::write(dir.0.join("model.safetensors"), b"invalid").unwrap();
    assert!(Checkpoint::open(&dir.0).is_err());
    fs::write(
        dir.0.join("model.safetensors.index.json"),
        r#"{"weight_map":{"a":"../outside.safetensors"}}"#,
    )
    .unwrap();
    assert!(
        Checkpoint::open(&dir.0)
            .err()
            .unwrap()
            .to_string()
            .contains("unsafe checkpoint path")
    );
}
