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
fn cached_metadata_keeps_shard_offsets_types_and_payloads() {
    use safetensors::{
        Dtype,
        tensor::{TensorView, serialize},
    };
    let dir = Scratch::new();
    for (file, tensors) in [
        (
            "one.safetensors",
            vec![
                (
                    "aaa_padding",
                    TensorView::new(Dtype::U8, vec![3], &[9, 8, 7]).unwrap(),
                ),
                (
                    "first",
                    TensorView::new(Dtype::U8, vec![2], &[1, 2]).unwrap(),
                ),
            ],
        ),
        (
            "two.safetensors",
            vec![(
                "second",
                TensorView::new(Dtype::I16, vec![1], &[3, 0]).unwrap(),
            )],
        ),
    ] {
        fs::write(dir.0.join(file), serialize(tensors, None).unwrap()).unwrap();
    }
    fs::write(
        dir.0.join("model.safetensors.index.json"),
        r#"{"weight_map":{"first":"one.safetensors","second":"two.safetensors"}}"#,
    )
    .unwrap();
    let cp = Checkpoint::open(&dir.0).unwrap();
    for _ in 0..3 {
        let a = cp.tensor("first").unwrap();
        let b = cp.tensor("second").unwrap();
        assert_eq!(
            (a.dtype(), a.shape(), a.data()),
            (Dtype::U8, &[2][..], &[1, 2][..])
        );
        assert_eq!(
            (b.dtype(), b.shape(), b.data()),
            (Dtype::I16, &[1][..], &[3, 0][..])
        );
        assert!(!cp.contains("aaa_padding"));
    }
    drop(cp);
    fs::write(
        dir.0.join("model.safetensors.index.json"),
        r#"{"weight_map":{"missing":"one.safetensors"}}"#,
    )
    .unwrap();
    assert!(Checkpoint::open(&dir.0).is_err());
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
