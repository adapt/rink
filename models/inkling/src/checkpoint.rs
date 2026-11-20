//! Memory-mapped SafeTensors loading. Files must not be modified while mapped.
use anyhow::{Context, Result, ensure};
use memmap2::Mmap;
use safetensors::{SafeTensors, tensor::TensorView};
use std::{
    collections::BTreeMap,
    fs::File,
    path::{Component, Path},
};

pub struct Checkpoint {
    maps: BTreeMap<String, Mmap>,
    index: BTreeMap<String, String>,
}
impl Checkpoint {
    pub fn open(directory: &Path) -> Result<Self> {
        let index_path = directory.join("model.safetensors.index.json");
        let mut index: BTreeMap<String, String> = if index_path.exists() {
            #[derive(serde::Deserialize)]
            struct Index {
                weight_map: BTreeMap<String, String>,
            }
            serde_json::from_slice::<Index>(&std::fs::read(index_path)?)?.weight_map
        } else {
            BTreeMap::new()
        };
        let files: std::collections::BTreeSet<_> = if index.is_empty() {
            ["model.safetensors".to_string()].into_iter().collect()
        } else {
            index.values().cloned().collect()
        };
        let mut maps = BTreeMap::new();
        for filename in files {
            ensure!(
                Path::new(&filename)
                    .components()
                    .all(|c| matches!(c, Component::Normal(_))),
                "unsafe checkpoint path: {filename}"
            );
            let file = File::open(directory.join(&filename))
                .with_context(|| format!("opening {filename}"))?;
            // SAFETY: read-only mapping. The caller must keep checkpoint files immutable.
            let map = unsafe { Mmap::map(&file)? };
            let tensors = SafeTensors::deserialize(&map)
                .with_context(|| format!("invalid SafeTensors file {filename}"))?;
            if index.is_empty() {
                for name in tensors.names() {
                    index.insert(name.to_string(), filename.clone());
                }
            }
            maps.insert(filename, map);
        }
        let result = Self { maps, index };
        for (name, file) in &result.index {
            ensure!(result.maps.contains_key(file), "missing shard for {name}");
        }
        Ok(result)
    }
    pub fn tensor(&self, name: &str) -> Result<TensorView<'_>> {
        let file = self
            .index
            .get(name)
            .with_context(|| format!("missing tensor {name}"))?;
        SafeTensors::deserialize(&self.maps[file])?
            .tensor(name)
            .with_context(|| format!("loading {name}"))
    }
    pub fn contains(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }
}
