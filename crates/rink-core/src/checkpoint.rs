//! Memory-mapped SafeTensors loading. Files must not be modified while mapped.
use anyhow::{Context, Result, ensure};
use memmap2::Mmap;
use safetensors::{
    SafeTensors,
    tensor::{Metadata, TensorView},
};
use std::{
    collections::BTreeMap,
    fs::File,
    path::{Component, Path},
};

struct Shard {
    map: Mmap,
    metadata: Metadata,
    data_start: usize,
}

pub struct Checkpoint {
    maps: BTreeMap<String, Shard>,
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
            let (header, metadata) = SafeTensors::read_metadata(&map)
                .with_context(|| format!("invalid SafeTensors file {filename}"))?;
            if index.is_empty() {
                for name in metadata.tensors().keys() {
                    index.insert(name.to_string(), filename.clone());
                }
            }
            maps.insert(
                filename,
                Shard {
                    map,
                    metadata,
                    data_start: 8 + header,
                },
            );
        }
        let result = Self { maps, index };
        for (name, file) in &result.index {
            let shard = result
                .maps
                .get(file)
                .with_context(|| format!("missing shard for {name}"))?;
            ensure!(
                shard.metadata.info(name).is_some(),
                "missing tensor {name} in {file}"
            );
        }
        Ok(result)
    }
    pub fn tensor(&self, name: &str) -> Result<TensorView<'_>> {
        let file = self
            .index
            .get(name)
            .with_context(|| format!("missing tensor {name}"))?;
        let shard = &self.maps[file];
        let info = shard
            .metadata
            .info(name)
            .with_context(|| format!("missing tensor {name}"))?;
        let (start, end) = info.data_offsets;
        // read_metadata validated all offsets, shapes and the complete data extent.
        TensorView::new(
            info.dtype,
            info.shape.clone(),
            &shard.map[shard.data_start + start..shard.data_start + end],
        )
        .with_context(|| format!("loading {name}"))
    }
    pub fn contains(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }
}
