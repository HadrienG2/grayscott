//! Pipeline cache

use super::ContextBuildResult;
use directories::ProjectDirs;
#[allow(unused_imports)]
use log::{debug, error, info, log, trace, warn};
use std::{
    fs::File,
    io::{Read, Write},
    ops::Deref,
    path::PathBuf,
    sync::Arc,
};
use vulkano::{device::Device, pipeline::cache::PipelineCache};

/// GPU pipeline cache
///
/// Used to avoid shader recompilations by caching precompiled compute and
/// render pipelines to disk.
pub struct PersistentPipelineCache {
    /// In-RAM cache
    cache: Arc<PipelineCache>,

    /// Path to be used for on-disk persistence
    path: PathBuf,
}
//
impl PersistentPipelineCache {
    /// Attempt to load the pipeline cache from disk, otherwise create a new one
    pub fn new(dirs: &ProjectDirs, device: Arc<Device>) -> ContextBuildResult<Self> {
        let path = dirs.cache_dir().join("gpu_pipelines.bin");

        // TODO: Consider treating some I/O errors as fatal and others as okay
        let cache = if let Ok(mut cache_file) = File::open(&path) {
            let mut data = Vec::new();
            cache_file.read_to_end(&mut data)?;
            // Assumed safe because we hopefully created this file...
            unsafe { PipelineCache::with_data(device, &data)? }
        } else {
            PipelineCache::empty(device)?
        };

        Ok(Self { cache, path })
    }

    /// Write the pipeline cache back to disk
    pub fn write(&self) -> ContextBuildResult<()> {
        let data = self.cache.get_data()?;

        let dir = self
            .path
            .parent()
            .expect("Pipeline cache shouldn't be dumped in current directory!");
        if !dir.try_exists()? {
            std::fs::create_dir_all(dir)?;
        }

        let wal_path = self.path.with_extension("wal");
        let mut file = File::create(&wal_path)?;
        match file.write_all(&data) {
            Ok(_) => {
                std::fs::rename(wal_path, &self.path)?;
                Ok(())
            }
            Err(e) => {
                std::fs::remove_file(wal_path)?;
                Err(e.into())
            }
        }
    }
}
//
impl Deref for PersistentPipelineCache {
    type Target = Arc<PipelineCache>;

    fn deref(&self) -> &Self::Target {
        &self.cache
    }
}
//
impl Drop for PersistentPipelineCache {
    fn drop(&mut self) {
        if let Err(e) = self.write() {
            error!("Failed to write pipeline cache to disk: {e}");
        }
    }
}
