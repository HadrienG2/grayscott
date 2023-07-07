//! Vulkan library loading

use super::ContextBuildResult;
#[allow(unused_imports)]
use log::{debug, error, info, log, trace, warn};
use std::sync::Arc;
use vulkano::VulkanLibrary;

/// Load the Vulkan library
pub fn load() -> ContextBuildResult<Arc<VulkanLibrary>> {
    let library = VulkanLibrary::new()?;
    info!("Loaded Vulkan library");
    trace!("- Supports Vulkan v{}", library.api_version());
    trace!(
        "- Supports instance extensions {}",
        super::format_extension_properties(library.extension_properties())
    );
    trace!(
        "- Supports layers {:#?}",
        library
            .layer_properties()?
            .map(|layer| {
                format!(
                    "{} v{} for Vulkan >= {}",
                    layer.name(),
                    layer.implementation_version(),
                    layer.vulkan_version()
                )
            })
            .collect::<Vec<_>>()
    );
    Ok(library)
}
