//! Common Vulkan context shared by all simulations

mod cache;
pub mod config;
mod device;
mod instance;
mod library;

use self::{cache::PersistentPipelineCache, config::VulkanConfig};
#[allow(unused_imports)]
use log::{debug, error, info, log, trace, warn};
use std::{borrow::Cow, sync::Arc};
use thiserror::Error;
#[cfg(feature = "livesim")]
use vulkano::swapchain::SurfaceCreationError;
use vulkano::{
    command_buffer::allocator::{CommandBufferAllocator, StandardCommandBufferAllocator},
    descriptor_set::allocator::{DescriptorSetAllocator, StandardDescriptorSetAllocator},
    device::{Device, DeviceCreationError, DeviceOwned, Queue},
    instance::{
        debug::{DebugUtilsMessenger, DebugUtilsMessengerCreationError},
        InstanceCreationError,
    },
    memory::allocator::{MemoryAllocator, StandardMemoryAllocator},
    swapchain::Surface,
    ExtensionProperties, LoadingError, OomError, VulkanError, VulkanObject,
};

/// Vulkan compute context
///
/// Common setup you need in order to perform any useful computation with Vulkan.
/// Designed to make the easy case easy, while enabling sufficient tweaking and
/// debugging when needed.
///
/// Keep this struct alive as long as you're using Vulkan, as that's how long
/// debug logging is going to keep printing useful info ;)
///
/// Built using the [`VulkanConfig`] configuration struct
pub struct VulkanContext<
    MemAlloc: MemoryAllocator = StandardMemoryAllocator,
    CommAlloc: CommandBufferAllocator = StandardCommandBufferAllocator,
    DescAlloc: DescriptorSetAllocator = StandardDescriptorSetAllocator,
> {
    /// Window surface (set if a window was specified in the VulkanConfig)
    pub surface: Option<Arc<Surface>>,

    /// Logical device (used for resource allocation)
    pub device: Arc<Device>,

    /// Command queues (used for command submission)
    pub queues: Box<[Arc<Queue>]>,

    /// Memory allocator (used for image and buffer allocation)
    pub memory_allocator: Arc<MemAlloc>,

    /// Command buffer allocator
    pub command_allocator: Arc<CommAlloc>,

    /// Descriptor set allocator
    pub descriptor_set_allocator: DescAlloc,

    /// Pipeline cache (used for e.g. compiled shader caching)
    pub pipeline_cache: PersistentPipelineCache,

    /// Messenger that sends Vulkan debug messages to the [`log`] crate
    pub(crate) _messenger: Option<DebugUtilsMessenger>,
}
//
impl<MemAlloc, CommAlloc, DescAlloc> VulkanContext<MemAlloc, CommAlloc, DescAlloc>
where
    MemAlloc: MemoryAllocator,
    CommAlloc: CommandBufferAllocator,
    DescAlloc: DescriptorSetAllocator,
{
    /// Build a Vulkan context with a certain configuration
    pub fn new(config: VulkanConfig<MemAlloc, CommAlloc, DescAlloc>) -> ContextBuildResult<Self> {
        config.build()
    }

    /// Give a Vulkan entity a name, if gpu_debug_utils is enabled
    pub fn set_debug_utils_object_name<Object: VulkanObject + DeviceOwned>(
        &self,
        object: &Object,
        make_name: impl FnOnce() -> Cow<'static, str>,
    ) -> Result<(), OomError> {
        if cfg!(feature = "gpu-debug-utils") {
            let name = make_name();
            self.device
                .set_debug_utils_object_name(object, Some(&name))?;
        }
        Ok(())
    }
}
//
impl<MemAlloc, CommAlloc, DescAlloc> TryFrom<VulkanConfig<MemAlloc, CommAlloc, DescAlloc>>
    for VulkanContext<MemAlloc, CommAlloc, DescAlloc>
where
    MemAlloc: MemoryAllocator,
    CommAlloc: CommandBufferAllocator,
    DescAlloc: DescriptorSetAllocator,
{
    type Error = ContextBuildError;

    fn try_from(config: VulkanConfig<MemAlloc, CommAlloc, DescAlloc>) -> ContextBuildResult<Self> {
        config.build()
    }
}

/// Things that can go wrong while setting up a VulkanContext
#[derive(Debug, Error)]
pub enum ContextBuildError {
    #[error("failed to load the Vulkan library")]
    Loading(#[from] LoadingError),

    #[error("failed to create a Vulkan instance")]
    InstanceCreation(#[from] InstanceCreationError),

    #[error("failed to create a debug utils messenger")]
    DebugUtilsMessengerCreation(#[from] DebugUtilsMessengerCreationError),

    #[cfg(feature = "livesim")]
    #[error("failed to create a surface from specified window")]
    SurfaceCreation(#[from] SurfaceCreationError),

    #[error("no physical device matches requirements")]
    NoMatchingDevice,

    #[error("failed to create a logical device")]
    DeviceCreation(#[from] DeviceCreationError),

    #[error("ran out of memory")]
    Oom(#[from] OomError),

    #[error("encountered a Vulkan runtime error")]
    Vulkan(#[from] VulkanError),

    #[error("did not find home directory")]
    HomeDirNotFound,

    #[error("failed to read or write on-disk pipeline cache")]
    PipelineCacheIo(#[from] std::io::Error),
}
//
/// Result type associated with VulkanContext setup issues
pub type ContextBuildResult<T> = std::result::Result<T, ContextBuildError>;

/// Format Vulkan extension properties for display
fn format_extension_properties(extension_properties: &[ExtensionProperties]) -> String {
    format!(
        "{:#?}",
        extension_properties
            .iter()
            .map(|ext| format!("{} v{}", ext.extension_name, ext.spec_version))
            .collect::<Vec<_>>()
    )
}
