//! Common Vulkan context shared by all simulations

mod cache;
pub mod config;
mod device;
mod instance;
mod library;

use self::{cache::PersistentPipelineCache, config::VulkanConfig};
#[allow(unused_imports)]
use log::{debug, error, info, log, trace, warn};
use std::{
    borrow::{Borrow, Cow},
    collections::HashSet,
    sync::Arc,
};
use thiserror::Error;
use vulkano::{
    command_buffer::allocator::{CommandBufferAllocator, StandardCommandBufferAllocator},
    descriptor_set::allocator::{DescriptorSetAllocator, StandardDescriptorSetAllocator},
    device::{Device, DeviceOwnedVulkanObject, Queue},
    instance::debug::DebugUtilsMessenger,
    memory::allocator::{MemoryAllocator, StandardMemoryAllocator},
    swapchain::{FromWindowError, Surface},
    ExtensionProperties, LoadingError, Validated, ValidationError, VulkanError,
};
use winit::raw_window_handle::HandleError;

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
    pub descriptor_set_allocator: Arc<DescAlloc>,

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
    pub fn set_debug_utils_object_name<Object: DeviceOwnedVulkanObject>(
        &self,
        object: &Object,
        make_name: impl FnOnce() -> Cow<'static, str>,
    ) -> Result<(), Validated<VulkanError>> {
        if cfg!(feature = "gpu-debug-utils") {
            let name = make_name();
            object.set_debug_utils_object_name(Some(&name))?;
        }
        Ok(())
    }

    /// Queue families present in [`Self::queues`]
    pub fn queue_family_indices(&self) -> impl Borrow<HashSet<u32>> + '_ {
        let queue_family_indices = self
            .queues
            .iter()
            .map(|queue| queue.queue_family_index())
            .collect::<HashSet<_>>();
        queue_family_indices
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

    #[error("no physical device matches requirements")]
    NoMatchingDevice,

    #[error("a Vulkan API call errored out or failed validation ({0})")]
    Vulkan(#[from] Validated<VulkanError>),

    #[error("did not find home directory")]
    HomeDirNotFound,

    #[error("failed to read or write on-disk pipeline cache")]
    PipelineCacheIo(#[from] std::io::Error),

    #[error("failed to build swapchain from window")]
    SwapchainFromWindow(#[from] FromWindowError),

    #[error("failed to fetch display/window hadle")]
    Handle(#[from] HandleError),
}
//
impl From<VulkanError> for ContextBuildError {
    fn from(value: VulkanError) -> Self {
        Self::Vulkan(Validated::Error(value))
    }
}
//
impl From<Box<ValidationError>> for ContextBuildError {
    fn from(value: Box<ValidationError>) -> Self {
        Self::Vulkan(value.into())
    }
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
