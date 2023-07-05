//! Common facilities shared by all GPU compute backends

#![allow(clippy::result_large_err)]

mod cache;
pub mod config;
mod device;
mod instance;
mod library;

use self::{cache::PersistentPipelineCache, config::VulkanConfig};
use crate::{SimulateBase, SimulateCreate};
use data::{concentration::Species, parameters::Parameters};
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
    sync::{future::NowFuture, FlushError, GpuFuture},
    ExtensionProperties, LoadingError, OomError, VulkanError, VulkanObject,
};

/// Lower-level, asynchronous interface to a GPU compute backend
///
/// GPU programming is, by nature, asynchronous. After the CPU has submitted
/// work to the GPU, it can move on to other things, and only wait for the GPU
/// when it needs actual results from it. This interface lets you leverage this
/// property for performance by exposing the asynchronism in the API.
///
/// Furthermore, creating multiple Vulkan contexts is expensive, and there is no
/// easy way to communicate between them, so for visualization purposes, we will
/// want to use a single context for both compute and visualization. This
/// requires a little more API surface, which is exposed by this interface.
///
/// If you implement this, then SimulateCreate will be implemented for free and
/// the provided `perform_steps_impl()` method can be used to implement
/// `Simulate`. We can't provide a blanket `Simulate` impl for both
/// `SimulateStep` and `SimulateGpu`, and since `SimulateStep` is more newbie
/// focused it took priority for usage simplicity.
pub trait SimulateGpu: SimulateBase
where
    <Self as SimulateBase>::Error: From<FlushError>,
{
    /// Variant of SimulateCreate::new() that also accepts a preliminary Vulkan
    /// context configuration
    ///
    /// Used by clients who intend to reuse the simulation's Vulkan context for
    /// other purposes, in order to specify their requirements on the Vulkan
    /// context.
    ///
    /// Implementors of SimulateGpu should ensure that their final Vulkan
    /// configuration accepts a subset of the devices accepted by `config`.
    fn with_config(
        params: Parameters,
        args: Self::CliArgs,
        config: VulkanConfig,
    ) -> std::result::Result<Self, Self::Error>;

    /// Access the Vulkan context used by the simulation
    fn context(&self) -> &VulkanContext;

    /// Quick access to `vulkano::now()` on our device
    fn now(&self) -> NowFuture {
        vulkano::sync::now(self.context().device.clone())
    }

    /// GpuFuture returned by `prepare_steps`
    type PrepareStepsFuture<After: GpuFuture + 'static>: GpuFuture + 'static;

    /// Prepare to perform `steps` simulation steps
    ///
    /// This is an asynchronous version of `Simulate::perform_steps`: it
    /// schedules for some simulation steps to occur after the work designated
    /// by `after`, but does not submit the work to the GPU.
    ///
    /// It is then up to the caller to schedule any extra GPU work they need,
    /// then synchronize as needed.
    fn prepare_steps<After: GpuFuture>(
        &self,
        after: After,
        species: &mut Species<Self::Concentration>,
        steps: usize,
    ) -> std::result::Result<Self::PrepareStepsFuture<After>, Self::Error>;

    /// Use this to implement `Simulate::perform_steps`
    fn perform_steps_impl(
        &self,
        species: &mut Species<Self::Concentration>,
        steps: usize,
    ) -> std::result::Result<(), Self::Error> {
        self.prepare_steps(
            vulkano::sync::now(self.context().device.clone()),
            species,
            steps,
        )?
        .then_signal_fence_and_flush()?
        .wait(None)?;
        Ok(())
    }
}
//
impl<T: SimulateGpu> SimulateCreate for T
where
    <T as SimulateBase>::Error: From<FlushError>,
{
    fn new(params: Parameters, args: Self::CliArgs) -> std::result::Result<Self, Self::Error> {
        Self::with_config(params, args, VulkanConfig::default())
    }
}

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
    _messenger: Option<DebugUtilsMessenger>,
}
//
impl<MemAlloc, CommAlloc, DescAlloc> VulkanContext<MemAlloc, CommAlloc, DescAlloc>
where
    MemAlloc: MemoryAllocator,
    CommAlloc: CommandBufferAllocator,
    DescAlloc: DescriptorSetAllocator,
{
    /// Give a Vulkan entity a name, if gpu_debug_utils is enabled
    pub fn set_debug_utils_object_name<Object: VulkanObject + DeviceOwned>(
        &self,
        object: &Object,
        make_name: impl FnOnce() -> Cow<'static, str>,
    ) -> Result<()> {
        if cfg!(feature = "gpu-debug-utils") {
            let name = make_name();
            self.device
                .set_debug_utils_object_name(object, Some(&name))?;
        }
        Ok(())
    }
}

/// Things that can go wrong while setting up a VulkanContext
#[derive(Debug, Error)]
pub enum Error {
    #[error("failed to create a debug utils messenger")]
    DebugUtilsMessengerCreation(#[from] DebugUtilsMessengerCreationError),

    #[error("failed to create logical device")]
    DeviceCreation(#[from] DeviceCreationError),

    #[error("failed to create a Vulkan instance")]
    InstanceCreation(#[from] InstanceCreationError),

    #[error("failed to load the Vulkan library")]
    Loading(#[from] LoadingError),

    #[error("no physical device matches requirements")]
    NoMatchingDevice,

    #[error("ran out of memory")]
    Oom(#[from] OomError),

    #[error("encountered Vulkan runtime error")]
    Vulkan(#[from] VulkanError),

    #[error("home directory not found")]
    HomeDirNotFound,

    #[error("failed to read or write on-disk pipeline cache")]
    PipelineCacheIo(#[from] std::io::Error),

    #[cfg(feature = "livesim")]
    #[error("failed to create a surface from specified window")]
    SurfaceCreation(#[from] SurfaceCreationError),
}
//
/// Result type associated with VulkanContext setup issues
pub type Result<T> = std::result::Result<T, Error>;

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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Once;

    fn init_logger() {
        static INIT_LOGGER: Once = Once::new();
        INIT_LOGGER.call_once(|| env_logger::init());
    }

    #[test]
    fn setup_vulkan() -> Result<()> {
        init_logger();
        VulkanConfig {
            enumerate_portability: true,
            ..VulkanConfig::default()
        }
        .setup()?;
        Ok(())
    }
}
