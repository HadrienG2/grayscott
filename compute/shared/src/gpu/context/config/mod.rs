//! Vulkan context configuration

mod defaults;
pub mod requirements;

use super::{
    cache::PersistentPipelineCache,
    device,
    instance::{self, DebuggedInstance},
    library, ContextBuildError, ContextBuildResult, VulkanContext,
};
use directories::ProjectDirs;
#[allow(unused_imports)]
use log::{debug, error, info, log, trace, warn};
use std::{borrow::Cow, cmp::Ordering, sync::Arc};
#[cfg(feature = "livesim")]
use vulkano::instance::Instance;
use vulkano::{
    command_buffer::allocator::{CommandBufferAllocator, StandardCommandBufferAllocator},
    descriptor_set::allocator::{DescriptorSetAllocator, StandardDescriptorSetAllocator},
    device::{physical::PhysicalDevice, Device, DeviceExtensions, Features, QueueCreateInfo},
    instance::InstanceExtensions,
    memory::allocator::{MemoryAllocator, StandardMemoryAllocator},
    swapchain::Surface,
    VulkanLibrary,
};
#[cfg(feature = "livesim")]
use winit::window::Window;

/// Device requirements linked to use of a Surface
pub type DeviceSurfaceRequirements = Box<dyn FnMut(&PhysicalDevice, &Surface) -> bool>;

/// Vulkan compute context configuration
///
/// A default configuration is provided via the [`default()`] method and
/// documented in the various fields of this struct. You can change these fields
/// to adjust the configuration, check out their documentation to see what their
/// default behavior is.
///
/// Once you're satisfied with the configuration, used the [`build()`] method
/// to set up the Vulkan context.
///
/// [`default()`]: VulkanConfig::default()
/// [`build()`]: VulkanConfig::build()
#[allow(clippy::type_complexity)]
pub struct VulkanConfig<
    MemAlloc: MemoryAllocator = StandardMemoryAllocator,
    CommAlloc: CommandBufferAllocator = StandardCommandBufferAllocator,
    DescAlloc: DescriptorSetAllocator = StandardDescriptorSetAllocator,
> {
    /// Window to which this Vulkan context is meant to render, if any,
    /// along with associated surface requirements
    #[cfg(feature = "livesim")]
    pub window_and_reqs: Option<(Arc<Window>, DeviceSurfaceRequirements)>,

    /// Decide which Vulkan layers should be enabled
    ///
    /// Layers are used for various debugging and profiling tasks. Note that
    /// you can use the `VK_INSTANCE_LAYERS` environment variable to activate
    /// layers, instead of activating them in code.
    ///
    /// By default, "VK_LAYER_KHRONOS_validation" is enabled on debug builds.
    pub layers: Box<dyn FnOnce(&VulkanLibrary) -> Vec<String>>,

    /// Decide which instance extensions should be enabled
    ///
    /// By default, we require the ext_debug_utils extension in order to provide
    /// a good debugging experience. Additionally, in debug builds, we also
    /// enable khr_get_physical_device_properties2 on older Vulkan versions,
    /// which is a prerequisite for the device robustness features that we
    /// enable by default.
    ///
    /// If the gpu-debug-utils feature is enabled, then we force the debug_utils
    /// extension to be enabled, even if your configuration says otherwise, as
    /// it is required for that functionality to work.
    pub instance_extensions: Box<dyn FnOnce(&VulkanLibrary) -> InstanceExtensions>,

    /// Truth that Vulkan Portability devices should be enumerated
    ///
    /// Some Vulkan implementations, like MoltenVK on macOS and iOS, do not
    /// support the full Vulkan 1.0 specification, leaving out some relatively
    /// obscure features. By setting this flag, you enable these devices to be
    /// discovered. In exchange, your device selection callback needs to account
    /// for the possibility that some core Vulkan features may be absent.
    pub enumerate_portability: bool,

    /// Decide which device features and extensions should be enabled
    ///
    /// Features are basically extensions that were integrated into the Core
    /// Vulkan specification, so it makes sense to decide on these two matters
    /// together.
    ///
    /// You can adapt your choice of features and extensions depending on what a
    /// device actually supports. If the final set of features and extensions
    /// that you return is not supported by a device, that device will be
    /// discarded at enumeration time.
    ///
    /// By default, if portability devices are enumerated, the associated
    /// khr_portability_subset extension is enabled. Additionally, in debug
    /// builds, all available robustness features are enabled:
    ///
    /// - robust_buffer_access
    /// - robust_image_access
    /// - robust_buffer_access2
    /// - robust_image_access2
    ///
    /// Associated robustness extensions are enabled as appropriate for the
    /// device's supported Vulkan version.
    pub device_features_extensions: Box<dyn FnMut(&PhysicalDevice) -> (Features, DeviceExtensions)>,

    /// Impose additional device requirements
    ///
    /// Requirements on the compute device other than features and extensions
    /// to be enabled. This is where you can check for things like minimal
    /// extension versions, device limits, Vulkan API support...
    ///
    /// Return `true` to signify that a device meets your requirements. By
    /// default, no additional requirements are imposed.
    pub other_device_requirements: Box<dyn FnMut(&PhysicalDevice) -> bool>,

    /// Decide which device is best
    ///
    /// At the moment, this code only supports computing using a single GPU
    /// device. This means that among the GPUs exposed by the system, we will
    /// need to pick one "best" GPU. This ordering relationship is used to guide
    /// this choice. In case of a tie, the first "best" device in the order of
    /// Vulkan device enumeration is selected.
    ///
    /// By default, we pick the device type which is most likely to exhibit
    /// maximal compute performance:
    ///
    /// - Physical discrete GPU is best
    /// - Then virtual GPU (hopefully a discrete one, with virtualization overhead)
    /// - Then integrated GPU (at least it's a GPU)
    /// - Then CPUs (at least we know what it is)
    /// - Then others (we don't even know what it is)
    ///
    /// This should be enough to disambiguate all common multi-device scenarios,
    /// but edge cases like machines with heterogeneous GPUs plugged in will
    /// require better preference criteria.
    ///
    /// Furthermore, this default logic is also tunable through the
    /// `GRAYSCOTT_PREFER_DEVICE` environment variable, which sets a preferred
    /// device type that overrides the default precedence order :
    ///
    /// - "discrete" is the default setting : prefer discrete GPUs.
    /// - "integrated" = prefer integrated GPUs
    /// - "virtual" = prefer virtual GPUs
    /// - "cpu" = prefer CPUs
    /// - "other" = prefer other things
    pub device_preference: Box<dyn FnMut(&PhysicalDevice, &PhysicalDevice) -> Ordering>,

    /// Configure command queues
    ///
    /// Many devices can perform multiple kinds of work in parallel, e.g.
    /// compute in parallel with graphics and CPU-GPU data transfers in
    /// parallel with either. In Vulkan, this is expressed through the notion
    /// of queue families, from which concrete command queues are constructed
    ///
    /// This lets you decide which queues you want to allocate, and how you want
    /// to name them when the gpu-debug-utils features is enabled.
    ///
    /// If a surface is passed in as a parameter, please ensure that the first
    /// selected queue should be able to perform compute operations and present
    /// to this surface. This is required so that "livesim" does not need to
    /// muck around with your queue configuration.
    ///
    /// By default, we only allocate a single queue, suitable for compute
    /// (and thus data transfers) and presenting to the target surface. We try
    /// to pick the device's main queue family for this purpose, at it may be
    /// most performant in single-queue use cases. If you want to experiment
    /// with multi-queue workflows, this is the tuning knob that you want.
    pub queues: Box<
        dyn FnOnce(
            &PhysicalDevice,
            Option<&Surface>,
        ) -> (Vec<QueueCreateInfo>, Vec<Cow<'static, str>>),
    >,

    /// Set up a memory allocator
    ///
    /// To allocate device memory or device-accessible host memory, you will
    /// need a memory allocator.
    ///
    /// By default, we use vulkano's standard general-purpose memory allocator.
    /// You can switch to a different memory allocation strategy by tuning this.
    pub memory_allocator: Box<dyn FnOnce(Arc<Device>) -> MemAlloc>,

    /// Set up a command buffer allocator
    ///
    /// Command buffers are used to submit work to the GPU. They too need to be
    /// allocated using a specialized allocator.
    ///
    /// By default, we use vulkano's standard general-purpose allocator.
    /// You can switch to a different allocation strategy by tuning this.
    pub command_allocator: Box<dyn FnOnce(Arc<Device>) -> CommAlloc>,

    /// Set up a descriptor set allocator
    ///
    /// Descriptor sets are used to bind resources to GPU pipelines. Again, a
    /// specialized allocator is needed here.
    ///
    /// By default, we use vulkano's standard general-purpose allocator.
    /// You can switch to a different allocation strategy by tuning this.
    pub descriptor_set_allocator: Box<dyn FnOnce(Arc<Device>) -> DescAlloc>,
}
//
impl Default
    for VulkanConfig<
        StandardMemoryAllocator,
        StandardCommandBufferAllocator,
        StandardDescriptorSetAllocator,
    >
{
    /// Suggested defaults for all configuration items
    ///
    /// You can use struct update syntax to change only some settings, keeping
    /// the others to their default values:
    ///
    /// ```
    /// # use compute::gpu::VulkanConfig;
    /// let config = VulkanConfig {
    ///     enumerate_portability: true,
    ///     .. VulkanConfig::default()
    /// };
    /// ```
    fn default() -> Self {
        defaults::config()
    }
}
//
impl<
        MemAlloc: MemoryAllocator,
        CommAlloc: CommandBufferAllocator,
        DescAlloc: DescriptorSetAllocator,
    > VulkanConfig<MemAlloc, CommAlloc, DescAlloc>
{
    /// Set up a Vulkan compute context with this configuration
    ///
    /// ```
    /// # use compute::gpu::VulkanConfig;
    /// let context = VulkanConfig::default().build()?;
    /// # Ok::<(), compute::gpu::Error>(())
    /// ```
    #[allow(unused_assignments, unused_mut)]
    pub fn build(mut self) -> ContextBuildResult<VulkanContext<MemAlloc, CommAlloc, DescAlloc>> {
        // Load vulkan library
        let library = library::load()?;

        // Set up instance
        let mut will_render = false;
        #[cfg(feature = "livesim")]
        {
            will_render = self.window_and_reqs.is_some();
        }
        let instance_extensions = instance::select_extensions(
            &library,
            (self.instance_extensions)(&library),
            will_render,
        );
        let layers = (self.layers)(&library);
        let instance = DebuggedInstance::new(
            library,
            layers,
            instance_extensions,
            self.enumerate_portability,
        )?;

        // Set up surface (if a window is provided)
        let mut surface_and_reqs: Option<(Arc<Surface>, DeviceSurfaceRequirements)> = None;
        #[cfg(feature = "livesim")]
        {
            if let Some((window, reqs)) = self.window_and_reqs {
                let created_surface = create_surface(instance.clone(), window)?;
                surface_and_reqs = Some((created_surface, reqs));
            }
        }

        // Select physical device
        let physical_device = device::select_physical(
            &instance,
            &mut self.device_features_extensions,
            self.other_device_requirements,
            self.device_preference,
            &mut surface_and_reqs,
        )?;
        let surface = surface_and_reqs.map(|t| t.0);

        // Set up logical device and queues
        let (features, mut extensions) = (self.device_features_extensions)(&physical_device);
        if surface.is_some() {
            extensions.khr_swapchain = true;
        }
        let (device, queues) = device::create_logical(
            physical_device.clone(),
            features,
            extensions,
            (self.queues)(&physical_device, surface.as_deref()),
        )?;

        // Set up memory allocators
        let memory_allocator = Arc::new((self.memory_allocator)(device.clone()));
        let command_allocator = Arc::new((self.command_allocator)(device.clone()));
        let descriptor_set_allocator = (self.descriptor_set_allocator)(device.clone());

        // Set up pipeline cache
        let dirs =
            ProjectDirs::from("", "", "grayscott").ok_or(ContextBuildError::HomeDirNotFound)?;
        let pipeline_cache = PersistentPipelineCache::new(&dirs, device.clone())?;

        // We're done!
        Ok(VulkanContext {
            _messenger: instance._messenger,
            surface,
            device,
            queues,
            memory_allocator,
            command_allocator,
            pipeline_cache,
            descriptor_set_allocator,
        })
    }
}

/// Create a Surface
#[cfg(feature = "livesim")]
pub fn create_surface(
    instance: Arc<Instance>,
    window: Arc<Window>,
) -> ContextBuildResult<Arc<Surface>> {
    let created_surface = vulkano_win::create_surface_from_winit(window, instance)?;
    info!("Created a surface from {:?} window", created_surface.api());
    Ok(created_surface)
}
