//! Common facilities shared by all GPU compute backends

use super::{SimulateBase, SimulateCreate};
use data::{concentration::Species, parameters::Parameters};
use directories::ProjectDirs;
#[allow(unused_imports)]
use log::{debug, error, info, log, trace, warn};
use std::{
    borrow::Cow,
    cmp::Ordering,
    fs::File,
    io::{Read, Write},
    ops::Deref,
    path::PathBuf,
    sync::Arc,
};
use thiserror::Error;
use vulkano::{
    command_buffer::allocator::{
        CommandBufferAllocator, StandardCommandBufferAllocator,
        StandardCommandBufferAllocatorCreateInfo,
    },
    descriptor_set::allocator::{DescriptorSetAllocator, StandardDescriptorSetAllocator},
    device::{
        physical::{PhysicalDevice, PhysicalDeviceType},
        Device, DeviceCreateInfo, DeviceCreationError, DeviceExtensions, DeviceOwned, Features,
        Queue, QueueCreateInfo, QueueFamilyProperties, QueueFlags,
    },
    instance::{
        debug::{
            DebugUtilsMessageSeverity, DebugUtilsMessageType, DebugUtilsMessenger,
            DebugUtilsMessengerCreateInfo, DebugUtilsMessengerCreationError, Message,
        },
        Instance, InstanceCreateInfo, InstanceCreationError, InstanceExtensions, Version,
    },
    memory::allocator::{MemoryAllocator, StandardMemoryAllocator},
    pipeline::cache::PipelineCache,
    swapchain::{Surface, SurfaceCreationError, SurfaceInfo},
    sync::{future::NowFuture, FlushError, GpuFuture},
    ExtensionProperties, LoadingError, OomError, VulkanError, VulkanLibrary, VulkanObject,
};
#[cfg(feature = "livesim")]
use winit::window::Window;

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
    ) -> Result<Self, Self::Error>;

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
    ) -> Result<Self::PrepareStepsFuture<After>, Self::Error>;

    /// Use this to implement `Simulate::perform_steps`
    fn perform_steps_impl(
        &self,
        species: &mut Species<Self::Concentration>,
        steps: usize,
    ) -> Result<(), Self::Error> {
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
    fn new(params: Parameters, args: Self::CliArgs) -> Result<Self, Self::Error> {
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
/// Built from the [`VulkanConfig`] configuration struct
pub struct VulkanContext<
    MemAlloc: MemoryAllocator = StandardMemoryAllocator,
    CommAlloc: CommandBufferAllocator = StandardCommandBufferAllocator,
    DescAlloc: DescriptorSetAllocator = StandardDescriptorSetAllocator,
> {
    /// Messenger that sends Vulkan debug messages to the [`log`] crate
    _messenger: Option<DebugUtilsMessenger>,

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
    pub descriptor_allocator: DescAlloc,

    /// Pipeline cache (used for e.g. compiled shader caching)
    pub pipeline_cache: PersistentPipelineCache,
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
    ) -> Result<(), Error> {
        if cfg!(feature = "gpu-debug-utils") {
            let name = make_name();
            self.device
                .set_debug_utils_object_name(object, Some(&name))?;
        }
        Ok(())
    }
}

/// Vulkan compute context configuration
///
/// A default configuration is provided via the [`default()`] method and
/// documented in the various fields of this struct. You can change these fields
/// to adjust the configuration, check out their documentation to see what their
/// default behavior is.
///
/// Once you're satisfied with the configuration, used the [`setup()`] method
/// to set up the Vulkan context.
///
/// [`default()`]: VulkanConfig::default()
/// [`setup()`]: VulkanConfig::setup()
pub struct VulkanConfig<
    MemAlloc: MemoryAllocator = StandardMemoryAllocator,
    CommAlloc: CommandBufferAllocator = StandardCommandBufferAllocator,
    DescAlloc: DescriptorSetAllocator = StandardDescriptorSetAllocator,
> {
    /// Window to which this Vulkan context is meant to render, if any
    #[cfg(feature = "livesim")]
    pub window: Option<Arc<Window>>,

    /// Decide which Vulkan layers should be enabled
    ///
    /// Layers are used for various debugging and profiling tasks. Note that
    /// you can use the `VK_INSTANCE_LAYERS` environment variable to activate
    /// layers, instead of activating them in code.
    ///
    /// By default, VK_LAYER_KHRONOS_validation is enabled on debug builds.
    pub layers: Box<dyn FnOnce(&VulkanLibrary) -> Vec<String>>,

    /// Decide which instance extensions should be enabled
    ///
    /// Instance extensions are mostly used for interoperability with other APIs
    /// (resource sharing between graphics APIs, surfaces from windowing
    /// APIs...), but were also occasionally used to plug holes in the Vulkan
    /// specification that affect device enumeration.
    ///
    /// By default, we require the ext_debug_utils extension in order to provide
    /// a good debugging experience. Additionally, in debug builds, we also
    /// enable khr_get_physical_device_properties2 on older Vulkan versions,
    /// which is a prerequisite for the device robustness features that we
    /// enable by default.
    ///
    /// If the gpu-debug-utils feature is enabled, then we force the debug_utils
    /// extension to be enabled, no matter if your configuration says otherwise,
    /// as it is required functionality for naming GPU objects.
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
    //
    // TODO: Consider trying these optional device features later:
    //       - compute_full_subgroups => Garantie d'avoir des subgroups complets
    //       - event => Possibilité de synchroniser des commandes par événement
    //       - shader_subgroup_uniform_control_flow => Plus de garanties sur la
    //         reconvergence des subgroups, utile pour les opérations collectives
    //          * khr_shader_subgroup_uniform_control_flow sur les vieux Vulkan
    //       - subgroup_size_control => Utile pour avoir des algos subgroups stables
    //          * ext_subgroup_size_control sur les vieux Vulkan
    //
    //       ...and these extensions:
    //       - ext_shader_subgroup_xyz => Opérations subgroups
    pub device_features_extensions: Box<dyn FnMut(&PhysicalDevice) -> (Features, DeviceExtensions)>,

    /// Impose additional device requirements
    ///
    /// Requirements on the compute device other than features and extensions
    /// to be enabled. This is where you can check for things like minimal
    /// extension versions, device limits, Vulkan API support...
    ///
    /// Return `true` to signify that a device meets your requirements. By
    /// default, no additional requirements are imposed.
    //
    // TODO: Full requirements may also want to check these device properties,
    //       among others depending on workload's specifics.
    //
    //       - max_bound_descriptor_sets (can be 1)
    //       - max_buffer_size (if supported)
    //       - max_compute_shared_memory_size
    //       - max_compute_work_group_count
    //       - max_compute_work_group_invocations
    //       - max_compute_work_group_size
    //       - max_compute_workgroup_subgroups (if supported)
    //       - max_descriptor_buffer_bindings
    //       - max_descriptor_set_<resource>
    //       - max_image_array_layers
    //       - max_image_dimension2_d
    //       - max_memory_allocation_count
    //       - max_memory_allocation_size
    //       - max_per_set_descriptors
    //       - max_per_stage_descriptor_<resource>
    //       - max_per_stage_resources
    //       - max_push_constants_size
    //       - max_sampler_allocation_count
    //       - max_storage_buffer_range
    //       - (max|min)_subgroup_size et required_subgroup_size_stages (if using subgroups)
    //       - max_uniform_buffer_range
    //       - subgroup_<properties> (if using subgroups)
    //       - subgroup_supported_(operations|stages) (if using subgroups)
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
    /// If a surface is passed in as a parameter, then the first selected
    /// queue should be able to perform compute operations and present to this
    /// surface.
    ///
    /// By default, we only allocate a single queue, suitable for compute (and
    /// thus data transfers). We try to pick the device's main queue family for
    /// this purpose, at it may be the most performant in single-queue use
    /// cases. If you want to experiment with multi-queue workflows, this is
    /// the tuning knob that you want.
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
    pub descriptor_allocator: Box<dyn FnOnce(Arc<Device>) -> DescAlloc>,
}
//
impl
    VulkanConfig<
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
    pub fn default() -> Self {
        let layers = Box::new(default_layers);
        let instance_extensions = Box::new(default_instance_extensions);
        let enumerate_portability = false;
        let device_features_extensions = Box::new(default_features_extensions);
        let other_device_requirements = Box::new(default_other_device_requirements);
        let device_preference = Box::new(default_device_preference);
        let queues = Box::new(default_queues);
        let memory_allocator = Box::new(default_memory_allocator);
        let command_allocator = Box::new(default_command_buffer_allocator);
        let descriptor_allocator = Box::new(StandardDescriptorSetAllocator::new);
        #[cfg(feature = "livesim")]
        {
            Self {
                window: None,
                layers,
                instance_extensions,
                enumerate_portability,
                device_features_extensions,
                other_device_requirements,
                device_preference,
                queues,
                memory_allocator,
                command_allocator,
                descriptor_allocator,
            }
        }
        #[cfg(not(feature = "livesim"))]
        {
            Self {
                layers,
                instance_extensions,
                enumerate_portability,
                device_features_extensions,
                other_device_requirements,
                device_preference,
                queues,
                memory_allocator,
                command_allocator,
                descriptor_allocator,
            }
        }
    }
}
//
impl<MemAlloc: MemoryAllocator, CommAlloc: CommandBufferAllocator>
    VulkanConfig<MemAlloc, CommAlloc>
{
    /// Set up a Vulkan compute context with this configuration
    ///
    /// ```
    /// # use compute::gpu::VulkanConfig;
    /// let context = VulkanConfig::default().setup()?;
    /// # Ok::<(), compute::gpu::Error>(())
    /// ```
    #[allow(unused_assignments, unused_mut)]
    pub fn setup(mut self) -> Result<VulkanContext<MemAlloc, CommAlloc>, Error> {
        let library = load_library()?;

        let mut instance_extensions = (self.instance_extensions)(&library);
        if cfg!(feature = "gpu-debug-utils") {
            instance_extensions.ext_debug_utils = true;
        }
        #[cfg(feature = "livesim")]
        {
            if self.window.is_some() {
                instance_extensions =
                    instance_extensions.union(&vulkano_win::required_extensions(&library));
            }
        }
        let instance = DebuggedInstance::setup(
            library.clone(),
            (self.layers)(&library),
            instance_extensions,
            self.enumerate_portability,
        )?;

        let mut surface = None;
        #[cfg(feature = "livesim")]
        {
            if let Some(window) = self.window {
                let new_surface = vulkano_win::create_surface_from_winit(window, instance.clone())?;
                info!(
                    "Created a surface from window with API {:?}",
                    new_surface.api()
                );
                surface = Some(new_surface);
            }
        }

        let physical_device = select_physical_device(
            &instance,
            |device| {
                let (features, extensions) = (self.device_features_extensions)(&device);
                device.supported_features().contains(&features)
                    && device.supported_extensions().contains(&extensions)
                    && (self.other_device_requirements)(device)
            },
            self.device_preference,
            surface.as_deref(),
        )?;

        let (features, mut extensions) = (self.device_features_extensions)(&physical_device);
        if surface.is_some() {
            extensions.khr_swapchain = true;
        }
        let (queues_info, queue_names) = (self.queues)(&physical_device, surface.as_deref());
        let (device, queues) =
            create_logical_device(physical_device.clone(), features, extensions, queues_info)?;

        assert_eq!(
            queues.len(),
            queue_names.len(),
            "Number of queue names doesn't match number of queues"
        );
        if cfg!(feature = "gpu-debug-utils") {
            for (queue, name) in queues.iter().zip(queue_names.into_iter()) {
                device.set_debug_utils_object_name(queue, Some(&name))?;
            }
        }

        let memory_allocator = Arc::new((self.memory_allocator)(device.clone()));
        let command_allocator = Arc::new((self.command_allocator)(device.clone()));
        let descriptor_allocator = (self.descriptor_allocator)(device.clone());

        let dirs = ProjectDirs::from("", "", "grayscott").ok_or(Error::HomeDirNotFound)?;
        let pipeline_cache = PersistentPipelineCache::new(&dirs, device.clone())?;

        Ok(VulkanContext {
            _messenger: instance._messenger,
            surface,
            device,
            queues,
            memory_allocator,
            command_allocator,
            pipeline_cache,
            descriptor_allocator,
        })
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

/// Suggested set of layers to enable
#[allow(unused_variables)]
fn default_layers(library: &VulkanLibrary) -> Vec<String> {
    if cfg!(debug_assertions) {
        vec!["VK_LAYER_KHRONOS_validation".to_owned()]
    } else {
        vec![]
    }
}

/// Suggested set of instance extensions to enable
fn default_instance_extensions(library: &VulkanLibrary) -> InstanceExtensions {
    let mut suggested_extensions = InstanceExtensions {
        ext_debug_utils: true,
        ..Default::default()
    };
    if cfg!(debug_assertions) && library.api_version() < Version::V1_1 {
        suggested_extensions.khr_get_physical_device_properties2 = true;
    }
    suggested_extensions
}

/// Suggested device features and extensions to enable at the logical device
/// creation stage
fn default_features_extensions(device: &PhysicalDevice) -> (Features, DeviceExtensions) {
    let mut features = Features::empty();
    let mut extensions = DeviceExtensions::empty();
    if cfg!(debug_assertions) {
        features.robust_buffer_access = true;
        features.robust_image_access = true;
        features.robust_buffer_access2 = true;
        features.robust_image_access2 = true;
        extensions.ext_robustness2 = true;
        if device.api_version() < Version::V1_3 {
            extensions.ext_image_robustness = true;
        }
    }
    if device.supported_extensions().khr_portability_subset {
        extensions.khr_portability_subset = true;
    }
    (features, extensions)
}

/// Suggested device requirements
#[allow(unused_variables)]
fn default_other_device_requirements(device: &PhysicalDevice) -> bool {
    true
}

/// Suggested device preference
fn default_device_preference(device1: &PhysicalDevice, device2: &PhysicalDevice) -> Ordering {
    let device_type_score = |device: &PhysicalDevice| match device.properties().device_type {
        PhysicalDeviceType::DiscreteGpu => 5,
        PhysicalDeviceType::VirtualGpu => 4,
        PhysicalDeviceType::IntegratedGpu => 3,
        PhysicalDeviceType::Cpu => 2,
        PhysicalDeviceType::Other => 1,
        _ => 0,
    };
    device_type_score(device1).cmp(&device_type_score(device2))
}

/// Suggested single-queue creation info
///
/// Will pick a single queue, in the family which is presumed to be the "main"
/// queue family of the the device.
fn default_queues(
    device: &PhysicalDevice,
    surface: Option<&Surface>,
) -> (Vec<QueueCreateInfo>, Vec<Cow<'static, str>>) {
    let (queue_family_index, _) = device
        .queue_family_properties()
        .iter()
        .enumerate()
        .map(|(idx, family)| (idx as u32, family))
        .filter(|(idx, family)| {
            // We need a queue with compute support
            if !family.queue_flags.contains(QueueFlags::COMPUTE) {
                return false;
            }

            // If a surface is specified, then we must be able to present to
            // that surface too
            surface.map_or(true, |surface| {
                device.surface_support(*idx, surface).unwrap_or(false)
            })
        })
        .max_by(|(idx1, queue1), (idx2, queue2)| {
            // Queues that support graphics are most likely to be the main queue
            let supports_graphics =
                |queue: &QueueFamilyProperties| queue.queue_flags.contains(QueueFlags::GRAPHICS);
            match (supports_graphics(queue1), supports_graphics(queue2)) {
                (true, false) => return Ordering::Greater,
                (false, true) => return Ordering::Less,
                (false, false) | (true, true) => {}
            }

            // As a last resort, pick the queue that comes first in the list
            idx2.cmp(idx1)
        })
        .expect("There should be at least one queue family");

    (
        vec![QueueCreateInfo {
            queue_family_index,
            ..Default::default()
        }],
        vec!["Main queue".into()],
    )
}

/// Suggested memory allocator
fn default_memory_allocator(device: Arc<Device>) -> StandardMemoryAllocator {
    StandardMemoryAllocator::new_default(device)
}

/// Suggested command buffer allocator
fn default_command_buffer_allocator(device: Arc<Device>) -> StandardCommandBufferAllocator {
    StandardCommandBufferAllocator::new(device, StandardCommandBufferAllocatorCreateInfo::default())
}

/// Load the Vulkan library
fn load_library() -> Result<Arc<VulkanLibrary>, Error> {
    let library = VulkanLibrary::new()?;
    info!("Loaded Vulkan library");
    trace!("- Supports Vulkan v{}", library.api_version());
    trace!(
        "- Supports instance extensions {}",
        format_extension_properties(library.extension_properties())
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

/// Vulkan instance with debug logging
///
/// Logging will stop once this struct is dropped, even if there are
/// other Arc<Instance> remaining in flight
struct DebuggedInstance {
    /// Vulkan instance
    instance: Arc<Instance>,

    /// Messenger that logs instance debug messages
    _messenger: Option<DebugUtilsMessenger>,
}
//
impl DebuggedInstance {
    /// Set up a Vulkan instance
    ///
    /// This is the point where layers and instance extensions are enabled. See
    /// [`suggested_layers()`] and [`suggested_instance_extensions()`] for suggested
    /// layers and instance extensions.
    ///
    /// If you set `enumerate_portability` to `true` here, you will be able to use
    /// devices that do not fully conform to the Vulkan specification, like MoltenVK
    /// on macOS and iOS, but in that case your device requirements and preferences
    /// must account for the associated [non-conforming
    /// behavior](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_portability_subset.html).
    fn setup(
        library: Arc<VulkanLibrary>,
        enabled_layers: Vec<String>,
        enabled_extensions: InstanceExtensions,
        enumerate_portability: bool,
    ) -> Result<DebuggedInstance, Error> {
        let unsupported_extensions = *library.supported_extensions()
            - library
                .supported_extensions_with_layers(enabled_layers.iter().map(String::as_ref))?;
        if unsupported_extensions != InstanceExtensions::empty() {
            debug!(
                "Selected layer(s) {enabled_layers:?} do NOT support extensions {unsupported_extensions:#?}"
            );
        }

        let create_info = InstanceCreateInfo {
            enabled_extensions,
            enabled_layers,
            enumerate_portability,
            ..InstanceCreateInfo::application_from_cargo_toml()
        };
        info!("Will now create a Vulkan instance with configuration {create_info:#?}");

        let result = if enabled_extensions.ext_debug_utils {
            type DUMS = DebugUtilsMessageSeverity;
            type DUMT = DebugUtilsMessageType;
            let mut debug_messenger_info = DebugUtilsMessengerCreateInfo {
                message_severity: DUMS::empty(),
                message_type: DUMT::GENERAL,
                ..DebugUtilsMessengerCreateInfo::user_callback(Arc::new(|message: &Message| {
                    // SAFETY: This callback must not call into Vulkan APIs
                    let level = match message.severity {
                        DUMS::ERROR => log::Level::Error,
                        DUMS::WARNING => log::Level::Warn,
                        DUMS::INFO => log::Level::Debug,
                        DUMS::VERBOSE => log::Level::Trace,
                        _ => log::Level::Info,
                    };
                    let target = message
                        .layer_prefix
                        .map(|layer| format!("Vulkan {:?} {layer}", message.ty))
                        .unwrap_or(format!("Vulkan {:?}", message.ty));
                    log!(target: &target, level, "{}", message.description);
                }))
            };
            if log::STATIC_MAX_LEVEL >= log::Level::Error {
                debug_messenger_info.message_severity |= DUMS::ERROR;
            }
            if log::STATIC_MAX_LEVEL >= log::Level::Warn {
                debug_messenger_info.message_severity |= DUMS::WARNING;
            }
            if log::STATIC_MAX_LEVEL >= log::Level::Debug {
                debug_messenger_info.message_severity |= DUMS::INFO;
            }
            if log::STATIC_MAX_LEVEL >= log::Level::Trace {
                debug_messenger_info.message_severity |= DUMS::VERBOSE;
            }
            if cfg!(debug_assertions) {
                debug_messenger_info.message_type |= DUMT::VALIDATION | DUMT::PERFORMANCE;
            };
            let instance = unsafe {
                // Safe because our logger does not call into Vulkan APIs
                Instance::with_debug_utils_messengers(
                    library,
                    create_info,
                    std::iter::once(debug_messenger_info.clone()),
                )?
            };
            let messenger =
                // Safe because our logger does not call into Vulkan APIs
                unsafe { DebugUtilsMessenger::new(instance.clone(), debug_messenger_info)? };
            Self {
                instance,
                _messenger: Some(messenger),
            }
        } else {
            Self {
                instance: Instance::new(library, create_info)?,
                _messenger: None,
            }
        };

        trace!("Vulkan instance supports Vulkan v{}", result.api_version());

        Ok(result)
    }
}
//
impl Deref for DebuggedInstance {
    type Target = Arc<Instance>;

    fn deref(&self) -> &Self::Target {
        &self.instance
    }
}

/// Pick a physical device to run the simulation on
///
/// As currently written, the simulation code does not support running on
/// multiple devices. This would add quite a bit of complexity due to the need
/// for efficient data transfers and synchronization between devices, and it is
/// often possible to just run independent simulations on each device instead.
///
/// Therefore, this code picks a single physical device, based on device
/// requirements specified through the `requirements` callback and device
/// preferences (among viable devices) specified through the `preference` callback.
///
/// If rendering to a window is enabled, device compatibility with the window's
/// surface is also checked.
///
/// See [`suggested_requirements()`] for suggested device requirements that you
/// should AND with your own requirements.
///
/// If several devices compare equal according to the `preference` callback, the
/// first matching device will be selected.
fn select_physical_device(
    instance: &DebuggedInstance,
    mut requirements: impl FnMut(&PhysicalDevice) -> bool,
    mut preference: impl FnMut(&PhysicalDevice, &PhysicalDevice) -> Ordering,
    surface: Option<&Surface>,
) -> Result<Arc<PhysicalDevice>, Error> {
    let selected_device = instance
        .enumerate_physical_devices()?
        .inspect(|device| {
            info!("Found physical device {}", device.properties().device_name);
            log_device_description(&device, surface);
        })
        .filter(|device| {
            let can_use = requirements(device)
                && surface
                    .as_ref()
                    .map_or(true, |surface| device_supports_surface(device, surface));
            if can_use {
                info!("=> Device meets requirements");
            } else {
                info!("=> Device does NOT meet requirements");
            }
            can_use
        })
        // Using minimum ensures we pick the first device given equal preference
        .min_by(|a, b| preference(a, b).reverse());
    if let Some(device) = selected_device {
        info!("Selected device {}", device.properties().device_name);
        Ok(device)
    } else {
        Err(Error::NoMatchingDevice)
    }
}

/// Log a description of the device at higher log levels
fn log_device_description(device: &PhysicalDevice, surface: Option<&Surface>) {
    trace!("- With {:#?}", device.properties());
    trace!(
        "- With device extensions {}",
        format_extension_properties(device.extension_properties())
    );
    trace!("- With features {:#?}", device.supported_features());
    trace!("- With {:#?}", device.memory_properties());
    trace!(
        "- With queue families {:#?}",
        device.queue_family_properties()
    );
    if device.api_version() >= Version::V1_3 || device.supported_extensions().ext_tooling_info {
        trace!("- With tools {:#?}", device.tool_properties().unwrap());
    }
    if let Some(surface) = surface {
        trace!("- And when it comes to the requested drawing surface...");
        trace!(
            "  * Can present from queue families {:?}",
            device
                .queue_family_properties()
                .iter()
                .enumerate()
                .filter_map(|(idx, _family)| {
                    device
                        .surface_support(idx as u32, surface)
                        .unwrap_or(false)
                        .then_some(idx)
                })
                .collect::<Vec<_>>()
        );
        trace!(
            "  * Supports present modes {:?}",
            device
                .surface_present_modes(surface)
                .map(|iter| iter.collect::<Vec<_>>())
                .unwrap_or_default()
        );
        let surface_info = SurfaceInfo::default();
        trace!(
            "  * Supports surface formats {:?}",
            device
                .surface_formats(surface, surface_info.clone())
                .unwrap_or_default()
        );
        trace!(
            "  * Supports surface capabilities {:#?}",
            device.surface_capabilities(surface, surface_info),
        );
    }
}

/// Truth that a device can present to a certain surface via a swapchain
fn device_supports_surface(device: &PhysicalDevice, surface: &Surface) -> bool {
    let supports_swapchain = device.supported_extensions().khr_swapchain;
    let can_present = device
        .queue_family_properties()
        .iter()
        .enumerate()
        .any(|(idx, family)| {
            family.queue_flags.contains(QueueFlags::COMPUTE)
                && device.surface_support(idx as u32, surface).unwrap_or(false)
        });
    supports_swapchain && can_present
}

/// Create a logical device and associated command queues
///
/// This is the point where optional core Vulkan features and extensions are
/// enabled, and where the desired queue configuration is specified.
///
/// See [`suggested_features_extensions()`] for suggestions of features and
/// extensions to enable.
///
/// Queues must be picked manually because the choice of one queue family can
/// influence that of another (e.g. you want to pick a different queue family
/// for async data transfers than the one you use for compute). You can use
/// [`suggested_queue()`] as your starting point, which tries to picks the main
/// queue of the device.
fn create_logical_device(
    physical_device: Arc<PhysicalDevice>,
    enabled_features: Features,
    enabled_extensions: DeviceExtensions,
    queue_create_infos: Vec<QueueCreateInfo>,
) -> Result<(Arc<Device>, Box<[Arc<Queue>]>), Error> {
    let create_info = DeviceCreateInfo {
        enabled_features,
        enabled_extensions,
        queue_create_infos,
        ..Default::default()
    };
    info!("Will now create a logical device with configuration {create_info:#?}");
    let (device, queues) = Device::new(physical_device, create_info)?;
    Ok((device, queues.collect()))
}

/// GPU pipeline cache (to avoid things like shader recompilations)
pub struct PersistentPipelineCache {
    /// In-RAM cache
    cache: Arc<PipelineCache>,

    /// Path to be used for on-disk persistence
    path: PathBuf,
}
//
impl PersistentPipelineCache {
    /// Attempt to load the pipeline cache from disk, otherwise create a new one
    fn new(dirs: &ProjectDirs, device: Arc<Device>) -> Result<Self, Error> {
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
    fn write(&self) -> Result<(), Error> {
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
    fn setup_vulkan() -> Result<(), Error> {
        init_logger();
        VulkanConfig {
            enumerate_portability: true,
            ..VulkanConfig::default()
        }
        .setup()?;
        Ok(())
    }
}
