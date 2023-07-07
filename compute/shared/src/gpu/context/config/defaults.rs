//! Default Vulkan context configuration

use super::VulkanConfig;
#[allow(unused_imports)]
use log::{debug, error, info, log, trace, warn};
use std::{borrow::Cow, cmp::Ordering, env::VarError, sync::Arc};
use vulkano::{
    command_buffer::allocator::{
        StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo,
    },
    descriptor_set::allocator::StandardDescriptorSetAllocator,
    device::{
        physical::{PhysicalDevice, PhysicalDeviceType},
        Device, DeviceExtensions, Features, QueueCreateInfo, QueueFamilyProperties, QueueFlags,
    },
    instance::{InstanceExtensions, Version},
    memory::allocator::StandardMemoryAllocator,
    swapchain::Surface,
    VulkanLibrary,
};

/// Suggested VulkanConfig
pub fn config() -> VulkanConfig {
    let layers = Box::new(layers);
    let instance_extensions = Box::new(instance_extensions);
    let enumerate_portability = false;
    let device_features_extensions = Box::new(device_features_extensions);
    let other_device_requirements = Box::new(other_device_requirements);
    let device_preference = Box::new(device_preference);
    let queues = Box::new(queues);
    let memory_allocator = Box::new(memory_allocator);
    let command_allocator = Box::new(command_buffer_allocator);
    let descriptor_set_allocator = Box::new(descriptor_set_allocator);
    #[cfg(feature = "livesim")]
    {
        VulkanConfig {
            window_and_reqs: None,
            layers,
            instance_extensions,
            enumerate_portability,
            device_features_extensions,
            other_device_requirements,
            device_preference,
            queues,
            memory_allocator,
            command_allocator,
            descriptor_set_allocator,
        }
    }
    #[cfg(not(feature = "livesim"))]
    {
        VulkanConfig {
            layers,
            instance_extensions,
            enumerate_portability,
            device_features_extensions,
            other_device_requirements,
            device_preference,
            queues,
            memory_allocator,
            command_allocator,
            descriptor_set_allocator,
        }
    }
}

/// Suggested set of instance layers
#[allow(unused_variables)]
pub fn layers(library: &VulkanLibrary) -> Vec<String> {
    if cfg!(debug_assertions) {
        vec!["VK_LAYER_KHRONOS_validation".to_owned()]
    } else {
        vec![]
    }
}

/// Suggested set of instance extensions
pub fn instance_extensions(library: &VulkanLibrary) -> InstanceExtensions {
    let mut suggested_extensions = InstanceExtensions {
        ext_debug_utils: true,
        ..Default::default()
    };
    if cfg!(debug_assertions) && library.api_version() < Version::V1_1 {
        suggested_extensions.khr_get_physical_device_properties2 = true;
    }
    suggested_extensions
}

/// Suggested device features and extensions
pub fn device_features_extensions(device: &PhysicalDevice) -> (Features, DeviceExtensions) {
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

/// Suggested other device requirements
#[allow(unused_variables)]
pub fn other_device_requirements(device: &PhysicalDevice) -> bool {
    true
}

/// Suggested device preference
pub fn device_preference(device1: &PhysicalDevice, device2: &PhysicalDevice) -> Ordering {
    let preferred_device_type = match std::env::var("GRAYSCOTT_PREFER_DEVICE") {
        Ok(string) => match string.as_str() {
            "" | "discrete" => PhysicalDeviceType::DiscreteGpu,
            "integrated" => PhysicalDeviceType::IntegratedGpu,
            "virtual" => PhysicalDeviceType::VirtualGpu,
            "cpu" => PhysicalDeviceType::Cpu,
            "other" => PhysicalDeviceType::Other,
            unknown => panic!("GRAYSCOTT_PREFER_DEVICE contains unknown device type {unknown}"),
        },
        Err(VarError::NotPresent) => PhysicalDeviceType::DiscreteGpu,
        Err(VarError::NotUnicode(s)) => {
            panic!("GRAYSCOTT_PREFER_DEVICE contains non-unicode data : {s:?}")
        }
    };
    let device_type_score = |device: &PhysicalDevice| match device.properties().device_type {
        x if x == preferred_device_type => 6,
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
/// queue family of the the device, with compute support.
pub fn queues(
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
pub fn memory_allocator(device: Arc<Device>) -> StandardMemoryAllocator {
    StandardMemoryAllocator::new_default(device)
}

/// Suggested command buffer allocator
pub fn command_buffer_allocator(device: Arc<Device>) -> StandardCommandBufferAllocator {
    StandardCommandBufferAllocator::new(device, StandardCommandBufferAllocatorCreateInfo::default())
}

/// Suggested descriptor set allocator
pub fn descriptor_set_allocator(device: Arc<Device>) -> StandardDescriptorSetAllocator {
    StandardDescriptorSetAllocator::new(device)
}
