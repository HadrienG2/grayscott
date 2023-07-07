//! Device management

use super::{ContextBuildError, ContextBuildResult};
#[allow(unused_imports)]
use log::{debug, error, info, log, trace, warn};
use std::{cmp::Ordering, ops::Deref, sync::Arc};
use vulkano::{
    device::{
        physical::PhysicalDevice, Device, DeviceCreateInfo, DeviceExtensions, Features, Queue,
        QueueCreateInfo, QueueFamilyProperties, QueueFlags,
    },
    instance::{Instance, Version},
    swapchain::{Surface, SurfaceInfo},
};

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
/// If several devices compare equal according to the `preference` callback, the
/// first matching device will be selected.
pub fn select_physical(
    instance: &Arc<Instance>,
    mut features_extensions: impl FnMut(&PhysicalDevice) -> (Features, DeviceExtensions),
    mut other_requirements: impl FnMut(&PhysicalDevice) -> bool,
    mut preference: impl FnMut(&PhysicalDevice, &PhysicalDevice) -> Ordering,
    mut surface_and_reqs: &mut Option<(
        Arc<Surface>,
        impl FnMut(&PhysicalDevice, &Surface) -> bool,
    )>,
) -> ContextBuildResult<Arc<PhysicalDevice>> {
    let selected_device = instance
        .enumerate_physical_devices()?
        .filter(|device| {
            // Log information on every device we find
            info!("Found physical device {}", device.properties().device_name);
            log_description(device, surface_and_reqs.as_ref().map(|t| t.0.deref()));

            // Check if the device is usable
            let (features, extensions) = (features_extensions)(device);
            let mut can_use = device.supported_features().contains(&features)
                && device.supported_extensions().contains(&extensions)
                && (other_requirements)(device);
            if let Some((surface, requirements)) = &mut surface_and_reqs {
                can_use |= supports_surface(device, surface) && requirements(device, surface);
            }

            // Log device usability and proceed
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
        Err(ContextBuildError::NoMatchingDevice)
    }
}

/// Log a description of the device at higher log levels
fn log_description(device: &PhysicalDevice, surface: Option<&Surface>) {
    trace!("- With {:#?}", device.properties());
    trace!(
        "- With device extensions {}",
        super::format_extension_properties(device.extension_properties())
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
            present_queues(device, surface)
                .map(|(idx, _family)| idx)
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

/// Truth that a device can render to a certain surface via a compute pipeline
fn supports_surface(device: &PhysicalDevice, surface: &Surface) -> bool {
    let supports_swapchain = device.supported_extensions().khr_swapchain;
    let can_present = present_queues(device, surface)
        .any(|(_idx, family)| family.queue_flags.contains(QueueFlags::COMPUTE));
    supports_swapchain && can_present
}

/// Enumerate queue families which can present to a surface
fn present_queues<'input>(
    device: &'input PhysicalDevice,
    surface: &'input Surface,
) -> impl Iterator<Item = (u32, &'input QueueFamilyProperties)> {
    device
        .queue_family_properties()
        .iter()
        .enumerate()
        .map(|(idx, family)| (idx as u32, family))
        .filter(move |(idx, _family)| device.surface_support(*idx, surface).unwrap_or(false))
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
#[allow(clippy::type_complexity)]
pub fn create_logical(
    physical_device: Arc<PhysicalDevice>,
    enabled_features: Features,
    enabled_extensions: DeviceExtensions,
    (queue_create_infos, queue_names): (Vec<QueueCreateInfo>, Vec<impl AsRef<str>>),
) -> ContextBuildResult<(Arc<Device>, Box<[Arc<Queue>]>)> {
    // Configure device and queues
    let create_info = DeviceCreateInfo {
        enabled_features,
        enabled_extensions,
        queue_create_infos,
        ..Default::default()
    };
    info!("Will now create a logical device with {create_info:#?}");
    let (device, queues) = Device::new(physical_device, create_info)?;

    // Do not rename device (its default name is very good), but do name queues
    let queues = queues.collect::<Box<[_]>>();
    assert_eq!(
        queues.len(),
        queue_names.len(),
        "Number of queue names doesn't match number of queues"
    );
    if cfg!(feature = "gpu-debug-utils") {
        for (queue, name) in queues.iter().zip(queue_names) {
            device.set_debug_utils_object_name(queue, Some(name.as_ref()))?;
        }
    }
    Ok((device, queues))
}
