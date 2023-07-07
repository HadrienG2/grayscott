//! Rendering surface management and swapchain creation

use crate::{context::SimulationContext, input::Input, palette, pipeline, Result};
use compute::gpu::context::VulkanContext;
use data::concentration::gpu::shape::Shape;
use log::info;
use std::sync::Arc;
use vulkano::{
    descriptor_set::PersistentDescriptorSet,
    device::physical::PhysicalDevice,
    image::SwapchainImage,
    pipeline::ComputePipeline,
    swapchain::{Surface, SurfaceInfo, Swapchain, SwapchainCreateInfo},
};
use winit::{
    dpi::PhysicalSize,
    event_loop::EventLoop,
    window::{Theme, Window, WindowBuilder},
};

/// Surface-dependent device requirements
pub fn requirements(device: &PhysicalDevice, surface: &Surface) -> bool {
    let surface_info = SurfaceInfo::default();
    device
        .surface_formats(surface, surface_info.clone())
        .map(|vec| vec.into_iter().any(palette::is_supported_surface_format))
        .unwrap_or(false)
        && device
            .surface_capabilities(surface, surface_info)
            .map(|caps| {
                caps.max_image_count.unwrap_or(u32::MAX) >= MIN_SWAPCHAIN_IMAGES
                    && caps
                        .supported_usage_flags
                        .contains(pipeline::output_usage())
            })
            .unwrap_or(false)
}

/// Set up a window and associated event loop
pub fn create_window(shape: Shape) -> Result<(EventLoop<()>, Arc<Window>)> {
    let event_loop = EventLoop::new();
    let window = Arc::new(
        WindowBuilder::new()
            .with_inner_size(PhysicalSize::new(shape.width(), shape.height()))
            .with_resizable(false)
            .with_title("Gray-Scott reaction")
            .with_visible(false)
            .with_theme(Some(Theme::Dark))
            .build(&event_loop)?,
    );
    Ok((event_loop, window))
}

/// Create a swapchain
pub fn create_swapchain(
    context: &SimulationContext,
) -> Result<(Arc<Swapchain>, Vec<Arc<SwapchainImage>>)> {
    let vulkan = context.vulkan();
    let physical_device = vulkan.device.physical_device();
    let surface = context.surface();

    let surface_info = SurfaceInfo::default();
    let surface_capabilities =
        physical_device.surface_capabilities(surface, surface_info.clone())?;
    let (image_format, image_color_space) = physical_device
        .surface_formats(surface, surface_info)?
        .into_iter()
        .find(|format| palette::is_supported_surface_format(*format))
        .expect("There should be one valid format (checked at device creation time)");

    let create_info = SwapchainCreateInfo {
        min_image_count: surface_capabilities
            .min_image_count
            .max(MIN_SWAPCHAIN_IMAGES),
        image_format: Some(image_format),
        image_color_space,
        image_usage: pipeline::output_usage(),
        ..Default::default()
    };
    info!("Will now create a swapchain with {create_info:#?}");

    let (swapchain, swapchain_images) =
        Swapchain::new(vulkan.device.clone(), surface.clone(), create_info)?;
    vulkan.set_debug_utils_object_name(&swapchain, || "Rendering swapchain".into())?;

    Ok((swapchain, swapchain_images))
}

/// Recreate the swapchain if it went invalid, along with the I/O descriptor
/// sets associated with the inputs and swapchain images
pub fn recreate_swapchain(
    vulkan: &VulkanContext,
    pipeline: &ComputePipeline,
    upload_buffers: &[Input],
    swapchain: &Arc<Swapchain>,
) -> Result<(Arc<Swapchain>, Vec<Arc<PersistentDescriptorSet>>)> {
    let (swapchain, images) = swapchain.recreate(swapchain.create_info())?;
    let inout_sets = pipeline::new_inout_sets(vulkan, &pipeline, upload_buffers, images)?;
    Ok((swapchain, inout_sets))
}

/// Minimum number of swapchain images
const MIN_SWAPCHAIN_IMAGES: u32 = 2;
