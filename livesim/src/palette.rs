//! Color palette management

use crate::{
    context::SimulationContext,
    pipeline::{self},
    Result,
};
use std::sync::Arc;
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage},
    command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferToImageInfo},
    descriptor_set::DescriptorSet,
    format::{Format, NumericFormat},
    image::{
        sampler::{Filter, SamplerCreateInfo},
        Image, ImageAspects, ImageCreateInfo, ImageType, ImageUsage,
    },
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
    pipeline::ComputePipeline,
    swapchain::ColorSpace,
    sync::GpuFuture,
};

// TODO: Expose device requirements

/// Surface formats that this color palette is compatible with
pub fn is_supported_surface_format((format, colorspace): (Format, ColorSpace)) -> bool {
    let Some(color_type) = format.numeric_format_color() else {
        return false;
    };
    format.aspects().contains(ImageAspects::COLOR)
        && format.components().iter().take(3).all(|&bits| bits > 0)
        && color_type == NumericFormat::UNORM
        && colorspace == ColorSpace::SrgbNonLinear
}

/// Create the color palette used for data -> color translation
///
/// Returns a GPU future that will be signaled once the palette has been
/// uploaded to the GPU, along with a descriptor set that can be used in order
/// to bind the palette to the rendering pipeline.
pub fn create(
    context: &SimulationContext,
    pipeline: &ComputePipeline,
    resolution: u32,
) -> Result<(impl GpuFuture, Arc<DescriptorSet>)> {
    // Prepare to upload color palette
    assert!(
        resolution >= 2,
        "Color palette must have at least two endpoints"
    );
    let vulkan = context.vulkan();
    let upload_queue = context.queue();
    let mut upload_builder = AutoCommandBufferBuilder::primary(
        vulkan.command_allocator.clone(),
        upload_queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )?;

    // Create color palette image, record upload commands
    // FIXME: Expose requirements
    let palette_image = Image::new(
        vulkan.memory_allocator.clone(),
        ImageCreateInfo {
            image_type: ImageType::Dim1d,
            format: Format::B8G8R8A8_UNORM,
            extent: [resolution, 1, 1],
            usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
            ..Default::default()
        },
        AllocationCreateInfo::default(),
    )?;
    vulkan.set_debug_utils_object_name(&palette_image, || "Color palette".into())?;

    // Record upload command
    let palette_buffer = Buffer::from_iter(
        vulkan.memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_SRC,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        colors(resolution),
    )?;
    vulkan.set_debug_utils_object_name(palette_buffer.buffer(), || "Color palette".into())?;
    upload_builder.copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(
        palette_buffer,
        palette_image.clone(),
    ))?;

    // Create palette descriptor set
    let palette = pipeline::new_palette_set(vulkan, pipeline, palette_image.clone())?;

    // Schedule upload
    let upload = upload_builder.build()?;
    vulkan.set_debug_utils_object_name(&upload, || "Color palette upload".into())?;
    let upload_future =
        vulkano::sync::now(vulkan.device.clone()).then_execute(upload_queue.clone(), upload)?;
    Ok((upload_future, palette))
}

/// Desired sampler configuration
pub fn sampler_conig() -> SamplerCreateInfo {
    SamplerCreateInfo {
        mag_filter: Filter::Linear,
        min_filter: Filter::Linear,
        ..Default::default()
    }
}

/// Generate color palette for a given resolution
fn colors(resolution: u32) -> impl ExactSizeIterator<Item = [u8; 4]> {
    (0..resolution).map(move |idx| {
        let position = idx as f64 / (resolution - 1) as f64;
        let color = ui::GRADIENT.eval_continuous(position);
        [color.b, color.g, color.r, 255]
    })
}
