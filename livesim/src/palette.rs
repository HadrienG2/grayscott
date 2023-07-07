//! Color palette management

use crate::{
    context::SimulationContext,
    pipeline::{self},
    Result,
};
use std::sync::Arc;
use vulkano::{
    command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage},
    descriptor_set::PersistentDescriptorSet,
    format::{Format, NumericType},
    image::{ImageAccess, ImageAspects, ImageDimensions, ImmutableImage, MipmapsCount},
    pipeline::ComputePipeline,
    sampler::{Filter, SamplerCreateInfo},
    swapchain::ColorSpace,
    sync::GpuFuture,
};

// TODO: Expose device requirements

/// Surface formats that this color palette is compatible with
pub fn is_supported_surface_format((format, colorspace): (Format, ColorSpace)) -> bool {
    let Some(color_type) = format.type_color() else {
        return false;
    };
    format.aspects().contains(ImageAspects::COLOR)
        && format.components().iter().take(3).all(|&bits| bits > 0)
        && color_type == NumericType::UNORM
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
) -> Result<(impl GpuFuture, Arc<PersistentDescriptorSet>)> {
    // Prepare to upload color palette
    assert!(
        resolution >= 2,
        "Color palette must have at least two endpoints"
    );
    let vulkan = context.vulkan();
    let upload_queue = context.queue();
    let mut upload_builder = AutoCommandBufferBuilder::primary(
        &vulkan.command_allocator,
        upload_queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )?;

    // Create color palette, record upload commands
    // FIXME: Expose requirements
    let palette_image = ImmutableImage::from_iter(
        &vulkan.memory_allocator,
        colors(resolution),
        ImageDimensions::Dim1d {
            width: resolution,
            array_layers: 1,
        },
        MipmapsCount::One,
        Format::B8G8R8A8_UNORM,
        &mut upload_builder,
    )?;
    vulkan.set_debug_utils_object_name(&palette_image.inner().image, || "Color palette".into())?;

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
fn colors(resolution: u32) -> impl Iterator<Item = [u8; 4]> + ExactSizeIterator {
    (0..resolution).map(move |idx| {
        let position = idx as f64 / (resolution - 1) as f64;
        let color = ui::GRADIENT.eval_continuous(position);
        [color.b, color.g, color.r, 255]
    })
}
