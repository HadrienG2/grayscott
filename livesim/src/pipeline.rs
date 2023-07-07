//! Rendering pipeline

use crate::{input::Input, palette, Result, SimulationContext};
use compute::gpu::context::VulkanContext;
use data::concentration::gpu::shape::{self, PartialWorkGroupError, Shape};
use std::sync::Arc;
use vulkano::{
    buffer::BufferUsage,
    command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer},
    descriptor_set::{
        layout::DescriptorSetLayoutCreateInfo, PersistentDescriptorSet, WriteDescriptorSet,
    },
    image::{
        view::{ImageView, ImageViewCreateInfo},
        ImageUsage, ImmutableImage, SwapchainImage,
    },
    pipeline::{ComputePipeline, Pipeline, PipelineBindPoint},
    sampler::Sampler,
};

/// Dispatch size required by this pipeline, for a certain simulation domain and
/// work-group shape
pub fn dispatch_size(
    domain_shape: Shape,
    work_group_shape: Shape,
) -> Result<[u32; 3], PartialWorkGroupError> {
    shape::full_dispatch_size(domain_shape, work_group_shape)
}

// TODO: Expose device requirements

/// Create the pipeline
pub fn create(vulkan: &VulkanContext, work_group_shape: Shape) -> Result<Arc<ComputePipeline>> {
    // Load the rendering shader
    let shader = shader::load(vulkan.device.clone())?;
    vulkan.set_debug_utils_object_name(&shader, || "Live renderer shader".into())?;

    // Set up the rendering pipeline
    let pipeline = ComputePipeline::new(
        vulkan.device.clone(),
        shader.entry_point("main").expect("Should be present"),
        &shader::SpecializationConstants {
            constant_0: work_group_shape.width(),
            constant_1: work_group_shape.height(),
            amplitude_scale: ui::AMPLITUDE_SCALE,
        },
        Some(vulkan.pipeline_cache.clone()),
        sampler_setup_callback(vulkan)?,
    )?;
    vulkan.set_debug_utils_object_name(&pipeline, || "Live renderer".into())?;
    Ok(pipeline)
}

/// Manner in which the color palette is used
pub fn palette_usage() -> ImageUsage {
    ImageUsage::SAMPLED
}

/// Create the color palette descriptor set
pub fn new_palette_set(
    vulkan: &VulkanContext,
    pipeline: &ComputePipeline,
    palette_image: Arc<ImmutableImage>,
) -> Result<Arc<PersistentDescriptorSet>> {
    let palette_info = ImageViewCreateInfo {
        usage: palette_usage(),
        ..ImageViewCreateInfo::from_image(&palette_image)
    };
    let palette = PersistentDescriptorSet::new(
        &vulkan.descriptor_set_allocator,
        pipeline.layout().set_layouts()[PALETTE_SET as usize].clone(),
        [WriteDescriptorSet::image_view(
            PALETTE,
            ImageView::new(palette_image, palette_info)?,
        )],
    )?;
    // FIXME: Name this descriptor set once vulkano allows for it
    Ok(palette)
}

/// Manner in which the input is used
pub fn input_usage() -> BufferUsage {
    BufferUsage::STORAGE_BUFFER
}

/// Manner in which the rendering surface is used
pub fn output_usage() -> ImageUsage {
    ImageUsage::STORAGE
}

/// Create descriptor sets for each (upload buffer, swapchain image) pair
pub fn new_inout_sets(
    vulkan: &VulkanContext,
    pipeline: &ComputePipeline,
    upload_buffers: &[Input],
    swapchain_images: Vec<Arc<SwapchainImage>>,
) -> Result<Vec<Arc<PersistentDescriptorSet>>> {
    assert_eq!(upload_buffers.len(), swapchain_images.len());
    // FIXME: Name swapchain images once vulkano allows for it
    let set_layout = &pipeline.layout().set_layouts()[INOUT_SET as usize];
    upload_buffers
        .iter()
        .zip(swapchain_images.into_iter())
        .map(|(buffer, swapchain_image)| {
            let output_info = ImageViewCreateInfo {
                usage: output_usage(),
                ..ImageViewCreateInfo::from_image(&swapchain_image)
            };
            let descriptor_set = PersistentDescriptorSet::new(
                &vulkan.descriptor_set_allocator,
                set_layout.clone(),
                [
                    WriteDescriptorSet::buffer(DATA_INPUT, buffer.clone()),
                    WriteDescriptorSet::image_view(
                        SCREEN_OUTPUT,
                        ImageView::new(swapchain_image, output_info)?,
                    ),
                ],
            )?;
            // FIXME: Name descriptor set once vulkano allows for it
            Ok(descriptor_set)
        })
        .collect()
}

/// Record rendering commands
pub fn record_render_commands(
    context: &SimulationContext,
    pipeline: Arc<ComputePipeline>,
    inout_set: Arc<PersistentDescriptorSet>,
    palette_set: Arc<PersistentDescriptorSet>,
    dispatch_size: [u32; 3],
) -> Result<PrimaryAutoCommandBuffer> {
    let vulkan = context.vulkan();
    let queue = context.queue();

    let mut builder = AutoCommandBufferBuilder::primary(
        &vulkan.command_allocator,
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )?;

    let layout = pipeline.layout().clone();
    builder
        .bind_pipeline_compute(pipeline)
        .bind_descriptor_sets(
            PipelineBindPoint::Compute,
            layout.clone(),
            INOUT_SET,
            inout_set,
        )
        .bind_descriptor_sets(PipelineBindPoint::Compute, layout, PALETTE_SET, palette_set)
        .dispatch(dispatch_size)?;

    let commands = builder.build()?;
    vulkan.set_debug_utils_object_name(&commands, || "Render to screen".into())?;
    Ok(commands)
}

// Rendering shader used when data comes from the CPU
// TODO: Use different shaders when rendering from GPU data
mod shader {
    vulkano_shaders::shader! {
        ty: "compute",
        vulkan_version: "1.0",
        spirv_version: "1.0",
        path: "src/cpu.comp",
    }
}

/// Shader descriptor set to which input and output data are bound
const INOUT_SET: u32 = 0;

/// Descriptor within `INOUT_SET` for readout of simulation output
const DATA_INPUT: u32 = 0;

/// Descriptor within `INOUT_SET` for writing to screen
const SCREEN_OUTPUT: u32 = 1;

/// Shader descriptor set to which the color palette is bound
const PALETTE_SET: u32 = 1;

/// Descriptor within `PALETTE_SET` for reading the color palette
const PALETTE: u32 = 0;

/// Callback to configure palette sampling during pipeline construction
fn sampler_setup_callback(
    vulkan: &VulkanContext,
) -> Result<impl FnOnce(&mut [DescriptorSetLayoutCreateInfo])> {
    let palette_sampler = Sampler::new(vulkan.device.clone(), palette::sampler_conig())?;
    vulkan.set_debug_utils_object_name(&palette_sampler, || "Color palette sampler".into())?;
    Ok(
        move |descriptor_sets: &mut [DescriptorSetLayoutCreateInfo]| {
            descriptor_sets[PALETTE_SET as usize]
                .bindings
                .get_mut(&PALETTE)
                .expect("Color palette descriptor should be present")
                .immutable_samplers = vec![palette_sampler];
        },
    )
}
