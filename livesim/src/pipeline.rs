//! Rendering pipeline

use crate::{input::Input, palette, Result, SimulationContext};
use compute::gpu::context::VulkanContext;
use data::concentration::gpu::shape::{self, PartialWorkGroupError, Shape};
use std::{collections::HashMap, hash::BuildHasher, sync::Arc};
use vulkano::{
    buffer::BufferUsage,
    command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer},
    descriptor_set::{layout::DescriptorSetLayoutCreateInfo, DescriptorSet, WriteDescriptorSet},
    image::{
        sampler::Sampler,
        view::{ImageView, ImageViewCreateInfo},
        Image, ImageUsage,
    },
    pipeline::{
        compute::ComputePipelineCreateInfo, layout::PipelineDescriptorSetLayoutCreateInfo,
        ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout,
        PipelineShaderStageCreateInfo,
    },
    shader::{ShaderModule, SpecializationConstant, SpecializedShaderModule},
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
    // Load and specialize the rendering shader
    let shader = shader::load(vulkan.device.clone())?;
    vulkan.set_debug_utils_object_name(&shader, || "Live renderer shader".into())?;
    let entry_point = specialize_shader(&shader, work_group_shape)?
        .single_entry_point()
        .expect("shader should have an entry point");
    let shader_stage = PipelineShaderStageCreateInfo::new(entry_point.clone());

    // Autogenerate a pipeline layout
    let mut pipeline_layout_cfg =
        PipelineDescriptorSetLayoutCreateInfo::from_stages([&shader_stage]);
    setup_palette_sampler(vulkan, &mut pipeline_layout_cfg.set_layouts[..])?;
    let pipeline_layout_cfg =
        pipeline_layout_cfg.into_pipeline_layout_create_info(vulkan.device.clone())?;
    let pipeline_layout = PipelineLayout::new(vulkan.device.clone(), pipeline_layout_cfg)?;

    // Set up the rendering pipeline
    let pipeline = ComputePipeline::new(
        vulkan.device.clone(),
        Some(vulkan.pipeline_cache.clone()),
        ComputePipelineCreateInfo::stage_layout(shader_stage, pipeline_layout),
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
    palette_image: Arc<Image>,
) -> Result<Arc<DescriptorSet>> {
    let palette_info = ImageViewCreateInfo {
        usage: palette_usage(),
        ..ImageViewCreateInfo::from_image(&palette_image)
    };
    let palette = DescriptorSet::new(
        vulkan.descriptor_set_allocator.clone(),
        pipeline.layout().set_layouts()[PALETTE_SET as usize].clone(),
        [WriteDescriptorSet::image_view(
            PALETTE,
            ImageView::new(palette_image, palette_info)?,
        )],
        [],
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
    swapchain_images: Vec<Arc<Image>>,
) -> Result<Vec<Arc<DescriptorSet>>> {
    assert_eq!(upload_buffers.len(), swapchain_images.len());
    // FIXME: Name swapchain images once vulkano allows for it
    let set_layout = &pipeline.layout().set_layouts()[INOUT_SET as usize];
    upload_buffers
        .iter()
        .zip(swapchain_images)
        .map(|(buffer, swapchain_image)| {
            let output_info = ImageViewCreateInfo {
                usage: output_usage(),
                ..ImageViewCreateInfo::from_image(&swapchain_image)
            };
            let descriptor_set = DescriptorSet::new(
                vulkan.descriptor_set_allocator.clone(),
                set_layout.clone(),
                [
                    WriteDescriptorSet::buffer(DATA_INPUT, buffer.clone()),
                    WriteDescriptorSet::image_view(
                        SCREEN_OUTPUT,
                        ImageView::new(swapchain_image, output_info)?,
                    ),
                ],
                [],
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
    inout_set: Arc<DescriptorSet>,
    palette_set: Arc<DescriptorSet>,
    dispatch_size: [u32; 3],
) -> Result<Arc<PrimaryAutoCommandBuffer>> {
    let vulkan = context.vulkan();
    let queue = context.queue();

    let mut builder = AutoCommandBufferBuilder::primary(
        vulkan.command_allocator.clone(),
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )?;

    let layout = pipeline.layout().clone();
    builder
        .bind_pipeline_compute(pipeline)?
        .bind_descriptor_sets(
            PipelineBindPoint::Compute,
            layout.clone(),
            INOUT_SET,
            inout_set,
        )?
        .bind_descriptor_sets(PipelineBindPoint::Compute, layout, PALETTE_SET, palette_set)?;
    // SAFETY: Compute pipeline was manually checked for safety
    unsafe { builder.dispatch(dispatch_size)? };

    let commands = builder.build()?;
    vulkan.set_debug_utils_object_name(&commands, || "Render to screen".into())?;
    Ok(commands)
}

/// Specialize the rendering shader
fn specialize_shader(
    shader: &Arc<ShaderModule>,
    work_group_shape: Shape,
) -> Result<Arc<SpecializedShaderModule>> {
    // Read out shader specialization constants and make sure GPU specialization
    // constants seem in sync with CPU specialization constants
    let mut constants = shader.specialization_constants().clone();
    assert_eq!(
        constants.len(),
        3,
        "number of specialization constants doesn't match"
    );
    fn set(
        constants: &mut HashMap<u32, SpecializationConstant, impl BuildHasher>,
        id: u32,
        value: impl Into<SpecializationConstant>,
    ) {
        constants
            .insert(id, value.into())
            .expect("no specialization constant at expected ID");
    }

    // Set up specialization constants
    set(&mut constants, 0, work_group_shape.width());
    set(&mut constants, 1, work_group_shape.height());
    set(&mut constants, 2, ui::AMPLITUDE_SCALE);

    // Specialize the shader
    Ok(shader.specialize(constants)?)
}

/// Callback to configure palette sampling during pipeline construction
fn setup_palette_sampler(
    vulkan: &VulkanContext,
    set_layouts: &mut [DescriptorSetLayoutCreateInfo],
) -> Result<()> {
    // Create palette sampler
    let palette_sampler = Sampler::new(vulkan.device.clone(), palette::sampler_conig())?;
    vulkan.set_debug_utils_object_name(&palette_sampler, || "Color palette sampler".into())?;

    // Configure the pipeline descriptor sets to use it
    set_layouts[PALETTE_SET as usize]
        .bindings
        .get_mut(&PALETTE)
        .expect("Color palette descriptor should be present")
        .immutable_samplers = vec![palette_sampler];
    Ok(())
}

// Rendering shader used when data comes from the CPU
// TODO: Use different shaders when rendering from GPU data
mod shader {
    vulkano_shaders::shader! {
        ty: "compute",
        vulkan_version: "1.0",
        spirv_version: "1.0",
        path: "src/from_cpu.comp",
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
