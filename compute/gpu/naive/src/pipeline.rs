//! Simulation pipeline

use crate::{parameters::GpuParameters, Result};
use compute::gpu::context::{config::requirements, VulkanContext};
use data::concentration::gpu::{
    image::ImageConcentration,
    shape::{self, Shape},
};
use std::sync::Arc;
use vulkano::{
    buffer::{BufferUsage, Subbuffer},
    command_buffer::{
        allocator::CommandBufferAllocator, AutoCommandBufferBuilder, PrimaryAutoCommandBuffer,
    },
    descriptor_set::{
        layout::{DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo},
        PersistentDescriptorSet, WriteDescriptorSet,
    },
    device::physical::PhysicalDevice,
    format::FormatFeatures,
    image::{
        view::{ImageView, ImageViewCreateInfo},
        ImageUsage, StorageImage,
    },
    pipeline::{ComputePipeline, Pipeline, PipelineBindPoint},
    sampler::{BorderColor, Sampler, SamplerAddressMode, SamplerCreateInfo},
};

/// Device requirements for this backend
pub(crate) fn requirements(device: &PhysicalDevice) -> bool {
    let properties = device.properties();
    let num_uniforms = 1;
    image_requirements(device, WORK_GROUP_SHAPE)
        && properties.max_bound_descriptor_sets >= 2
        && properties.max_descriptor_set_uniform_buffers >= num_uniforms
        && properties.max_per_stage_descriptor_uniform_buffers >= num_uniforms
        && properties.max_per_set_descriptors.unwrap_or(u32::MAX) >= num_uniforms
        && properties.max_per_stage_resources >= NUM_COMMON_RESOURCES + num_uniforms
        && properties.max_uniform_buffer_range
            >= u32::try_from(std::mem::size_of::<GpuParameters>())
                .expect("Parameters can't fit on any Vulkan device!")
}

/// Device requirements shared by all image-based pipelines
pub fn image_requirements(device: &PhysicalDevice, work_group_shape: Shape) -> bool {
    let properties = device.properties();
    let Ok(format_properties) = device.format_properties(ImageConcentration::format()) else {
        return false;
    };
    requirements::for_work_group(properties, work_group_shape)
        && properties.max_bound_descriptor_sets >= 1
        && properties.max_per_stage_descriptor_samplers >= NUM_SAMPLED_IMAGES
        && properties.max_per_stage_descriptor_sampled_images >= NUM_SAMPLED_IMAGES
        && properties.max_per_stage_descriptor_storage_images >= NUM_STORAGE_IMAGES
        && properties.max_per_set_descriptors.unwrap_or(u32::MAX)
            >= NUM_SAMPLED_IMAGES + NUM_STORAGE_IMAGES
        && properties.max_per_stage_resources >= NUM_COMMON_RESOURCES
        && properties.max_sampler_allocation_count >= NUM_SAMPLERS
        && format_properties.optimal_tiling_features.contains(
            FormatFeatures::SAMPLED_IMAGE
                | FormatFeatures::STORAGE_IMAGE
                | ImageConcentration::required_image_format_features(),
        )
}

/// Work-group shape hardcoded in this specific shader
pub(crate) const WORK_GROUP_SHAPE: Shape = Shape::from_width_height([8, 8]);

/// Create the simulation pipeline
pub(crate) fn create(context: &VulkanContext) -> Result<Arc<ComputePipeline>> {
    // Load the compute shader
    let shader = shader::load(context.device.clone())?;
    context.set_debug_utils_object_name(&shader, || "Simulation stepper shader".into())?;

    // Set up the compute pipeline
    let pipeline = ComputePipeline::new(
        context.device.clone(),
        shader.entry_point("main").expect("Should be present"),
        &(),
        Some(context.pipeline_cache.clone()),
        sampler_setup_callback(&context)?,
    )?;
    context.set_debug_utils_object_name(&pipeline, || "Simulation stepper".into())?;
    Ok(pipeline)
}

/// Callback to configure input image sampling during pipeline construction
pub fn sampler_setup_callback(
    context: &VulkanContext,
) -> Result<impl FnOnce(&mut [DescriptorSetLayoutCreateInfo])> {
    let input_sampler = Sampler::new(
        context.device.clone(),
        SamplerCreateInfo {
            address_mode: [SamplerAddressMode::ClampToBorder; 3],
            border_color: BorderColor::FloatOpaqueBlack,
            unnormalized_coordinates: true,
            ..Default::default()
        },
    )?;
    context.set_debug_utils_object_name(&input_sampler, || "Concentration sampler".into())?;
    Ok(
        move |descriptor_sets: &mut [DescriptorSetLayoutCreateInfo]| {
            fn binding(
                set: &mut DescriptorSetLayoutCreateInfo,
                idx: u32,
            ) -> &mut DescriptorSetLayoutBinding {
                set.bindings.get_mut(&idx).unwrap()
            }
            let images_set = &mut descriptor_sets[IMAGES_SET as usize];
            binding(images_set, IN_U).immutable_samplers = vec![input_sampler.clone()];
            binding(images_set, IN_V).immutable_samplers = vec![input_sampler];
        },
    )
}

/// Manner in which the parameters are used
pub(crate) fn parameters_usage() -> BufferUsage {
    BufferUsage::UNIFORM_BUFFER
}

/// Create the parameters descriptor set
pub(crate) fn new_parameters_set(
    context: &VulkanContext,
    pipeline: &ComputePipeline,
    parameters: Subbuffer<GpuParameters>,
) -> Result<Arc<PersistentDescriptorSet>> {
    let descriptor_set = PersistentDescriptorSet::new(
        &context.descriptor_set_allocator,
        pipeline.layout().set_layouts()[PARAMS_SET as usize].clone(),
        [WriteDescriptorSet::buffer(PARAMS, parameters)],
    )?;
    // FIXME: Name this descriptor set once vulkano allows for it
    Ok(descriptor_set)
}

/// Manner in which input images are used
pub fn input_usage() -> ImageUsage {
    ImageUsage::SAMPLED
}

/// Manner in which output images are used
pub fn output_usage() -> ImageUsage {
    ImageUsage::STORAGE
}

/// Bind the pipeline and its parameters
pub(crate) fn bind_pipeline(
    builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer, impl CommandBufferAllocator>,
    pipeline: Arc<ComputePipeline>,
    parameters: Arc<PersistentDescriptorSet>,
) {
    let layout = pipeline.layout().clone();
    builder
        .bind_pipeline_compute(pipeline)
        .bind_descriptor_sets(PipelineBindPoint::Compute, layout, PARAMS_SET, parameters);
}

/// Dispatch size required by an image-based pipeline, for a certain simulation
/// domain and work-group shape
pub fn dispatch_size(domain_shape: Shape, work_group_shape: Shape) -> Result<[u32; 3]> {
    Ok(shape::full_dispatch_size(domain_shape, work_group_shape)?)
}

/// Create a descriptor set for a particular (in_u, in_v, out_u, out_v) configuration
pub fn new_images_set(
    context: &VulkanContext,
    pipeline: &ComputePipeline,
    [in_u, in_v, out_u, out_v]: [Arc<StorageImage>; 4],
) -> Result<Arc<PersistentDescriptorSet>> {
    let layout = pipeline.layout().set_layouts()[usize::try_from(IMAGES_SET).unwrap()].clone();
    let binding = |binding, image: Arc<StorageImage>, usage| -> Result<WriteDescriptorSet> {
        let view_info = ImageViewCreateInfo {
            usage,
            ..ImageViewCreateInfo::from_image(&image)
        };
        Ok(WriteDescriptorSet::image_view(
            binding,
            ImageView::new(image, view_info)?,
        ))
    };
    let descriptor_set = PersistentDescriptorSet::new(
        &context.descriptor_set_allocator,
        layout,
        [
            binding(IN_U, in_u, input_usage())?,
            binding(IN_V, in_v, input_usage())?,
            binding(OUT_U, out_u, output_usage())?,
            binding(OUT_V, out_v, output_usage())?,
        ],
    )?;
    // FIXME: Name this descriptor set once vulkano allows for it
    Ok(descriptor_set)
}

/// Record a simulation step
pub fn record_step(
    builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer, impl CommandBufferAllocator>,
    pipeline: &ComputePipeline,
    images: Arc<PersistentDescriptorSet>,
    dispatch_size: [u32; 3],
) -> Result<()> {
    builder
        .bind_descriptor_sets(
            PipelineBindPoint::Compute,
            pipeline.layout().clone(),
            IMAGES_SET,
            images,
        )
        .dispatch(dispatch_size)?;
    Ok(())
}

/// Compute shader used for GPU-side simulation
mod shader {
    vulkano_shaders::shader! {
        ty: "compute",
        vulkan_version: "1.0",
        spirv_version: "1.0",
        path: "src/main.comp",
    }
}

/// Shader descriptor set to which input and output images are bound
const IMAGES_SET: u32 = 0;

/// Descriptor within `IMAGES_SET` for sampling of input U concentration
const IN_U: u32 = 0;

/// Descriptor within `IMAGES_SET` for sampling of input V concentration
const IN_V: u32 = 1;

/// Descriptor within `IMAGES_SET` for writing to output U concentration
const OUT_U: u32 = 2;

/// Descriptor within `IMAGES_SET` for writing to output V concentration
const OUT_V: u32 = 3;

/// Shader descriptor set to which simulation parameters are bound
const PARAMS_SET: u32 = 1;

/// Descriptor within `PARAMS_SET` for simulation parameters
const PARAMS: u32 = 0;

/// Number of sampled images used by image-based pipelines
const NUM_SAMPLED_IMAGES: u32 = 2;

/// Number of storage images used by image-based pipelines
const NUM_STORAGE_IMAGES: u32 = 2;

/// Number of samplers used by image-based pipelines
const NUM_SAMPLERS: u32 = 1;

/// Minimal number of resources used by image-based pipelines
const NUM_COMMON_RESOURCES: u32 = NUM_SAMPLERS + NUM_SAMPLED_IMAGES + NUM_STORAGE_IMAGES;
