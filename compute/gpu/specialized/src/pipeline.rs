//! Simulation pipeline

use crate::Result;
use compute::gpu::context::VulkanContext;
use compute_gpu_naive::pipeline as naive_pipeline;
use data::{concentration::gpu::shape::Shape, parameters::Parameters};
use std::{collections::HashMap, hash::BuildHasher, sync::Arc};
use vulkano::{
    command_buffer::{
        allocator::CommandBufferAllocator, AutoCommandBufferBuilder, PrimaryAutoCommandBuffer,
    },
    descriptor_set::PersistentDescriptorSet,
    device::physical::PhysicalDevice,
    pipeline::{
        compute::ComputePipelineCreateInfo, layout::PipelineDescriptorSetLayoutCreateInfo,
        ComputePipeline, PipelineLayout, PipelineShaderStageCreateInfo,
    },
    shader::{ShaderModule, SpecializationConstant, SpecializedShaderModule},
};

/// Device requirements for this backend
pub(crate) fn requirements(device: &PhysicalDevice, work_group_shape: Shape) -> bool {
    naive_pipeline::image_requirements(device, work_group_shape)
}

/// Create the simulation pipeline
pub(crate) fn create(
    context: &VulkanContext,
    parameters: Parameters,
    work_group_shape: Shape,
) -> Result<Arc<ComputePipeline>> {
    // Load and specialize the compute shader
    let shader = shader::load(context.device.clone())?;
    context.set_debug_utils_object_name(&shader, || "Simulation stepper shader".into())?;
    let entry_point = specialize_shader(&shader, parameters, work_group_shape)?
        .single_entry_point()
        .expect("shader should have an entry point");
    let shader_stage = PipelineShaderStageCreateInfo::new(entry_point.clone());

    // Autogenerate a pipeline layout
    let mut pipeline_layout_cfg =
        PipelineDescriptorSetLayoutCreateInfo::from_stages([&shader_stage]);
    naive_pipeline::setup_input_sampler(context, &mut pipeline_layout_cfg.set_layouts[..])?;
    let pipeline_layout_cfg =
        pipeline_layout_cfg.into_pipeline_layout_create_info(context.device.clone())?;
    let pipeline_layout = PipelineLayout::new(context.device.clone(), pipeline_layout_cfg)?;

    // Set up the compute pipeline, with specialization constants
    let pipeline = ComputePipeline::new(
        context.device.clone(),
        Some(context.pipeline_cache.clone()),
        ComputePipelineCreateInfo::stage_layout(shader_stage, pipeline_layout),
    )?;
    context.set_debug_utils_object_name(&pipeline, || "Simulation stepper".into())?;
    Ok(pipeline)
}

/// Record a simulation step
pub fn record_step<CommAlloc: CommandBufferAllocator>(
    builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer<CommAlloc>, CommAlloc>,
    pipeline: &ComputePipeline,
    images: Arc<PersistentDescriptorSet>,
    dispatch_size: [u32; 3],
) -> Result<()> {
    naive_pipeline::record_step(builder, pipeline, images, dispatch_size)
}

/// Specialize the simulation shader
fn specialize_shader(
    shader: &Arc<ShaderModule>,
    parameters: Parameters,
    work_group_shape: Shape,
) -> Result<Arc<SpecializedShaderModule>> {
    // Read out shader specialization constants and make sure GPU specialization
    // constants seem in sync with CPU specialization constants
    let mut constants = shader.specialization_constants().clone();
    assert_eq!(
        constants.len(),
        16,
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
    //
    set(&mut constants, 2, parameters.weights().0[0][0]);
    set(&mut constants, 3, parameters.weights().0[1][0]);
    set(&mut constants, 4, parameters.weights().0[2][0]);
    set(&mut constants, 5, parameters.weights().0[0][1]);
    set(&mut constants, 6, parameters.weights().0[1][1]);
    set(&mut constants, 7, parameters.weights().0[2][1]);
    set(&mut constants, 8, parameters.weights().0[0][2]);
    set(&mut constants, 9, parameters.weights().0[1][2]);
    set(&mut constants, 10, parameters.weights().0[2][2]);
    //
    set(&mut constants, 11, parameters.diffusion_rate_u);
    set(&mut constants, 12, parameters.diffusion_rate_v);
    set(&mut constants, 13, parameters.feed_rate);
    set(&mut constants, 14, parameters.kill_rate);
    set(&mut constants, 15, parameters.time_step);

    // Specialize the shader
    Ok(shader.specialize(constants)?)
}

/// Compute shader used for GPU-side simulation
mod shader {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/main.comp",
    }
}
