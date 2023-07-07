//! Simulation pipeline

use self::shader::SpecializationConstants;
use crate::Result;
use compute::gpu::context::VulkanContext;
use compute_gpu_naive::pipeline as naive_pipeline;
use data::{
    concentration::gpu::shape::Shape,
    parameters::{Parameters, StencilWeights},
};
use std::sync::Arc;
use vulkano::{
    command_buffer::{
        allocator::CommandBufferAllocator, AutoCommandBufferBuilder, PrimaryAutoCommandBuffer,
    },
    descriptor_set::PersistentDescriptorSet,
    device::physical::PhysicalDevice,
    pipeline::ComputePipeline,
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
    // Load the compute shader
    let shader = shader::load(context.device.clone())?;
    context.set_debug_utils_object_name(&shader, || "Simulation stepper shader".into())?;

    // Set up the compute pipeline, with specialization constants
    let pipeline = ComputePipeline::new(
        context.device.clone(),
        shader.entry_point("main").expect("Should be present"),
        &specialization_constants(parameters, work_group_shape),
        Some(context.pipeline_cache.clone()),
        naive_pipeline::sampler_setup_callback(&context)?,
    )?;
    context.set_debug_utils_object_name(&pipeline, || "Simulation stepper".into())?;
    Ok(pipeline)
}

/// Record a simulation step
pub fn record_step(
    builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer, impl CommandBufferAllocator>,
    pipeline: &ComputePipeline,
    images: Arc<PersistentDescriptorSet>,
    dispatch_size: [u32; 3],
) -> Result<()> {
    naive_pipeline::record_step(builder, pipeline, images, dispatch_size)
}

/// Generate GPU specialization constants
fn specialization_constants(
    parameters: Parameters,
    work_group_shape: Shape,
) -> SpecializationConstants {
    // By using struct patterns as done here, we ensure that many possible
    // mismatches between CPU and GPU expectations can be detected.
    let Parameters {
        weights:
            StencilWeights(
                [[weight00, weight10, weight20], [weight01, weight11, weight21], [weight02, weight12, weight22]],
            ),
        diffusion_rate_u,
        diffusion_rate_v,
        feed_rate,
        kill_rate,
        time_step,
    } = parameters;
    SpecializationConstants {
        weight00,
        weight01,
        weight02,
        weight10,
        weight11,
        weight12,
        weight20,
        weight21,
        weight22,
        diffusion_rate_u,
        diffusion_rate_v,
        feed_rate,
        kill_rate,
        time_step,
        constant_0: work_group_shape.width(),
        constant_1: work_group_shape.height(),
    }
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
