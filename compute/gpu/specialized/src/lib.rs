//! Specialization-based implementation of GPU Gray-Scott simulation
//!
//! This version is quite similar to the naive version, except for the fact that
//! it uses specialization constants to pass parameters from CPU to GPU. This
//! allows the GPU compiler to optimize code specifically for the simulation
//! parameters, and also makes it a lot easier to keep CPU and GPU code in sync.

use compute::{Simulate, SimulateBase};
use compute_gpu::{VulkanConfig, VulkanContext};
use compute_gpu_naive::{Error, Result, Species, IMAGES_SET};
use data::{
    concentration::gpu::image::{ImageConcentration, ImageContext},
    parameters::{Parameters, StencilWeights},
};
use std::sync::Arc;
use vulkano::{
    command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage},
    device::Queue,
    pipeline::{ComputePipeline, Pipeline, PipelineBindPoint},
    sync::GpuFuture,
};

/// Work-group size used by the shader
// FIXME: Make configurable via CLI options
const WORK_GROUP_SIZE: [u32; 3] = [8, 8, 1];

/// Gray-Scott reaction simulation
pub struct Simulation {
    /// General-purpose Vulkan context
    context: VulkanContext,

    /// Compute pipeline
    pipeline: Arc<ComputePipeline>,
}
//
impl SimulateBase for Simulation {
    type Concentration = ImageConcentration;

    type Error = Error;

    fn new(params: Parameters) -> Result<Self> {
        // Check that ImageConcentration supports what we need
        compute_gpu_naive::check_image_concentration();

        // Set up Vulkan
        let context = VulkanConfig {
            other_device_requirements: Box::new(|device| {
                compute_gpu_naive::image_device_requirements(device, WORK_GROUP_SIZE)
            }),
            ..VulkanConfig::default()
        }
        .setup()?;

        // Load the compute shader + check shader code assumptions
        // when we can (not all data is available)
        let shader = shader::load(context.device.clone())?;

        // Set up the compute pipeline, with specialization constants for all
        // simulation parameters since they are known at GPU shader compile
        // time and won't change afterwards.
        //
        // By using struct patterns, we ensure that a large number of possible
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
        } = params;
        let pipeline = ComputePipeline::new(
            context.device.clone(),
            shader.entry_point("main").expect("Should be present"),
            &shader::SpecializationConstants {
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
                constant_0: WORK_GROUP_SIZE[0],
                constant_1: WORK_GROUP_SIZE[1],
            },
            Some(context.pipeline_cache.clone()),
            compute_gpu_naive::sampler_setup_callback(context.device.clone())?,
        )?;

        Ok(Self { context, pipeline })
    }

    fn make_species(&self, shape: [usize; 2]) -> Result<Species> {
        compute_gpu_naive::check_image_shape_requirements(
            self.context.device.physical_device(),
            shape,
        )?;
        Ok(Species::new(
            ImageContext::new(
                self.context.memory_allocator.clone(),
                self.context.command_allocator.clone(),
                self.queue().clone(),
                self.queue().clone(),
            )?,
            shape,
        )?)
    }
}
//
impl Simulate for Simulation {
    fn perform_steps(&self, species: &mut Species, steps: usize) -> Result<()> {
        // Prepare to record GPU commands
        let mut builder = AutoCommandBufferBuilder::primary(
            &self.context.command_allocator,
            self.queue().queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )?;

        // Set up the compute pipeline and parameters descriptor set
        builder.bind_pipeline_compute(self.pipeline.clone());

        // Determine the GPU dispatch size
        let dispatch_size = compute_gpu_naive::dispatch_size(species.shape(), WORK_GROUP_SIZE);

        // Record the simulation steps
        for _ in 0..steps {
            // Set up a descriptor set for the input and output images
            let images =
                compute_gpu_naive::images_descriptor_set(&self.context, &self.pipeline, species)?;

            // Attach the images and run the simulation
            builder
                .bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    self.pipeline.layout().clone(),
                    IMAGES_SET,
                    images,
                )
                .dispatch(dispatch_size)?;

            // Flip to the other set of images and start over
            species.flip()?;
        }

        // Synchronously execute the simulation steps
        let commands = builder.build()?;
        vulkano::sync::now(self.context.device.clone())
            .then_execute(self.queue().clone(), commands)?
            .then_signal_fence_and_flush()?
            .wait(None)?;
        Ok(())
    }
}
//
impl Simulation {
    /// This implementation only uses one queue, therefore we can take shortcuts
    fn queue(&self) -> &Arc<Queue> {
        &self.context.queues[0]
    }
}

mod shader {
    vulkano_shaders::shader! {
        ty: "compute",
        vulkan_version: "1.0",
        spirv_version: "1.0",
        path: "src/main.comp",
    }
}
