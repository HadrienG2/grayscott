//! Specialization-based implementation of GPU Gray-Scott simulation
//!
//! This version is quite similar to the naive version, except for the fact that
//! it uses specialization constants to pass parameters from CPU to GPU. This
//! allows the GPU compiler to optimize code specifically for the simulation
//! parameters, and also makes it a lot easier to keep CPU and GPU code in sync.

use clap::Args;
use compute::{
    gpu::{SimulateGpu, VulkanConfig, VulkanContext},
    Simulate, SimulateBase,
};
use compute_gpu_naive::{Error, Result, Species, IMAGES_SET};
use data::{
    concentration::gpu::image::{ImageConcentration, ImageContext},
    parameters::{Parameters, StencilWeights},
};
use std::{num::NonZeroU32, sync::Arc};
use vulkano::{
    command_buffer::{AutoCommandBufferBuilder, CommandBufferExecFuture, CommandBufferUsage},
    device::Queue,
    image::ImageUsage,
    pipeline::{ComputePipeline, Pipeline, PipelineBindPoint},
    sync::GpuFuture,
};

/// Kernel work-group size is tunable via CLI args and environment variables
#[derive(Args, Copy, Clone, Debug, Eq, Hash, PartialEq)]
pub struct CliArgs {
    /// Number of rows to be processed by each GPU work group
    #[arg(long, env, default_value_t = NonZeroU32::new(8).unwrap())]
    work_group_rows: NonZeroU32,

    /// Number of columns to be processed by each GPU work group
    #[arg(long, env, default_value_t = NonZeroU32::new(8).unwrap())]
    work_group_cols: NonZeroU32,
}

/// Gray-Scott reaction simulation
pub struct Simulation {
    /// General-purpose Vulkan context
    context: VulkanContext,

    /// Compute pipeline
    pipeline: Arc<ComputePipeline>,

    /// Work-group size
    work_group_size: [u32; 3],
}
//
impl SimulateBase for Simulation {
    type CliArgs = CliArgs;

    type Concentration = ImageConcentration;

    type Error = Error;

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
                [],
                ImageUsage::SAMPLED | ImageUsage::STORAGE,
            )?,
            shape,
        )?)
    }
}
//
impl SimulateGpu for Simulation {
    fn with_config(params: Parameters, args: CliArgs, mut config: VulkanConfig) -> Result<Self> {
        // Pick work-group size
        let work_group_size = [args.work_group_cols.into(), args.work_group_rows.into(), 1];

        // Set up Vulkan
        let context = VulkanConfig {
            other_device_requirements: Box::new(move |device| {
                (config.other_device_requirements)(device)
                    && compute_gpu_naive::image_device_requirements(device, work_group_size)
            }),
            ..config
        }
        .setup()?;

        // Load the compute shader
        let shader = shader::load(context.device.clone())?;
        context.set_debug_utils_object_name(&shader, || "Simulation stepper shader".into())?;

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
                constant_0: work_group_size[0],
                constant_1: work_group_size[1],
            },
            Some(context.pipeline_cache.clone()),
            compute_gpu_naive::sampler_setup_callback(&context)?,
        )?;
        context.set_debug_utils_object_name(&pipeline, || "Simulation stepper".into())?;

        Ok(Self {
            context,
            pipeline,
            work_group_size,
        })
    }

    fn context(&self) -> &VulkanContext {
        &self.context
    }

    type PrepareStepsFuture<After: GpuFuture + 'static> = CommandBufferExecFuture<After>;

    fn prepare_steps<After: GpuFuture>(
        &self,
        after: After,
        species: &mut Species,
        steps: usize,
    ) -> Result<CommandBufferExecFuture<After>> {
        // Prepare to record GPU commands
        let mut builder = AutoCommandBufferBuilder::primary(
            &self.context.command_allocator,
            self.queue().queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )?;

        // Set up the compute pipeline and parameters descriptor set
        builder.bind_pipeline_compute(self.pipeline.clone());

        // Determine the GPU dispatch size
        let dispatch_size = compute_gpu_naive::dispatch_size(species.shape(), self.work_group_size);

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
        self.context
            .set_debug_utils_object_name(&commands, || "Compute simulation steps".into())?;
        Ok(after.then_execute(self.queue().clone(), commands)?)
    }
}
//
impl Simulate for Simulation {
    fn perform_steps(&self, species: &mut Species, steps: usize) -> Result<()> {
        self.perform_steps_impl(species, steps)
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
