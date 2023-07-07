//! Specialization-based implementation of GPU Gray-Scott simulation
//!
//! This version is quite similar to the naive version, except for the fact that
//! it uses specialization constants to pass parameters from CPU to GPU. This
//! allows external control on the shader's work-group size, allows the GPU
//! compiler to optimize code specifically for the simulation parameters, and
//! makes it a lot easier to keep CPU and GPU code in sync.

mod args;
mod pipeline;

use self::args::GpuSpecializedArgs;
use compute::{
    gpu::{
        context::{config::VulkanConfig, VulkanContext},
        SimulateGpu,
    },
    Simulate, SimulateBase,
};
use compute_gpu_naive::{
    species::{self, Species},
    Error, Result,
};
use data::{
    concentration::gpu::{image::ImageConcentration, shape::Shape},
    parameters::Parameters,
};
use std::sync::Arc;
use vulkano::{
    command_buffer::{AutoCommandBufferBuilder, CommandBufferExecFuture, CommandBufferUsage},
    device::Queue,
    pipeline::ComputePipeline,
    sync::GpuFuture,
};

/// Gray-Scott reaction simulation
pub struct Simulation {
    /// General-purpose Vulkan context
    context: VulkanContext,

    /// Compute pipeline
    pipeline: Arc<ComputePipeline>,

    /// Work-group size
    work_group_shape: Shape,
}
//
impl SimulateBase for Simulation {
    type CliArgs = GpuSpecializedArgs;

    type Concentration = ImageConcentration;

    type Error = Error;

    fn make_species(&self, shape: [usize; 2]) -> Result<Species> {
        species::make_species(
            &self.context,
            shape,
            self.work_group_shape,
            self.queue().clone(),
        )
    }
}
//
impl SimulateGpu for Simulation {
    fn with_config(
        parameters: Parameters,
        args: GpuSpecializedArgs,
        mut config: VulkanConfig,
    ) -> Result<Self> {
        // Pick work-group shape
        let work_group_shape = Shape::new([
            args.compute_work_group_rows.into(),
            args.compute_work_group_cols.into(),
        ]);

        // Set up Vulkan
        let context = VulkanConfig {
            other_device_requirements: Box::new(move |device| {
                (config.other_device_requirements)(device)
                    && pipeline::requirements(device, work_group_shape)
            }),
            ..config
        }
        .build()?;

        // Set up compute pipeline
        let pipeline = pipeline::create(&context, parameters, work_group_shape)?;

        Ok(Self {
            context,
            pipeline,
            work_group_shape,
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

        // Prepare to dispatch compute operations
        builder.bind_pipeline_compute(self.pipeline.clone());
        let dispatch_size = species::dispatch_size_for(&species, self.work_group_shape);

        // Record the simulation steps
        for _ in 0..steps {
            // Record a simulation step using the current input and output images
            let images = species::images_descriptor_set(&self.context, &self.pipeline, species)?;
            pipeline::record_step(&mut builder, &self.pipeline, images, dispatch_size)?;

            // Flip to the other set of images and start over
            species.flip()?;
        }

        // Enqueue the simulation steps
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
