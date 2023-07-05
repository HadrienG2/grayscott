//! Specialization-based implementation of GPU Gray-Scott simulation
//!
//! This version is quite similar to the naive version, except for the fact that
//! it uses specialization constants to pass parameters from CPU to GPU. This
//! allows the GPU compiler to optimize code specifically for the simulation
//! parameters, and also makes it a lot easier to keep CPU and GPU code in sync.

mod args;
mod specialization;

use self::args::GpuSpecializedArgs;
use compute::{
    gpu::{config::VulkanConfig, SimulateGpu, VulkanContext},
    Simulate, SimulateBase,
};
use compute_gpu_naive::{images::IMAGES_SET, Error, Result, Species};
use data::{
    concentration::gpu::image::{context::ImageContext, ImageConcentration},
    parameters::Parameters,
};
use std::sync::Arc;
use vulkano::{
    command_buffer::{AutoCommandBufferBuilder, CommandBufferExecFuture, CommandBufferUsage},
    device::Queue,
    pipeline::{ComputePipeline, Pipeline, PipelineBindPoint},
    sync::GpuFuture,
};

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
    type CliArgs = GpuSpecializedArgs;

    type Concentration = ImageConcentration;

    type Error = Error;

    fn make_species(&self, shape: [usize; 2]) -> Result<Species> {
        compute_gpu_naive::requirements::check_image_shape(
            self.context.device.physical_device(),
            shape,
            self.work_group_size,
        )?;
        Ok(Species::new(
            ImageContext::new(
                self.context.memory_allocator.clone(),
                self.context.command_allocator.clone(),
                self.queue().clone(),
                self.queue().clone(),
                [],
                compute_gpu_naive::requirements::image_usage(),
            )?,
            shape,
        )?)
    }
}
//
impl SimulateGpu for Simulation {
    fn with_config(
        parameters: Parameters,
        args: GpuSpecializedArgs,
        mut config: VulkanConfig,
    ) -> Result<Self> {
        // Pick work-group size
        let work_group_size = [
            args.compute_work_group_cols.into(),
            args.compute_work_group_rows.into(),
            1,
        ];

        // Set up Vulkan
        let context = VulkanConfig {
            other_device_requirements: Box::new(move |device| {
                (config.other_device_requirements)(device)
                    && compute_gpu_naive::requirements::device_filter(device, work_group_size)
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
        let pipeline = ComputePipeline::new(
            context.device.clone(),
            shader.entry_point("main").expect("Should be present"),
            &specialization::constants(parameters, work_group_size),
            Some(context.pipeline_cache.clone()),
            compute_gpu_naive::images::sampler_setup_callback(&context)?,
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
                compute_gpu_naive::images::descriptor_set(&self.context, &self.pipeline, species)?;

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

/// Compute shader used for GPU-side simulation
mod shader {
    vulkano_shaders::shader! {
        ty: "compute",
        vulkan_version: "1.0",
        spirv_version: "1.0",
        path: "src/main.comp",
    }
}
