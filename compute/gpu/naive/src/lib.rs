//! Naive implementation of GPU-based Gray-Scott simulation
//!
//! This version uses GPU images in a straightforward way to perform Gray-Scott
//! simulation. It illustrates the challenge of keeping CPU and GPU code in sync
//! in the split-source model that Vulkan-based Rust code must sadly use until
//! rust-gpu is mature enough.

#![allow(clippy::result_large_err)]

mod parameters;
pub mod pipeline;
pub mod species;

use self::{pipeline::WORK_GROUP_SHAPE, species::Species};
use compute::{
    gpu::{
        context::{config::VulkanConfig, ContextBuildError, VulkanContext},
        SimulateGpu,
    },
    NoArgs, Simulate, SimulateBase,
};
use data::{
    concentration::gpu::{image::ImageConcentration, shape::PartialWorkGroupError},
    parameters::Parameters,
};
use std::sync::Arc;
use thiserror::Error;
use vulkano::{
    buffer::BufferError,
    command_buffer::{
        AutoCommandBufferBuilder, BuildError, CommandBufferBeginError, CommandBufferExecError,
        CommandBufferExecFuture, CommandBufferUsage, PipelineExecutionError,
    },
    descriptor_set::{DescriptorSetCreationError, PersistentDescriptorSet},
    device::Queue,
    image::view::ImageViewCreationError,
    pipeline::{compute::ComputePipelineCreationError, ComputePipeline},
    sampler::SamplerCreationError,
    shader::ShaderCreationError,
    sync::{FlushError, GpuFuture},
    OomError,
};

/// Gray-Scott reaction simulation
pub struct Simulation {
    /// General-purpose Vulkan context
    context: VulkanContext,

    /// Compute pipeline
    pipeline: Arc<ComputePipeline>,

    /// Simulation parameters descriptor set
    parameters: Arc<PersistentDescriptorSet>,
}
//
impl SimulateBase for Simulation {
    type CliArgs = NoArgs;

    type Concentration = ImageConcentration;

    type Error = Error;

    fn make_species(&self, shape: [usize; 2]) -> Result<Species> {
        species::make_species(&self.context, shape, WORK_GROUP_SHAPE, self.queue().clone())
    }
}
//
impl SimulateGpu for Simulation {
    fn with_config(params: Parameters, _args: NoArgs, mut config: VulkanConfig) -> Result<Self> {
        // Set up Vulkan
        let context = VulkanConfig {
            other_device_requirements: Box::new(move |device| {
                (config.other_device_requirements)(device) && pipeline::requirements(device)
            }),
            ..config
        }
        .build()?;

        // Set up the compute pipeline
        let pipeline = pipeline::create(&context)?;

        // Move parameters to GPU-accessible memory
        // NOTE: This memory is not the most efficient to access from GPU.
        //       See the `specialized` backend for a better way to handle this.
        let parameters = pipeline::new_parameters_set(
            &context,
            &pipeline,
            parameters::expose(&context, params)?,
        )?;

        Ok(Self {
            context,
            pipeline,
            parameters,
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
        pipeline::bind_pipeline(&mut builder, self.pipeline.clone(), self.parameters.clone());
        let dispatch_size = species::dispatch_size_for(&species, WORK_GROUP_SHAPE);

        // Record the simulation steps
        for _ in 0..steps {
            // Record a simulation step using the current input and output images
            let images = species::images_descriptor_set(&self.context, &self.pipeline, species)?;
            pipeline::record_step(&mut builder, &self.pipeline, images, dispatch_size)?;

            // Flip to another set of images and start over
            species.flip()?;
        }

        // Schedule simulation work
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

/// Errors that can occur during this computation
#[derive(Debug, Error)]
pub enum Error {
    #[error("failed to initialize the Vulkan API")]
    ContextBuild(#[from] ContextBuildError),

    #[error("failed to create the compute shader")]
    ShaderCreation(#[from] ShaderCreationError),

    #[error("failed to create the sampler")]
    SamplerCreation(#[from] SamplerCreationError),

    #[error("failed to create the compute pipeline")]
    PipelineCreation(#[from] ComputePipelineCreationError),

    #[error("failed to move parameters to GPU-accessible memory")]
    ParamsUpload(#[from] BufferError),

    #[error("failed to create a descriptor set")]
    DescriptorSetCreation(#[from] DescriptorSetCreationError),

    #[error("failed to manipulate concentration images")]
    Image(#[from] data::concentration::gpu::image::Error),

    #[error("failed to create an image view")]
    ImageViewCreation(#[from] ImageViewCreationError),

    #[error("failed to start recording a command buffer")]
    CommandBufferBegin(#[from] CommandBufferBeginError),

    #[error("failed to record a compute dispatch")]
    PipelineExecution(#[from] PipelineExecutionError),

    #[error("failed to build a command buffer")]
    CommandBufferBuild(#[from] BuildError),

    #[error("failed to submit commands to the queue")]
    CommandBufferExec(#[from] CommandBufferExecError),

    #[error("failed to flush the queue to the GPU")]
    Flush(#[from] FlushError),

    #[error("ran out of memory")]
    Oom(#[from] OomError),

    #[error("device or shader does not support this domain shape at all")]
    UnsupportedShape,

    #[error("domain shape is not supported with the current work-group shape")]
    ShapeGroupMismatch(#[from] PartialWorkGroupError),
}
//
pub type Result<T> = std::result::Result<T, Error>;
