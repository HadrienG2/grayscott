//! Naive implementation of GPU-based Gray-Scott simulation
//!
//! This version uses GPU images in a straightforward way to perform Gray-Scott
//! simulation. It illustrates the challenge of keeping CPU and GPU code in sync
//! in the split-source model that Vulkan-based Rust code must sadly use until
//! rust-gpu is mature enough.

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
    buffer::AllocateBufferError,
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferExecError, CommandBufferExecFuture,
        CommandBufferUsage,
    },
    descriptor_set::DescriptorSet,
    device::Queue,
    pipeline::{layout::IntoPipelineLayoutCreateInfoError, ComputePipeline},
    sync::GpuFuture,
    Validated, ValidationError, VulkanError,
};

/// Gray-Scott reaction simulation
pub struct Simulation {
    /// General-purpose Vulkan context
    context: VulkanContext,

    /// Compute pipeline
    pipeline: Arc<ComputePipeline>,

    /// Simulation parameters descriptor set
    parameters: Arc<DescriptorSet>,
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
            self.context.command_allocator.clone(),
            self.queue().queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )?;

        // Prepare to dispatch compute operations
        pipeline::bind_pipeline(&mut builder, self.pipeline.clone(), self.parameters.clone())?;
        let dispatch_size = species::dispatch_size_for(species, WORK_GROUP_SHAPE);

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
    #[error("failed to initialize the Vulkan API ({0})")]
    ContextBuild(#[from] ContextBuildError),

    #[error("failed to determine compute pipeine layout")]
    ComputePipelineLayout(#[from] IntoPipelineLayoutCreateInfoError),

    #[error("failed to allocate a buffer ({0})")]
    AllocateBuffer(#[from] Validated<AllocateBufferError>),

    #[error("failed to manipulate concentration images ({0})")]
    Image(#[from] data::concentration::gpu::image::Error),

    #[error("failed to submit commands to the queue ({0})")]
    CommandBufferExec(#[from] CommandBufferExecError),

    #[error("a Vulkan API call errored out or failed validation ({0})")]
    Vulkan(#[from] Validated<VulkanError>),

    #[error("device or shader does not support this domain shape at all")]
    UnsupportedShape,

    #[error("domain shape is not supported with the current work-group shape ({0})")]
    ShapeGroupMismatch(#[from] PartialWorkGroupError),
}
//
impl From<VulkanError> for Error {
    fn from(value: VulkanError) -> Self {
        Self::Vulkan(Validated::Error(value))
    }
}
//
impl From<Box<ValidationError>> for Error {
    fn from(value: Box<ValidationError>) -> Self {
        Self::Vulkan(value.into())
    }
}
//
pub type Result<T> = std::result::Result<T, Error>;
