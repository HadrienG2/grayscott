//! Naive implementation of GPU-based Gray-Scott simulation
//!
//! This version uses GPU images in a straightforward way to perform Gray-Scott
//! simulation. It illustrates the challenge of keeping CPU and GPU code in sync
//! in the split-source model that Vulkan-based Rust code must sadly use until
//! rust-gpu is mature enough.

#![allow(clippy::result_large_err)]

pub mod images;
pub mod requirements;

use self::images::IMAGES_SET;
use compute::{
    gpu::{config::VulkanConfig, SimulateGpu, VulkanContext},
    NoArgs, Simulate, SimulateBase,
};
use crevice::std140::AsStd140;
use data::{
    concentration::gpu::image::{context::ImageContext, ImageConcentration},
    parameters::Parameters,
};
use std::sync::Arc;
use thiserror::Error;
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferError, BufferUsage},
    command_buffer::{
        AutoCommandBufferBuilder, BuildError, CommandBufferBeginError, CommandBufferExecError,
        CommandBufferExecFuture, CommandBufferUsage, PipelineExecutionError,
    },
    descriptor_set::{DescriptorSetCreationError, PersistentDescriptorSet, WriteDescriptorSet},
    device::{
        physical::{PhysicalDevice, PhysicalDeviceError},
        Queue,
    },
    image::view::ImageViewCreationError,
    memory::allocator::{AllocationCreateInfo, MemoryUsage},
    pipeline::{
        compute::ComputePipelineCreationError, ComputePipeline, Pipeline, PipelineBindPoint,
    },
    sampler::SamplerCreationError,
    shader::ShaderCreationError,
    sync::{FlushError, GpuFuture},
};

/// Chosen concentration type
pub type Species = data::concentration::Species<ImageConcentration>;

/// Shader descriptor set to which simulation parameters are bound
const PARAMS_SET: u32 = 1;

/// Work-group size used by the shader (must be keep in sync with shader!)
const WORK_GROUP_SIZE: [u32; 3] = [8, 8, 1];

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
        requirements::check_image_shape(
            self.context.device.physical_device(),
            shape,
            WORK_GROUP_SIZE,
        )?;
        Ok(Species::new(
            ImageContext::new(
                self.context.memory_allocator.clone(),
                self.context.command_allocator.clone(),
                self.queue().clone(),
                self.queue().clone(),
                [],
                requirements::image_usage(),
            )?,
            shape,
        )?)
    }
}
//
impl SimulateGpu for Simulation {
    fn with_config(params: Parameters, _args: NoArgs, mut config: VulkanConfig) -> Result<Self> {
        // Set up Vulkan
        let context = VulkanConfig {
            other_device_requirements: Box::new(move |device| {
                (config.other_device_requirements)(device)
                    && Self::minimal_device_requirements(device)
            }),
            ..config
        }
        .setup()?;

        // Load the compute shader
        let shader = shader::load(context.device.clone())?;
        context.set_debug_utils_object_name(&shader, || "Simulation stepper shader".into())?;

        // Set up the compute pipeline
        let pipeline = ComputePipeline::new(
            context.device.clone(),
            shader.entry_point("main").expect("Should be present"),
            &(),
            Some(context.pipeline_cache.clone()),
            images::sampler_setup_callback(&context)?,
        )?;
        context.set_debug_utils_object_name(&pipeline, || "Simulation stepper".into())?;

        // Move parameters to GPU-accessible memory
        // NOTE: This memory is not the most efficient to access from GPU.
        //       See the `specialized` backend for a good way to address this.
        let parameters = Buffer::from_data(
            &context.memory_allocator,
            BufferCreateInfo {
                usage: BufferUsage::UNIFORM_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                usage: MemoryUsage::Upload,
                ..Default::default()
            },
            params.as_std140(),
        )?;
        context
            .set_debug_utils_object_name(parameters.buffer(), || "Simulation parameters".into())?;
        let parameters = PersistentDescriptorSet::new(
            &context.descriptor_set_allocator,
            pipeline.layout().set_layouts()[PARAMS_SET as usize].clone(),
            [WriteDescriptorSet::buffer(0, parameters)],
        )?;
        // FIXME: Name this descriptor set once vulkano allows for it

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

        // Set up the compute pipeline and parameters descriptor set
        builder
            .bind_pipeline_compute(self.pipeline.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.pipeline.layout().clone(),
                PARAMS_SET,
                self.parameters.clone(),
            );

        // Determine the GPU dispatch size
        let dispatch_size = dispatch_size(species.shape(), WORK_GROUP_SIZE);

        // Record the simulation steps
        for _ in 0..steps {
            // Set up a descriptor set for the input and output images
            let images = images::descriptor_set(&self.context, &self.pipeline, species)?;

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

    /// Check minimal device requirements for this backend
    fn minimal_device_requirements(device: &PhysicalDevice) -> bool {
        let properties = device.properties();

        let num_samplers = 2;
        let num_storage_images = 2;
        let num_uniforms = 1;

        requirements::device_filter(device, WORK_GROUP_SIZE)
            && properties.max_bound_descriptor_sets >= 2
            && properties.max_descriptor_set_uniform_buffers >= num_uniforms
            && properties.max_per_stage_descriptor_uniform_buffers >= num_uniforms
            && properties.max_per_set_descriptors.unwrap_or(u32::MAX) >= 1
            && properties.max_per_stage_resources
                >= 2 * num_samplers + num_storage_images + num_uniforms
            && properties.max_uniform_buffer_range
                >= u32::try_from(std::mem::size_of::<Parameters>())
                    .expect("Parameters can't fit on any Vulkan device!")
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

/// Convert an ImageConcentration shape into a global workload size
pub fn shape_to_global_size([rows, cols]: [usize; 2]) -> [usize; 3] {
    [cols, rows, 1]
}

/// Convert an ImageConcentration shape into a compute dispatch size
pub fn dispatch_size(shape: [usize; 2], work_group_size: [u32; 3]) -> [u32; 3] {
    let global_size = shape_to_global_size(shape);
    std::array::from_fn(|i| {
        let shape = global_size[i];
        let work_group_size = usize::try_from(work_group_size[i]).unwrap();
        debug_assert_eq!(shape % work_group_size, 0, "Checked by make_species");
        u32::try_from(shape / work_group_size).unwrap()
    })
}

/// Errors that can occur during this computation
#[derive(Debug, Error)]
pub enum Error {
    #[error("failed to initialize the Vulkan API")]
    Init(#[from] compute::gpu::Error),

    #[error("failed to manipulate images")]
    Image(#[from] data::concentration::gpu::image::Error),

    #[error("failed to move parameters to GPU-accessible memory")]
    ParamsUpload(#[from] BufferError),

    #[error("failed to create compute shader")]
    ShaderCreation(#[from] ShaderCreationError),

    #[error("failed to create compute pipeline")]
    PipelineCreation(#[from] ComputePipelineCreationError),

    #[error("failed to create sampler")]
    SamplerCreation(#[from] SamplerCreationError),

    #[error("failed to create image view")]
    ImageViewCreation(#[from] ImageViewCreationError),

    #[error("failed to create descriptor set")]
    DescriptorSetCreation(#[from] DescriptorSetCreationError),

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

    #[error("device or shader does not support this grid shape")]
    UnsupportedShape,

    #[error("failed to query physical device")]
    PhysicalDevice(#[from] PhysicalDeviceError),
}
//
pub type Result<T> = std::result::Result<T, Error>;
