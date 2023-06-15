//! Naive implementation of GPU-based Gray-Scott simulation
//!
//! This version follows the logic of the naive_propagation.cpp example from the
//! C++ tutorial, and is slow for the same reason.

use compute::{Simulate, SimulateBase};
use compute_gpu::{VulkanConfig, VulkanContext};
use crevice::std140::AsStd140;
use data::{
    concentration::gpu::image::{ImageConcentration, ImageContext},
    parameters::{Parameters, STENCIL_SHAPE},
    Precision,
};
use std::{
    collections::{hash_map::Entry, HashMap},
    sync::Arc,
};
use thiserror::Error;
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferError, BufferUsage},
    command_buffer::{
        AutoCommandBufferBuilder, BuildError, CommandBufferBeginError, CommandBufferExecError,
        CommandBufferUsage, PipelineExecutionError,
    },
    descriptor_set::{DescriptorSetCreationError, PersistentDescriptorSet, WriteDescriptorSet},
    device::{
        physical::{PhysicalDevice, PhysicalDeviceError},
        Queue,
    },
    format::FormatFeatures,
    image::{
        view::{ImageView, ImageViewCreateInfo, ImageViewCreationError},
        ImageUsage, StorageImage,
    },
    memory::allocator::{AllocationCreateInfo, MemoryUsage},
    pipeline::{
        compute::ComputePipelineCreationError, ComputePipeline, Pipeline, PipelineBindPoint,
    },
    sampler::{BorderColor, Sampler, SamplerAddressMode, SamplerCreateInfo, SamplerCreationError},
    shader::ShaderCreationError,
    sync::{FlushError, GpuFuture},
};

/// Chosen concentration type
type Species = data::concentration::Species<ImageConcentration>;

/// Shader descriptor set to which input and output images are bound
const IMAGES_SET: u32 = 0;

/// Shader descriptor set to which simulation parameters are bound
const PARAMS_SET: u32 = 1;

/// Work-group size used by the shader
const WORK_GROUP_SIZE: [u32; 3] = [8, 8, 1];

/// Gray-Scott reaction simulation
pub struct Simulation {
    /// General-purpose Vulkan context
    context: VulkanContext,

    /// Compute pipeline
    pipeline: Arc<ComputePipeline>,

    /// Input image samplers
    input_samplers: [Arc<Sampler>; 2],

    /// Simulation parameters descriptor set
    params: Arc<PersistentDescriptorSet>,
}
//
impl SimulateBase for Simulation {
    type Concentration = ImageConcentration;

    type Error = Error;

    fn new(params: Parameters) -> Result<Self> {
        // Check that ImageConcentration supports what we need
        assert!(
            ImageConcentration::image_format_info()
                .usage
                .contains(ImageUsage::SAMPLED | ImageUsage::STORAGE),
            "ImageConcentration does not enable all required usage flags"
        );

        // Set up Vulkan
        let context = VulkanConfig {
            other_device_requirements: Box::new(Self::minimal_device_requirements),
            ..VulkanConfig::default()
        }
        .setup()?;

        // Load the compute shader + check shader code assumptions
        assert_eq!(
            std::mem::size_of::<Precision>(),
            4,
            "Must adjust shader.glsl to use requested non-float precision"
        );
        assert_eq!(
            STENCIL_SHAPE,
            [3, 3],
            "Must adjust shader.glsl to account for stencil shape change"
        );
        let shader = shader::load(context.device.clone())?;

        // Set up the compute pipeline
        let pipeline = ComputePipeline::new(
            context.device.clone(),
            shader.entry_point("main").expect("Should be present"),
            &(),
            Some(context.pipeline_cache.clone()),
            |_| (),
        )?;

        // Set up input image sampling
        let input_sampler = || {
            Sampler::new(
                context.device.clone(),
                SamplerCreateInfo {
                    address_mode: [SamplerAddressMode::ClampToEdge; 3],
                    border_color: BorderColor::FloatOpaqueBlack,
                    unnormalized_coordinates: true,
                    ..Default::default()
                },
            )
        };
        let input_samplers = [input_sampler()?, input_sampler()?];

        // Move parameters to GPU-accessible memory
        let params = Buffer::from_data(
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
        let params_layout = pipeline.layout().set_layouts()[PARAMS_SET as usize].clone();
        let params = PersistentDescriptorSet::new(
            &context.descriptor_allocator,
            params_layout,
            [WriteDescriptorSet::buffer(0, params)],
        )?;

        Ok(Self {
            context,
            pipeline,
            input_samplers,
            params,
        })
    }

    fn make_species(&self, shape: [usize; 2]) -> Result<Species> {
        self.check_shape_requirements(shape)?;
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
        builder
            .bind_pipeline_compute(self.pipeline.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.pipeline.layout().clone(),
                PARAMS_SET,
                self.params.clone(),
            );

        // Determine the GPU dispatch size
        let global_size = Self::shape_to_global_size(species.shape());
        let dispatch_size = std::array::from_fn(|i| {
            let shape = global_size[i];
            let work_group_size = usize::try_from(WORK_GROUP_SIZE[i]).unwrap();
            debug_assert_eq!(shape % work_group_size, 0, "Checked by make_species");
            u32::try_from(shape / work_group_size).unwrap()
        });

        // Prepare to cache the image descriptor sets
        let mut images = HashMap::<[Arc<StorageImage>; 4], Arc<PersistentDescriptorSet>>::new();

        // Run the simulation steps
        for _ in 0..steps {
            // Acquire access to the input and output images
            let (in_u, out_u) = species.u.in_out();
            let (in_v, out_v) = species.v.in_out();
            let in_u = in_u.access_image();
            let in_v = in_v.access_image();
            let out_u = out_u.access_image();
            let out_v = out_v.access_image();

            // Have we seen this input + outpt images configuration before?
            let images =
                match images.entry([in_u.clone(), in_v.clone(), out_u.clone(), out_v.clone()]) {
                    // If so, reuse previously configured descriptor set
                    Entry::Occupied(occupied) => occupied.get().clone(),

                    // Otherwise, make a new descriptor set
                    Entry::Vacant(vacant) => {
                        let view = |image: &Arc<StorageImage>, usage| {
                            ImageView::new(
                                image.clone(),
                                ImageViewCreateInfo {
                                    usage,
                                    ..ImageViewCreateInfo::from_image(image)
                                },
                            )
                        };
                        let input_binding = |binding,
                                             image,
                                             sampler: &Arc<Sampler>|
                         -> Result<WriteDescriptorSet> {
                            Ok(WriteDescriptorSet::image_view_sampler(
                                binding,
                                view(image, ImageUsage::SAMPLED)?,
                                sampler.clone(),
                            ))
                        };
                        let output_binding = |binding, image| -> Result<WriteDescriptorSet> {
                            Ok(WriteDescriptorSet::image_view(
                                binding,
                                view(image, ImageUsage::STORAGE)?,
                            ))
                        };
                        let layout = self.pipeline.layout().set_layouts()
                            [usize::try_from(IMAGES_SET).unwrap()]
                        .clone();
                        let set = PersistentDescriptorSet::new(
                            &self.context.descriptor_allocator,
                            layout,
                            [
                                input_binding(0, in_u, &self.input_samplers[0])?,
                                input_binding(1, in_v, &self.input_samplers[1])?,
                                output_binding(2, out_u)?,
                                output_binding(3, out_v)?,
                            ],
                        )?;
                        vacant.insert(set).clone()
                    }
                };

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

    /// Convert a Concentration shape into a global device work size
    fn shape_to_global_size([rows, cols]: [usize; 2]) -> [usize; 3] {
        [cols, rows, 1]
    }

    /// Check minimal device requirements
    fn minimal_device_requirements(device: &PhysicalDevice) -> bool {
        let properties = device.properties();
        let Ok(format_properties) = device.format_properties(ImageConcentration::format()) else {
            return false
        };

        let num_samplers = 2;
        let num_storage_images = 2;
        let num_uniforms = 1;

        properties.max_bound_descriptor_sets >= 2
            && properties.max_compute_work_group_invocations
                >= WORK_GROUP_SIZE.into_iter().product::<u32>()
            && properties
                .max_compute_work_group_size
                .into_iter()
                .zip(WORK_GROUP_SIZE.into_iter())
                .all(|(max, req)| max >= req)
            && properties.max_descriptor_set_samplers >= num_samplers
            && properties.max_per_stage_descriptor_samplers >= num_samplers
            && properties.max_descriptor_set_storage_images >= num_storage_images
            && properties.max_per_stage_descriptor_storage_images >= num_storage_images
            && properties.max_descriptor_set_uniform_buffers >= num_uniforms
            && properties.max_per_stage_descriptor_uniform_buffers >= num_uniforms
            && properties.max_per_set_descriptors.unwrap_or(u32::MAX)
                >= num_samplers + num_storage_images
            && properties.max_per_stage_resources
                >= 2 * num_samplers + num_storage_images + num_uniforms
            && properties.max_sampler_allocation_count >= num_samplers
            && properties.max_uniform_buffer_range
                >= u32::try_from(std::mem::size_of::<Parameters>())
                    .expect("Parameters can't fit on any Vulkan device!")
            && format_properties.optimal_tiling_features.contains(
                FormatFeatures::SAMPLED_IMAGE
                    | FormatFeatures::STORAGE_IMAGE
                    | ImageConcentration::required_image_format_features(),
            )
    }

    /// Check requirements on the problem size
    fn check_shape_requirements(&self, shape: [usize; 2]) -> Result<()> {
        // The simple shader can't accomodate a problem size that is not a
        // multiple of the work group size
        let global_size = Self::shape_to_global_size(shape);
        if global_size
            .into_iter()
            .zip(WORK_GROUP_SIZE.into_iter())
            .any(|(sh, wg)| sh % wg as usize != 0)
        {
            return Err(Error::UnsupportedShape);
        }
        let dispatch_size: [usize; 3] =
            std::array::from_fn(|i| global_size[i] / WORK_GROUP_SIZE[i] as usize);

        // Check device properties
        let num_texels = shape.into_iter().product::<usize>();
        let image_size = num_texels * std::mem::size_of::<Precision>();
        let device = self.context.device.physical_device();
        let properties = device.properties();
        if properties
            .max_compute_work_group_count
            .into_iter()
            .zip(dispatch_size.into_iter())
            .any(|(max, req)| (max as usize) < req)
            || (properties.max_image_dimension2_d as usize) < shape.into_iter().max().unwrap()
            || properties.max_memory_allocation_size.unwrap_or(u64::MAX) < image_size as u64
            || properties.max_buffer_size.unwrap_or(u64::MAX) < image_size as u64
        {
            return Err(Error::UnsupportedShape);
        }

        // Check image format properties
        let Some(image_format_properties) =
            device.image_format_properties(ImageConcentration::image_format_info())? else {
                return Err(Error::UnsupportedShape);
            };
        if image_format_properties
            .max_extent
            .into_iter()
            .zip(global_size.into_iter())
            .any(|(max, req)| (max as usize) < req)
            || image_format_properties.max_resource_size < image_size as u64
        {
            return Err(Error::UnsupportedShape);
        }
        Ok(())
    }
}

mod shader {
    vulkano_shaders::shader! {
        ty: "compute",
        vulkan_version: "1.0",
        spirv_version: "1.0",
        path: "src/shader.comp",
    }
}

/// Errors that can occur during this computation
#[derive(Debug, Error)]
pub enum Error {
    /// Errors while initializing the Vulkan API
    #[error("failed to initialize the Vulkan API")]
    Init(#[from] compute_gpu::Error),

    /// Errors while manipulating species concentration images
    #[error("failed to manipulate images")]
    Image(#[from] data::concentration::gpu::image::Error),

    /// Errors while GPU-fying parameters
    #[error("failed to move parameters to GPU-accessible memory")]
    ParamsUpload(#[from] BufferError),

    /// Errors while creating the GPU shader
    #[error("failed to create compute shader")]
    ShaderCreation(#[from] ShaderCreationError),

    /// Errors while creating the compute pipeline
    #[error("failed to create compute pipeline")]
    PipelineCreation(#[from] ComputePipelineCreationError),

    /// Errors while creating samplers
    #[error("failed to create sampler")]
    SamplerCreation(#[from] SamplerCreationError),

    /// Errors while creating image views
    #[error("failed to create image view")]
    ImageViewCreation(#[from] ImageViewCreationError),

    /// Errors while creating descriptor sets
    #[error("failed to create descriptor set")]
    DescriptorSetCreation(#[from] DescriptorSetCreationError),

    /// Errors while starting to record a command buffer
    #[error("failed to start recording a command buffer")]
    CommandBufferBegin(#[from] CommandBufferBeginError),

    /// Errors while recording a compute dispatch command
    #[error("failed to record a compute dispatch")]
    PipelineExecution(#[from] PipelineExecutionError),

    /// Errors while building a command buffer
    #[error("failed to build a command buffer")]
    CommandBufferBuild(#[from] BuildError),

    /// Errors while submitting commands to a queue
    #[error("failed to submit commands to the queue")]
    CommandBufferExec(#[from] CommandBufferExecError),

    /// Errors while flushing commands to the GPU
    #[error("failed to flush the queue to the GPU")]
    Flush(#[from] FlushError),

    /// Shader does not support the requested grid shape
    #[error("device or shader does not support this grid shape")]
    UnsupportedShape,

    /// Failed to interrogate the physical device
    #[error("failed to query physical device")]
    PhysicalDevice(#[from] PhysicalDeviceError),
}
//
type Result<T> = std::result::Result<T, Error>;
