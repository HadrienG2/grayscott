//! Naive implementation of GPU-based Gray-Scott simulation
//!
//! This version uses GPU images in a straightforward way to perform Gray-Scott
//! simulation. It illustrates the challenge of keeping CPU and GPU code in sync
//! in the split-source model that Vulkan-based Rust code must sadly use until
//! rust-gpu is mature enough.

use compute::{
    gpu::{SimulateGpu, VulkanConfig, VulkanContext},
    NoArgs, Simulate, SimulateBase,
};
use crevice::std140::AsStd140;
use data::{
    concentration::gpu::image::{ImageConcentration, ImageContext},
    parameters::{Parameters, STENCIL_SHAPE},
    Precision,
};
use std::{collections::hash_map::Entry, sync::Arc};
use thiserror::Error;
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferError, BufferUsage},
    command_buffer::{
        AutoCommandBufferBuilder, BuildError, CommandBufferBeginError, CommandBufferExecError,
        CommandBufferExecFuture, CommandBufferUsage, PipelineExecutionError,
    },
    descriptor_set::{
        layout::{DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo},
        DescriptorSetCreationError, PersistentDescriptorSet, WriteDescriptorSet,
    },
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
pub type Species = data::concentration::Species<ImageConcentration>;

/// Shader descriptor set to which input and output images are bound
pub const IMAGES_SET: u32 = 0;

/// Descriptor within `IMAGES_SET` for sampling of input U concentration
const IN_U: u32 = 0;

/// Descriptor within `IMAGES_SET` for sampling of input V concentration
const IN_V: u32 = 1;

/// Descriptor within `IMAGES_SET` for writing to output U concentration
const OUT_U: u32 = 2;

/// Descriptor within `IMAGES_SET` for writing to output V concentration
const OUT_V: u32 = 3;

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
    params: Arc<PersistentDescriptorSet>,
}
//
impl SimulateBase for Simulation {
    type CliArgs = NoArgs;

    type Concentration = ImageConcentration;

    type Error = Error;

    fn make_species(&self, shape: [usize; 2]) -> Result<Species> {
        check_image_shape_requirements(self.context.device.physical_device(), shape)?;
        Ok(Species::new(
            ImageContext::new(
                self.context.memory_allocator.clone(),
                self.context.command_allocator.clone(),
                self.queue().clone(),
                self.queue().clone(),
                [],
                image_usage(),
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

        // Load the compute shader + check shader code assumptions
        // when we can (not all data is available)
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
            sampler_setup_callback(&context)?,
        )?;
        context.set_debug_utils_object_name(&pipeline, || "Simulation stepper".into())?;

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
        context.set_debug_utils_object_name(params.buffer(), || "Simulation parameters".into())?;
        let params_layout = pipeline.layout().set_layouts()[PARAMS_SET as usize].clone();
        let params = PersistentDescriptorSet::new(
            &context.descriptor_allocator,
            params_layout,
            [WriteDescriptorSet::buffer(0, params)],
        )?;

        Ok(Self {
            context,
            pipeline,
            params,
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
                self.params.clone(),
            );

        // Determine the GPU dispatch size
        let dispatch_size = dispatch_size(species.shape(), WORK_GROUP_SIZE);

        // Record the simulation steps
        for _ in 0..steps {
            // Set up a descriptor set for the input and output images
            let images = images_descriptor_set(&self.context, &self.pipeline, species)?;

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

        image_device_requirements(device, WORK_GROUP_SIZE)
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

/// GPU compute shader for this backend
mod shader {
    vulkano_shaders::shader! {
        ty: "compute",
        vulkan_version: "1.0",
        spirv_version: "1.0",
        path: "src/main.comp",
    }
}

/// Generate the callback to configure image sampling during GPU compute
/// pipeline construction
pub fn sampler_setup_callback(
    context: &VulkanContext,
) -> Result<impl FnOnce(&mut [DescriptorSetLayoutCreateInfo])> {
    let input_sampler = Sampler::new(
        context.device.clone(),
        SamplerCreateInfo {
            address_mode: [SamplerAddressMode::ClampToBorder; 3],
            border_color: BorderColor::FloatOpaqueBlack,
            unnormalized_coordinates: true,
            ..Default::default()
        },
    )?;
    context.set_debug_utils_object_name(&input_sampler, || "Concentration sampler".into())?;
    Ok(
        move |descriptor_sets: &mut [DescriptorSetLayoutCreateInfo]| {
            fn binding(
                set: &mut DescriptorSetLayoutCreateInfo,
                idx: u32,
            ) -> &mut DescriptorSetLayoutBinding {
                set.bindings.get_mut(&idx).unwrap()
            }
            let images_set = &mut descriptor_sets[IMAGES_SET as usize];
            binding(images_set, IN_U).immutable_samplers = vec![input_sampler.clone()];
            binding(images_set, IN_V).immutable_samplers = vec![input_sampler];
        },
    )
}

/// Acquire a descriptor set to bind the proper images
pub fn images_descriptor_set(
    context: &VulkanContext,
    pipeline: &ComputePipeline,
    species: &mut Species,
) -> Result<Arc<PersistentDescriptorSet>> {
    // Acquire access to the input and output images
    let (in_u, in_v, out_u, out_v) = species.in_out();
    let images = [in_u, in_v, out_u, out_v].map(|i| i.access_image().clone());

    // Have we seen this input + outpt images configuration before?
    match species.context().descriptor_sets.entry(images) {
        // If so, reuse previously configured descriptor set
        Entry::Occupied(occupied) => Ok(occupied.get().clone()),

        // Otherwise, make a new descriptor set
        Entry::Vacant(vacant) => {
            let [in_u, in_v, out_u, out_v] = vacant.key();
            let binding =
                |binding, image: &Arc<StorageImage>, usage| -> Result<WriteDescriptorSet> {
                    Ok(WriteDescriptorSet::image_view(
                        binding,
                        ImageView::new(
                            image.clone(),
                            ImageViewCreateInfo {
                                usage,
                                ..ImageViewCreateInfo::from_image(&image)
                            },
                        )?,
                    ))
                };
            let input_binding = |idx, image| -> Result<WriteDescriptorSet> {
                binding(idx, image, ImageUsage::SAMPLED)
            };
            let output_binding = |idx, image| -> Result<WriteDescriptorSet> {
                binding(idx, image, ImageUsage::STORAGE)
            };
            let layout =
                pipeline.layout().set_layouts()[usize::try_from(IMAGES_SET).unwrap()].clone();
            let set = PersistentDescriptorSet::new(
                &context.descriptor_allocator,
                layout,
                [
                    input_binding(IN_U, in_u)?,
                    input_binding(IN_V, in_v)?,
                    output_binding(OUT_U, out_u)?,
                    output_binding(OUT_V, out_v)?,
                ],
            )?;
            Ok(vacant.insert(set).clone())
        }
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

/// Subset of minimal_device_requirements that will be true for any backend
/// that uses an image-based logic.
pub fn image_device_requirements(device: &PhysicalDevice, work_group_size: [u32; 3]) -> bool {
    let properties = device.properties();
    let Ok(format_properties) = device.format_properties(ImageConcentration::format()) else {
        return false;
    };

    let num_samplers = 2;
    let num_storage_images = 2;

    properties.max_bound_descriptor_sets >= 1
        && properties.max_compute_work_group_invocations
            >= work_group_size.into_iter().product::<u32>()
        && properties
            .max_compute_work_group_size
            .into_iter()
            .zip(work_group_size.into_iter())
            .all(|(max, req)| max >= req)
        && properties.max_descriptor_set_samplers >= num_samplers
        && properties.max_per_stage_descriptor_samplers >= num_samplers
        && properties.max_descriptor_set_storage_images >= num_storage_images
        && properties.max_per_stage_descriptor_storage_images >= num_storage_images
        && properties.max_per_set_descriptors.unwrap_or(u32::MAX)
            >= num_samplers + num_storage_images
        && properties.max_per_stage_resources >= 2 * num_samplers + num_storage_images
        && properties.max_sampler_allocation_count >= num_samplers
        && format_properties.optimal_tiling_features.contains(
            FormatFeatures::SAMPLED_IMAGE
                | FormatFeatures::STORAGE_IMAGE
                | ImageConcentration::required_image_format_features(),
        )
}

/// Way in which we use images
fn image_usage() -> ImageUsage {
    ImageUsage::SAMPLED | ImageUsage::STORAGE
}

/// Requirements on the problem size that are true of any image-based backend
pub fn check_image_shape_requirements(device: &PhysicalDevice, shape: [usize; 2]) -> Result<()> {
    // The simple shader can't accomodate a problem size that is not a
    // multiple of the work group size
    let global_size = shape_to_global_size(shape);
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
    let image_format_info = ImageConcentration::image_format_info(image_usage());
    let Some(image_format_properties) = device.image_format_properties(image_format_info)? else {
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
