use anyhow::Result;
use clap::Parser;
#[cfg(feature = "gpu")]
use compute::gpu::SimulateGpu;
#[cfg(not(feature = "gpu"))]
use compute::SimulateCreate;
use compute::{
    gpu::{VulkanConfig, VulkanContext},
    Simulate, SimulateBase,
};
use compute_selector::Simulation;
use data::{concentration::AsScalars, parameters::Parameters, Precision};
use log::info;
use std::{num::NonZeroU32, sync::Arc};
use ui::SharedArgs;
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage},
    descriptor_set::{
        layout::DescriptorSetLayoutCreateInfo, PersistentDescriptorSet, WriteDescriptorSet,
    },
    device::{physical::PhysicalDevice, Queue},
    format::{Format, NumericType},
    image::{
        view::{ImageView, ImageViewCreateInfo},
        ImageAccess, ImageAspects, ImageDimensions, ImageUsage, ImmutableImage, MipmapsCount,
        SwapchainImage,
    },
    memory::allocator::{AllocationCreateInfo, MemoryUsage},
    pipeline::{ComputePipeline, Pipeline},
    sampler::{Filter, Sampler, SamplerCreateInfo},
    swapchain::{ColorSpace, Surface, SurfaceInfo, Swapchain, SwapchainCreateInfo},
    sync::GpuFuture,
};
use winit::{
    dpi::PhysicalSize,
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Theme, Window, WindowBuilder},
};

/// Shader descriptor set to which input and output data are bound
pub const INOUT_SET: u32 = 0;

/// Descriptor within `INOUT_SET` for readout of simulation output
const DATA_INPUT: u32 = 0;

/// Descriptor within `INOUT_SET` for writing to screen
const SCREEN_OUTPUT: u32 = 1;

/// Shader descriptor set to which the color palette is bound
const PALETTE_SET: u32 = 1;

/// Perform Gray-Scott simulation
#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// CLI arguments shared with the "simulate" executable
    #[command(flatten)]
    shared: SharedArgs<Simulation>,

    /// Number of rows processed by each GPU work group during rendering
    #[arg(long, default_value_t = NonZeroU32::new(8).unwrap())]
    render_work_group_rows: NonZeroU32,

    /// Number of columns processed by each GPU work group during rendering
    #[arg(long, default_value_t = NonZeroU32::new(8).unwrap())]
    render_work_group_cols: NonZeroU32,

    /// Color palette resolution
    ///
    /// There is a spectrum between producing a live color output that is a
    /// pixel-perfect clone of the `data-to-pics` images' on one end, and
    /// maximizing the efficiency/portability of the GPU code on the other end.
    /// This tuning parameter lets you control this compromise.
    ///
    /// Must be at least 2. Higher is more accurate/expensive and less portable.
    #[arg(long, default_value_t = 256)]
    color_palette_resolution: u32,
}

fn main() -> Result<()> {
    // Enable logging to stderr
    env_logger::init();

    // Parse CLI arguments and handle clap-incompatible defaults
    let args = Args::parse();
    // TODO: Autotune this, or at least autotune from this
    let steps_per_image = args.shared.nbextrastep.unwrap_or(1);

    // Set up an event loop and window
    let (event_loop, window) = create_window(&args)?;

    // Pick work-group size
    let work_group_size = [
        args.render_work_group_cols.into(),
        args.render_work_group_rows.into(),
        1,
    ];

    // Set up the simulation and Vulkan context
    let simulation_context = SimulationContext::new(&args, &window)?;

    // Set up the rendering pipeline
    let pipeline = create_pipeline(&simulation_context, work_group_size)?;

    // Create the color palette and prepare to upload it to the GPU
    let (upload_future, palette) = create_color_palette(
        &simulation_context,
        &pipeline,
        args.color_palette_resolution,
    )?;

    // Since we have no other initialization work to submit to the GPU, start
    // uploading the color palette right away
    let upload_future = upload_future.then_signal_fence_and_flush()?;

    // Set up a swapchain
    let (swapchain, swapchain_images) = create_swapchain(&simulation_context)?;

    // Set up chemical species concentration storage
    let mut species = simulation_context
        .simulation
        .make_species([args.shared.nbrow, args.shared.nbcol])?;

    // Set up buffers to upload simulation results to the GPU
    // TODO: If the simulation backend is GPU-based, directly access simulation
    //       storage instead (will require a different rendering pipeline).
    let upload_buffers = create_upload_buffers(
        &simulation_context,
        &pipeline,
        species.shape(),
        swapchain_images.len(),
    )?;

    // TODO: Other per-frame state + current frame tracking

    // Wait for GPU-side initialization tasks to finish
    upload_future.wait(None)?;

    // Show window and start event loop
    window.set_visible(true);
    event_loop.run(move |event, _, control_flow| {
        // Continuously run even if no events have been incoming
        control_flow.set_poll();

        // Process incoming events
        match event {
            // Render when all events have been processed
            Event::MainEventsCleared => {
                // TODO: Add fast/async path for GPU backends
                simulation_context
                    .simulation
                    .perform_steps(&mut species, steps_per_image)
                    .expect("Failed to compute simulation steps");

                // TODO: Add rendering, use write_result_view instead
                species
                    .make_result_view()
                    .expect("Failed to extract result")
                    .as_scalars();

                // TODO: Instead of making nbextrastep a tunable of this version too,
                //       consider making it simulate-specific, and rather starting at 1
                //       step/frame, then monitoring the VSync'd framerate and
                //       autotuning the number of simulation steps to fit a frame nicely
                //       with some margin, with 1 step/frame as the minimum for slow
                //       backends.
            }

            // Exit when the window is closed
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                *control_flow = ControlFlow::Exit;
            }

            // Ignore other events
            _ => {}
        }
    })
}

/// Set up a window and associated event loop
fn create_window(args: &Args) -> Result<(EventLoop<()>, Arc<Window>)> {
    let event_loop = EventLoop::new();
    let window = Arc::new(
        WindowBuilder::new()
            .with_inner_size(PhysicalSize::new(
                args.shared.nbcol as u32,
                args.shared.nbrow as u32,
            ))
            .with_resizable(false)
            .with_title("Gray-Scott reaction")
            .with_visible(false)
            .with_theme(Some(Theme::Dark))
            .build(&event_loop)?,
    );
    Ok((event_loop, window))
}

/// Simulation context = Simulation + custom VulkanContext if needed, lets you
/// borrow either the simulation or the VulkanContext
struct SimulationContext {
    /// Gray-Scott reaction simulation, which may have its own VulkanContext
    simulation: Simulation,

    /// Custom VulkanContext if `simulation` doesn't have one
    #[cfg(not(feature = "gpu"))]
    context: VulkanContext,
}
//
impl SimulationContext {
    /// Set up the simulation and Vulkan context
    fn new(args: &Args, window: &Arc<Window>) -> Result<Self> {
        // Configure simulation
        let [kill_rate, feed_rate, time_step] = ui::kill_feed_deltat(&args.shared);
        let parameters = Parameters {
            kill_rate,
            feed_rate,
            time_step,
            ..Default::default()
        };

        // Create simulation, forwarding our context config if it's Vulkan-based
        let simulation = {
            #[cfg(feature = "gpu")]
            {
                Simulation::with_config(
                    parameters,
                    args.shared.backend,
                    vulkan_config(window.clone()),
                )
            }
            #[cfg(not(feature = "gpu"))]
            {
                Simulation::new(parameters, args.shared.backend)
            }
        }?;

        // Create a dedicated context if the simulation is not Vulkan-based
        {
            #[cfg(feature = "gpu")]
            {
                Ok(Self { simulation })
            }
            #[cfg(not(feature = "gpu"))]
            {
                let context = vulkan_config(window.clone()).setup()?;
                Ok(Self {
                    simulation,
                    context,
                })
            }
        }
    }

    /// Get access to the vulkan context
    fn vulkan_context(&self) -> &VulkanContext {
        #[cfg(feature = "gpu")]
        {
            self.simulation.context()
        }
        #[cfg(not(feature = "gpu"))]
        {
            &self.context
        }
    }

    /// Get access to the rendering surface
    fn surface(&self) -> &Arc<Surface> {
        self.vulkan_context()
            .surface
            .as_ref()
            .expect("There should be one (window specified in VulkanConfig)")
    }

    /// Get access to the rendering queue (only queue currently used)
    fn queue(&self) -> &Arc<Queue> {
        &self.vulkan_context().queues[0]
    }
}

/// Vulkan context configuration that we need
fn vulkan_config(window: Arc<Window>) -> VulkanConfig {
    let default_config = VulkanConfig::default();
    VulkanConfig {
        window_and_reqs: Some((window, Box::new(surface_requirements))),
        // TODO: Flesh out as code is added
        ..default_config
    }
}

/// Surface-dependent device requirements
fn surface_requirements(device: &PhysicalDevice, surface: &Surface) -> bool {
    let surface_info = SurfaceInfo::default();
    device
        .surface_formats(surface, surface_info.clone())
        .map(|vec| vec.into_iter().any(is_supported_format))
        .unwrap_or(false)
        && device
            .surface_capabilities(surface, surface_info)
            .map(|caps| {
                caps.max_image_count.unwrap_or(u32::MAX) >= 2
                    && caps.supported_usage_flags.contains(ImageUsage::STORAGE)
            })
            .unwrap_or(false)
}

/// Supported surface formats
fn is_supported_format((format, colorspace): (Format, ColorSpace)) -> bool {
    let Some(color_type) = format.type_color() else {
        return false;
    };
    format.aspects().contains(ImageAspects::COLOR)
        && format.components().iter().take(3).all(|&bits| bits > 0)
        // This may seem surprising given that the source data is
        // NumericType::SRGB, but remember that Vulkan implicitly performs an
        // sRGB -> linear conversion when a shader loads a texel.
        && color_type == NumericType::UNORM
        && colorspace == ColorSpace::SrgbNonLinear
}

/// Create the rendering pipeline
fn create_pipeline(
    simulation_context: &SimulationContext,
    work_group_size: [u32; 3],
) -> Result<Arc<ComputePipeline>> {
    // Load the rendering shader
    let context = simulation_context.vulkan_context();
    let shader = shader::load(context.device.clone())?;
    context.set_debug_utils_object_name(&shader, || "Live renderer shader".into())?;

    // Set up the rendering pipeline
    let pipeline = ComputePipeline::new(
        context.device.clone(),
        shader.entry_point("main").expect("Should be present"),
        &shader::SpecializationConstants {
            constant_0: work_group_size[0],
            constant_1: work_group_size[1],
        },
        Some(context.pipeline_cache.clone()),
        sampler_setup_callback(context)?,
    )?;
    context.set_debug_utils_object_name(&pipeline, || "Live renderer".into())?;
    Ok(pipeline)
}

// Rendering shader used when data comes from the CPU
// TODO: Use different shaders when rendering from GPU data
mod shader {
    vulkano_shaders::shader! {
        ty: "compute",
        vulkan_version: "1.0",
        spirv_version: "1.0",
        path: "src/cpu.comp",
    }
}

/// Generate the callback to configure palette sampling during rendering
/// pipeline construction
pub fn sampler_setup_callback(
    context: &VulkanContext,
) -> Result<impl FnOnce(&mut [DescriptorSetLayoutCreateInfo])> {
    let palette_sampler = Sampler::new(
        context.device.clone(),
        SamplerCreateInfo {
            mag_filter: Filter::Linear,
            min_filter: Filter::Linear,
            ..Default::default()
        },
    )?;
    context.set_debug_utils_object_name(&palette_sampler, || "Color palette sampler".into())?;
    Ok(
        move |descriptor_sets: &mut [DescriptorSetLayoutCreateInfo]| {
            descriptor_sets[PALETTE_SET as usize]
                .bindings
                .get_mut(&0)
                .expect("Color palette descriptor should be present")
                .immutable_samplers = vec![palette_sampler];
        },
    )
}

/// Create the color palette used for data -> color translation
///
/// Returns a GPU future that will be signaled once the palette has been
/// uploaded to the GPU, along with a descriptor set that can be used in order
/// to bind the palette to the rendering pipeline.
fn create_color_palette(
    simulation_context: &SimulationContext,
    pipeline: &ComputePipeline,
    width: u32,
) -> Result<(impl GpuFuture, Arc<PersistentDescriptorSet>)> {
    // Prepare to upload color palette
    assert!(width >= 2, "Color palette must have at least two endpoints");
    let context = simulation_context.vulkan_context();
    let upload_queue = simulation_context.queue();
    let mut upload_builder = AutoCommandBufferBuilder::primary(
        &context.command_allocator,
        upload_queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )?;

    // Create color palette, record upload commands
    let palette_image = ImmutableImage::from_iter(
        &context.memory_allocator,
        (0..width).map(|idx| {
            let position = idx as f64 / (width - 1) as f64;
            let color = ui::GRADIENT.eval_continuous(position);
            [color.b, color.g, color.r, 255]
        }),
        ImageDimensions::Dim1d {
            width,
            array_layers: 1,
        },
        MipmapsCount::One,
        Format::B8G8R8A8_SRGB,
        &mut upload_builder,
    )?;
    context.set_debug_utils_object_name(&palette_image.inner().image, || "Color palette".into())?;

    // Create palette descriptor set
    let palette = PersistentDescriptorSet::new(
        &context.descriptor_allocator,
        pipeline.layout().set_layouts()[PALETTE_SET as usize].clone(),
        [WriteDescriptorSet::image_view(
            0,
            ImageView::new(
                palette_image.clone(),
                ImageViewCreateInfo {
                    usage: ImageUsage::SAMPLED,
                    ..ImageViewCreateInfo::from_image(&palette_image)
                },
            )?,
        )],
    )?;
    // FIXME: Name this descriptor set once vulkano allows for it

    // Schedule upload
    let upload = upload_builder.build()?;
    context.set_debug_utils_object_name(&upload, || "Color palette upload".into())?;
    let upload_future =
        vulkano::sync::now(context.device.clone()).then_execute(upload_queue.clone(), upload)?;
    Ok((upload_future, palette))
}

/// Create a swapchain
fn create_swapchain(
    simulation_context: &SimulationContext,
) -> Result<(Arc<Swapchain>, Vec<Arc<SwapchainImage>>)> {
    let context = simulation_context.vulkan_context();
    let physical_device = context.device.physical_device();
    let surface = simulation_context.surface();

    let surface_info = SurfaceInfo::default();
    let surface_capabilities =
        physical_device.surface_capabilities(surface, surface_info.clone())?;
    let (image_format, image_color_space) = physical_device
        .surface_formats(surface, surface_info)?
        .into_iter()
        .find(|format| is_supported_format(*format))
        .expect("There should be one (checked at device creation time)");

    let create_info = SwapchainCreateInfo {
        min_image_count: surface_capabilities.min_image_count.max(2),
        image_format: Some(image_format),
        image_color_space,
        image_usage: ImageUsage::STORAGE,
        ..Default::default()
    };
    info!("Will now create a swapchain with {create_info:#?}");

    let (swapchain, swapchain_images) =
        Swapchain::new(context.device.clone(), surface.clone(), create_info)?;
    context.set_debug_utils_object_name(&swapchain, || "Rendering swapchain".into())?;
    // FIXME: Name swapchain image once vulkano allows for it

    Ok((swapchain, swapchain_images))
}

/// Create buffers for upload of simulation output to the GPU
///
/// Returns the buffers along with persistent descriptor sets for binding them
/// to the rendering pipeline.
//
// TODO: Disable this function in optimized GPU mode
fn create_upload_buffers(
    simulation_context: &SimulationContext,
    pipeline: &ComputePipeline,
    shape: [usize; 2],
    count: usize,
) -> Result<Vec<(Subbuffer<[Precision]>, Arc<PersistentDescriptorSet>)>> {
    let context = simulation_context.vulkan_context();
    let buffer_len = shape.into_iter().product::<usize>();
    (0..count)
        .map(|_| {
            let buffer = Buffer::new_slice::<Precision>(
                &context.memory_allocator,
                BufferCreateInfo {
                    usage: BufferUsage::STORAGE_BUFFER,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    usage: MemoryUsage::Upload,
                    ..Default::default()
                },
                buffer_len as u64,
            )?;
            let descriptor_set = PersistentDescriptorSet::new(
                &context.descriptor_allocator,
                pipeline.layout().set_layouts()[INOUT_SET as usize].clone(),
                [WriteDescriptorSet::buffer(DATA_INPUT, buffer.clone())],
            )?;
            Ok((buffer, descriptor_set))
        })
        .collect()
}
