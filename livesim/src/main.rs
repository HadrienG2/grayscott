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
use data::{concentration::AsScalars, parameters::Parameters};
use std::sync::Arc;
use thiserror::Error;
use ui::SharedArgs;
use vulkano::{
    device::{physical::PhysicalDevice, Device},
    format::{Format, NumericType},
    image::{ImageAspects, ImageUsage, SwapchainImage},
    swapchain::{
        ColorSpace, Surface, SurfaceInfo, Swapchain, SwapchainCreateInfo, SwapchainCreationError,
    },
};
use winit::{
    dpi::PhysicalSize,
    error::OsError,
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Theme, Window, WindowBuilder},
};

/// Perform Gray-Scott simulation
#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// CLI arguments shared with the "simulate" executable
    #[command(flatten)]
    shared: SharedArgs<Simulation>,
}

fn main() {
    // Enable logging to stderr
    env_logger::init();

    // Parse CLI arguments and handle clap-incompatible defaults
    let args = Args::parse();

    // Set up an event loop and window
    let (event_loop, window) = create_window(&args).expect("Failed to build window");

    // Set up the simulation and Vulkan context
    let simulation_context = SimulationContext::new(&args, &window)
        .expect("Failed to create simulation and Vulkan context");
    let simulation = &simulation_context.simulation;
    let context = simulation_context.context();
    let device = &context.device;

    // Create swapchain
    let (swapchain, swapchain_images) =
        create_swapchain(device.clone(), simulation_context.surface().clone())
            .expect("Failed to create swapchain");

    // TODO: Create upload buffers as necessary, pipeline, etc (this step will
    //       change once we start sharing data with GPU contexts)

    // Set up chemical species concentration storage
    let mut species = simulation_context
        .simulation
        .make_species([args.shared.nbrow, args.shared.nbcol])
        .expect("Failed to set up simulation grid");

    // Show window and start event loop
    window.set_visible(true);
    event_loop.run(move |event, _, control_flow| {
        // Continuously run even if no events have been incoming
        control_flow.set_poll();

        // Process incoming events
        match event {
            // Render when all events have been processed
            Event::MainEventsCleared => {
                // TODO: Add fast path for GPU backends
                simulation_context
                    .simulation
                    .perform_steps(&mut species, args.shared.nbextrastep)
                    .expect("Failed to compute simulation steps");
                species
                    .make_result_view()
                    .expect("Failed to extract result")
                    .as_scalars();

                // TODO: Add rendering

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
    });
}

/// Set up a window and associated event loop
fn create_window(args: &Args) -> Result<(EventLoop<()>, Arc<Window>), OsError> {
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
///
/// This needs to be a thing because we can't permanently borrow the
/// Simulation's context, as it would prevent moving the Simulation into winit's
/// event loop closure (as doing so would invalidate the borrow's pointer).
struct SimulationContext {
    /// Gray-Scott reaction simulation, which may have its own VulkanContext
    simulation: Simulation,

    /// Custom VulkanContext if `simulation` doesn't has one
    #[cfg(not(feature = "gpu"))]
    context: VulkanContext,
}
//
impl SimulationContext {
    /// Set up the simulation and Vulkan context
    fn new(args: &Args, window: &Arc<Window>) -> Result<Self, SimulationContextError> {
        let [kill_rate, feed_rate, time_step] = ui::kill_feed_deltat(&args.shared);
        let parameters = Parameters {
            kill_rate,
            feed_rate,
            time_step,
            ..Default::default()
        };

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
        }
        .map_err(SimulationContextError::Simulation)?;

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
    fn context(&self) -> &VulkanContext {
        #[cfg(feature = "gpu")]
        {
            self.simulation.context()
        }
        #[cfg(not(feature = "gpu"))]
        {
            &self.context
        }
    }

    /// Get access to the rendering surface (we know that our context was
    /// configured with a window, and should thus have set up a surface)
    fn surface(&self) -> &Arc<Surface> {
        self.context().surface.as_ref().expect("Should be there")
    }
}
//
#[derive(Debug, Error)]
enum SimulationContextError {
    #[error("failed to create simulation")]
    Simulation(<Simulation as SimulateBase>::Error),

    #[error("failed to create Vulkan context")]
    VulkanContext(#[from] compute::gpu::Error),
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
    let Some(color_type) = format.type_color() else { return false };
    format.aspects().contains(ImageAspects::COLOR)
        && format.components().iter().take(3).all(|&bits| bits > 0)
        && color_type == NumericType::SRGB
        && colorspace == ColorSpace::SrgbNonLinear
}

/// Create a swapchain
fn create_swapchain(
    device: Arc<Device>,
    surface: Arc<Surface>,
) -> Result<(Arc<Swapchain>, Vec<Arc<SwapchainImage>>), SwapchainCreationError> {
    let physical_device = device.physical_device();
    let surface_info = SurfaceInfo::default();
    let surface_capabilities = physical_device
        .surface_capabilities(&surface, surface_info.clone())
        .expect("Failed to query surface capabilities");
    let (image_format, image_color_space) = physical_device
        .surface_formats(&surface, surface_info)
        .expect("Failed to query surface formats")
        .into_iter()
        .find(|format| is_supported_format(*format))
        .expect("There should be at least one supported format");
    Swapchain::new(
        device.clone(),
        surface.clone(),
        SwapchainCreateInfo {
            min_image_count: surface_capabilities.min_image_count.max(2),
            image_format: Some(image_format),
            image_color_space,
            image_usage: ImageUsage::STORAGE,
            ..Default::default()
        },
    )
}
