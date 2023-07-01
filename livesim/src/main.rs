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
use std::{borrow::Borrow, sync::Arc};
use ui::SharedArgs;
use winit::{
    dpi::PhysicalSize,
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Theme, Window, WindowBuilder},
};

/// Perform Gray-Scott simulation
#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// CLI arguments shared with the "livesim" executable
    #[command(flatten)]
    shared: SharedArgs<Simulation>,
}

fn main() {
    // Enable logging to stderr
    env_logger::init();

    // Parse CLI arguments and handle clap-incompatible defaults
    let args = Args::parse();

    // Set up an event loop and window
    let (event_loop, window) = create_window(args.shared.nbcol as u32, args.shared.nbrow as u32);

    // Set up the simulation
    let simulation = create_simulation(&args, &window);

    // Create a Vulkan context, or share that of the simulation if it has one
    let context = get_context(&simulation, &window).borrow();

    // TODO: Create swapchain, upload buffers, pipeline, etc

    // Set up chemical species concentration storage
    let mut species = simulation
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
                simulation
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

// Set up a window and associated event loop
fn create_window(width: u32, height: u32) -> (EventLoop<()>, Arc<Window>) {
    let event_loop = EventLoop::new();
    let window = Arc::new(
        WindowBuilder::new()
            .with_inner_size(PhysicalSize::new(width, height))
            .with_resizable(false)
            .with_title("Gray-Scott reaction")
            .with_visible(false)
            .with_theme(Some(Theme::Dark))
            .build(&event_loop)
            .expect("Failed to build window"),
    );
    (event_loop, window)
}

// Set up the simulation
#[allow(unused)]
fn create_simulation(args: &Args, window: &Arc<Window>) -> Simulation {
    let [kill_rate, feed_rate, time_step] = ui::kill_feed_deltat(&args.shared);
    let parameters = Parameters {
        kill_rate,
        feed_rate,
        time_step,
        ..Default::default()
    };
    {
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
    .expect("Failed to create simulation")
}

// Create a Vulkan context or reuse that of the simulation
#[allow(unused)]
fn get_context<'simulation>(
    simulation: &'simulation Simulation,
    window: &Arc<Window>,
) -> impl Borrow<VulkanContext> + 'simulation {
    #[cfg(feature = "gpu")]
    {
        simulation.context()
    }
    #[cfg(not(feature = "gpu"))]
    {
        vulkan_config(window.clone())
            .setup()
            .expect("Failed to set up Vulkan context")
    }
}

// Vulkan context configuration that we need
fn vulkan_config(window: Arc<Window>) -> VulkanConfig {
    let default_config = VulkanConfig::default();
    VulkanConfig {
        window: Some(window),
        // TODO: Flesh out as code is added
        ..default_config
    }
}
