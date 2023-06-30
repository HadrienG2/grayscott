use clap::Parser;
use compute::{gpu::VulkanConfig, Simulate, SimulateBase, SimulateCreate};
use compute_selector::Simulation;
use data::{concentration::AsScalars, parameters::Parameters};
use std::sync::Arc;
use ui::SharedArgs;
use winit::{
    dpi::PhysicalSize,
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Theme, WindowBuilder},
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
    let [kill_rate, feed_rate, time_step] = ui::kill_feed_deltat(&args.shared);

    // Set up an event loop and window
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
            .build(&event_loop)
            .expect("Failed to build window"),
    );

    // Vulkan context configuration
    // TODO: Fill in requirements as we move forward
    let default_config = VulkanConfig::default();
    let vulkan_config = VulkanConfig {
        window: Some(window.clone()),
        ..default_config
    };

    // Set up the simulation
    // TODO: Once ready to share the GPU context, send in our requirements to
    //       GPU backends so their context is right for us!
    let simulation = Simulation::new(
        Parameters {
            kill_rate,
            feed_rate,
            time_step,
            ..Default::default()
        },
        args.shared.backend,
    )
    .expect("Failed to create simulation");

    // TODO: Reuse simulation context instead in GPU mode
    let context = vulkan_config
        .setup()
        .expect("Failed to set up Vulkan context");

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
            // Render when asked to
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
