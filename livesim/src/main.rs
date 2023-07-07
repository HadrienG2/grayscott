mod context;
mod frames;
mod input;
mod palette;
mod pipeline;
mod surface;

use self::{context::SimulationContext, frames::Frames};
use clap::Parser;
use compute::{Simulate, SimulateBase};
use compute_selector::Simulation;
use data::concentration::gpu::shape::Shape;
use std::num::NonZeroU32;
use ui::SharedArgs;
use vulkano::sync::GpuFuture;
use winit::{
    dpi::PhysicalSize,
    event::{Event, WindowEvent},
    event_loop::ControlFlow,
};

/// Use anyhow for error handling
pub use anyhow::Result;

/// Perform Gray-Scott simulation
#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// CLI arguments shared with the "simulate" executable
    #[command(flatten)]
    shared: SharedArgs<Simulation>,

    /// Number of rows processed by each GPU work group during rendering
    ///
    /// The number of simulated rows must be a multiple of this.
    #[arg(long, default_value_t = NonZeroU32::new(8).unwrap())]
    render_work_group_rows: NonZeroU32,

    /// Number of columns processed by each GPU work group during rendering
    ///
    /// The number of simulated columns must be a multiple of this.
    #[arg(long, default_value_t = NonZeroU32::new(8).unwrap())]
    render_work_group_cols: NonZeroU32,

    /// Color palette resolution
    ///
    /// There is a spectrum between producing a live color output that is a
    /// pixel-perfect clone of the `data-to-pics` images on one end, and
    /// maximizing the efficiency/portability of the GPU code on the other end.
    /// This tuning parameter lets you control this tradeoff.
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
    let domain_shape = Shape::new([args.shared.nbrow, args.shared.nbcol]);
    let work_group_shape = Shape::new([
        args.render_work_group_rows.into(),
        args.render_work_group_cols.into(),
    ]);
    let dispatch_size = pipeline::dispatch_size(domain_shape, work_group_shape)?;
    // TODO: Instead of making nbextrastep a tunable of this version too,
    //       consider making it simulate-specific, and rather starting at 1
    //       step/frame, then monitoring the VSync'd framerate and
    //       autotuning the number of simulation steps to fit a frame nicely
    //       with some margin, with 1 step/frame as the minimum for slow
    //       backends.
    let steps_per_image = args.shared.nbextrastep.unwrap_or(1);

    // Set up basic rendering infrastructure
    let (event_loop, window) = surface::create_window(domain_shape)?;
    let context = SimulationContext::new(&args.shared, &window)?;
    let pipeline = pipeline::create(context.vulkan(), work_group_shape)?;

    // Create the color palette and prepare to upload it to the GPU
    let (upload_future, palette_set) =
        palette::create(&context, &pipeline, args.color_palette_resolution)?;

    // Since we have no other initialization work to submit to the GPU, start
    // uploading the color palette right away
    let upload_future = upload_future.then_signal_fence_and_flush()?;

    // Set up simulation domain and associated rendering state
    let mut species = context.simulation().make_species(domain_shape.ndarray())?;
    let mut frames = Frames::new(&context, &pipeline)?;

    // Wait for GPU-side initialization tasks to finish
    upload_future.wait(None)?;
    std::mem::drop(upload_future);

    // Start the event loop
    window.set_visible(true);
    event_loop.run(move |event, _, control_flow| {
        // Continuously run even if no events have been incoming
        control_flow.set_poll();

        // Process incoming events
        match event {
            // Render when all events have been processed
            // TODO: Add fast/async path for GPU backends
            Event::MainEventsCleared => {
                frames
                    .process_frame(&context, &pipeline, |upload_buffer, inout_set| {
                        // Synchronously run the simulation
                        context
                            .simulation()
                            .perform_steps(&mut species, steps_per_image)?;

                        // Make the simulation output available to the GPU
                        input::fill_upload_buffer(upload_buffer, &mut species)?;

                        // Record rendering commands
                        pipeline::record_render_commands(
                            &context,
                            pipeline.clone(),
                            inout_set,
                            palette_set.clone(),
                            dispatch_size,
                        )
                    })
                    .expect("Failed to process simulation frame")
            }

            // Exit when the window is closed
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                *control_flow = ControlFlow::Exit;
            }

            // The easiest way to support resize would probably be to copy
            // simulation data to an image and interpolate using a sampler.
            Event::WindowEvent {
                event: WindowEvent::Resized(PhysicalSize { width, height }),
                ..
            } => {
                assert_eq!(
                    [height as usize, width as usize],
                    domain_shape.ndarray(),
                    "Window resize is not supported yet (and should have been disabled)"
                );
            }

            // Ignore other events
            _ => {}
        }
    })
}
