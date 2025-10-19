mod context;
mod frames;
mod input;
mod palette;
mod pipeline;
mod surface;

use self::{context::SimulationContext, frames::Frames};
use clap::Parser;
use compute::{DenormalsFlusher, Simulate, SimulateBase};
use compute_selector::Simulation;
use data::concentration::{gpu::shape::Shape, Species};
use std::{num::NonZeroU32, sync::Arc};
use ui::SharedArgs;
use vulkano::{descriptor_set::DescriptorSet, pipeline::ComputePipeline, sync::GpuFuture};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::{StartCause, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::WindowId,
};

/// Use eyre for error handling
pub use eyre::Result;

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

    // Set up the event loop and app window
    struct WindowState {
        context: SimulationContext,
        pipeline: Arc<ComputePipeline>,
        palette_set: Arc<DescriptorSet>,
        species: Species<<compute_selector::Simulation as SimulateBase>::Concentration>,
        frames: Frames,
    }
    //
    impl WindowState {
        fn new(
            args: &Args,
            domain_shape: Shape,
            work_group_shape: Shape,
            event_loop: &ActiveEventLoop,
        ) -> Result<Self> {
            // Set up the window
            let window = surface::create_window(event_loop, domain_shape)?;
            window.set_visible(true);

            // Set up the basic rendering infrastructure
            let context = SimulationContext::new(&args.shared, &window)?;
            let pipeline = pipeline::create(context.vulkan(), work_group_shape)?;

            // Create the color palette and prepare to upload it to the GPU
            let (upload_future, palette_set) =
                palette::create(&context, &pipeline, args.color_palette_resolution)?;

            // Since we have no other initialization work to submit to the GPU, start
            // uploading the color palette right away
            let upload_future = upload_future.then_signal_fence_and_flush()?;

            // Set up simulation domain and associated rendering state
            let species = context.simulation().make_species(domain_shape.ndarray())?;
            let frames = Frames::new(&context, &pipeline, domain_shape)?;

            // Wait for GPU-side initialization tasks to finish
            upload_future.wait(None)?;
            std::mem::drop(upload_future);
            Ok(Self {
                context,
                pipeline,
                palette_set,
                species,
                frames,
            })
        }
    }
    //
    struct App {
        args: Args,
        domain_shape: Shape,
        work_group_shape: Shape,
        dispatch_size: [u32; 3],
        steps_per_image: usize,
        window_state: Option<WindowState>,
    }
    //
    impl ApplicationHandler for App {
        fn new_events(&mut self, event_loop: &ActiveEventLoop, cause: StartCause) {
            if cause == StartCause::Init {
                // Continuously run even if no events have been incoming
                event_loop.set_control_flow(ControlFlow::Poll);
            }
        }

        fn resumed(&mut self, event_loop: &ActiveEventLoop) {
            // Ignore app resumption after initial setup
            if self.window_state.is_some() {
                return;
            }

            // Set up the window and associated app state
            self.window_state = Some(
                WindowState::new(
                    &self.args,
                    self.domain_shape,
                    self.work_group_shape,
                    event_loop,
                )
                .expect("Failed to set up window and associated state"),
            );
        }

        fn window_event(
            &mut self,
            event_loop: &ActiveEventLoop,
            _window_id: WindowId,
            event: WindowEvent,
        ) {
            match event {
                // Exit when the window is closed
                WindowEvent::CloseRequested => event_loop.exit(),

                // The easiest way to support resize would probably be to copy
                // simulation data to an image and interpolate using a sampler.
                WindowEvent::Resized(PhysicalSize { width, height }) => {
                    log::error!("Ignoring meaningless window resizing to {width}x{height} from buggy window manager");
                    /* FIXME: Understand why it really happens
                    assert_eq!(
                        [height as usize, width as usize],
                        domain_shape.ndarray(),
                        "Window resize is not supported yet (and should have been disabled)"
                    );*/
                }

                // Ignore other events
                _ => {}
            }
        }

        fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
            let state = self
                .window_state
                .as_mut()
                .expect("Should not occur before resume()");
            state
                .frames
                .process_frame(
                    &state.context,
                    &state.pipeline,
                    |upload_buffer, inout_set| {
                        // Synchronously run the simulation
                        {
                            let _flush_denormals = DenormalsFlusher::new();
                            state
                                .context
                                .simulation()
                                .perform_steps(&mut state.species, self.steps_per_image)?;
                        }

                        // Make the simulation output available to the GPU
                        input::fill_upload_buffer(upload_buffer, &mut state.species)?;

                        // Record rendering commands
                        pipeline::record_render_commands(
                            &state.context,
                            state.pipeline.clone(),
                            inout_set,
                            state.palette_set.clone(),
                            self.dispatch_size,
                        )
                    },
                )
                .expect("Failed to process simulation frame")
        }
    }
    //
    let mut app = App {
        args,
        domain_shape,
        work_group_shape,
        dispatch_size,
        steps_per_image,
        window_state: None,
    };
    //
    EventLoop::new()?.run_app(&mut app)?;
    Ok(())
}
