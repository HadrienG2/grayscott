use clap::Parser;
#[cfg(feature = "async-gpu")]
use compute::gpu::SimulateGpu;
use compute::{Simulate, SimulateBase, SimulateCreate};
use compute_selector::Simulation;
use data::{
    concentration::{AsScalars, ScalarConcentration},
    hdf5::{self, Writer},
    parameters::Parameters,
};
use indicatif::{ProgressBar, ProgressFinish, ProgressStyle};
use log::LevelFilter;
use std::{num::NonZeroUsize, path::PathBuf, sync::mpsc, time::Duration};
use syslog::Facility;
use ui::SharedArgs;

/// Perform Gray-Scott simulation
#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// CLI arguments shared with the "livesim" executable
    #[command(flatten)]
    shared: SharedArgs<Simulation>,

    /// Number of images to be created
    #[arg(short, long, default_value_t = 1000)]
    nbimage: usize,

    /// Path to the results output file
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Size of the image buffer between the compute and I/O thread
    ///
    /// A larger buffer enables better performance, at the cost of higher RAM
    /// utilization. 2 is the minimum to fully decouple compute and I/O, higher
    /// values may be beneficial if the I/O backend works in a batched fashion.
    #[arg(long, default_value_t = NonZeroUsize::new(2).unwrap())]
    output_buffer: NonZeroUsize,
}

fn main() {
    // Enable logging to syslog
    syslog::init(
        Facility::default(),
        if cfg!(debug_assertions) {
            LevelFilter::Debug
        } else {
            LevelFilter::Info
        },
        None,
    )
    .expect("Failed to initialize syslog");

    // Parse CLI arguments and handle clap-incompatible defaults
    let args = Args::parse();
    let [kill_rate, feed_rate, time_step] = ui::kill_feed_deltat(&args.shared);
    let file_name = args.output.unwrap_or_else(|| "output.h5".into());

    // Set up the simulation
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

    // Set up chemical species concentration storage
    let mut species = simulation
        .make_species([args.shared.nbrow, args.shared.nbcol])
        .expect("Failed to set up simulation grid");
    let mut writer = Writer::create(
        hdf5::Config {
            file_name,
            ..Default::default()
        },
        &species,
        args.nbimage,
    )
    .expect("Failed to open output file");

    // Set up progress reporting
    let progress = ProgressBar::new(args.nbimage as u64)
        .with_message("Generating image")
        .with_style(
            ProgressStyle::with_template("{msg} {pos}/{len} {wide_bar} {elapsed}/~{duration}")
                .expect("Failed to parse style"),
        )
        .with_finish(ProgressFinish::AndClear);
    progress.enable_steady_tick(Duration::from_millis(100));

    // Set up the HDF5 writer thread
    std::thread::scope(|s| {
        // Start the writer thread
        let (sender, receiver) =
            mpsc::sync_channel::<ScalarConcentration>(args.output_buffer.into());
        let writer = &mut writer;
        s.spawn(move || {
            for result in receiver {
                writer
                    .write(result.view())
                    .expect("Failed to write down results");
                progress.inc(1);
            }
        });

        // Run the simulation on the main thread
        for _ in 0..args.nbimage {
            // Move the simulation forward and collect image
            let image = {
                // If gpu-specific asynchronous commands are available, use them
                #[cfg(feature = "async-gpu")]
                {
                    let steps = simulation
                        .prepare_steps(simulation.now(), &mut species, args.shared.nbextrastep)
                        .expect("Failed to prepare simulation steps");
                    species.access_result(|v, context| {
                        v.make_scalar_view_after(steps, context)
                            .expect("Failed to run simulation and collect results")
                    })
                }

                // Otherwise, use synchronous commands
                #[cfg(not(feature = "async-gpu"))]
                {
                    simulation
                        .perform_steps(&mut species, args.shared.nbextrastep)
                        .expect("Failed to compute simulation steps");
                    species
                        .make_result_view()
                        .expect("Failed to extract result")
                }
            };

            // Schedule writing the image
            sender
                .send(image.as_scalars().to_owned())
                .expect("I/O thread has died");
        }
    });

    // Make sure output data is written correctly
    writer.close().expect("Failed to close output file");
}
