use anyhow::Result;
use clap::Parser;
#[cfg(feature = "async-gpu")]
use compute::gpu::SimulateGpu;
#[cfg(not(feature = "async-gpu"))]
use compute::Simulate;
use compute::{SimulateBase, SimulateCreate};
use compute_selector::Simulation;
use data::{
    concentration::{AsScalars, ScalarConcentration},
    hdf5::{self, Writer},
    parameters::Parameters,
};
use std::{num::NonZeroUsize, path::PathBuf, sync::mpsc};
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

fn main() -> Result<()> {
    // Enable logging to syslog
    ui::init_syslog();

    // Parse CLI arguments and handle clap-incompatible defaults
    let args = Args::parse();
    let [kill_rate, feed_rate, time_step] = ui::kill_feed_deltat(&args.shared);
    let steps_per_image = args.shared.nbextrastep.unwrap_or(34);
    let file_name = ui::simulation_output_path(args.output);

    // Set up the simulation
    let simulation = Simulation::new(
        Parameters {
            kill_rate,
            feed_rate,
            time_step,
            ..Default::default()
        },
        args.shared.backend,
    )?;

    // Set up chemical species concentration storage
    let mut species = simulation.make_species([args.shared.nbrow, args.shared.nbcol])?;
    let mut writer = Writer::create(
        hdf5::Config {
            file_name,
            ..Default::default()
        },
        &species,
        args.nbimage,
    )?;

    // Set up progress reporting
    let progress = ui::init_progress_reporting("Running simulation step", args.nbimage);

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
                // If GPU-specific asynchronous commands are available, use them
                #[cfg(feature = "async-gpu")]
                {
                    let steps = simulation.prepare_steps(
                        simulation.now(),
                        &mut species,
                        steps_per_image,
                    )?;
                    species.access_result(|v, context| v.make_scalar_view_after(steps, context))?
                }

                // Otherwise, use synchronous commands
                #[cfg(not(feature = "async-gpu"))]
                {
                    simulation.perform_steps(&mut species, steps_per_image)?;
                    species.make_result_view()?
                }
            };

            // Schedule writing the image
            sender.send(image.as_scalars().to_owned())?;
        }
        Ok::<_, anyhow::Error>(())
    })?;

    // Make sure output data is written correctly
    writer.close()?;
    Ok(())
}
