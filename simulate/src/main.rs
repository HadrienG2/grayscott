use anyhow::Result;
use clap::Parser;
#[cfg(feature = "async-gpu")]
use compute::gpu::SimulateGpu;
#[cfg(not(feature = "async-gpu"))]
use compute::Simulate;
use compute::{SimulateBase, SimulateCreate};
use compute_selector::Simulation;
use data::{
    concentration::ScalarConcentration,
    hdf5::{self, Writer},
    parameters::Parameters,
};
use ndarray::Array2;
use std::{
    num::NonZeroUsize,
    path::PathBuf,
    sync::mpsc::{self, TryRecvError},
};
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
    let [kill_rate, feed_rate, time_step] = args.shared.kill_feed_deltat();
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
    let mut species = simulation.make_species(args.shared.domain_shape())?;
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

    // Set up the I/O thread
    std::thread::scope(|s| {
        // HDF5 writer thread writes down simulated data to HDF5
        let (image_send, image_recv) =
            mpsc::sync_channel::<ScalarConcentration>(args.output_buffer.into());
        let (image_recycle_send, image_recycle_recv) = mpsc::channel();
        let writer = &mut writer;
        s.spawn(move || {
            for image in image_recv {
                writer
                    .write(image.view())
                    .expect("Failed to write down results");
                let _ = image_recycle_send.send(image);
                progress.inc(1);
            }
        });

        // Run the simulation on the main thread
        for _ in 0..args.nbimage {
            // Allocate or reuse output data storage
            let mut image = match image_recycle_recv.try_recv() {
                Ok(image) => image,
                Err(TryRecvError::Empty) => Array2::from_elem(species.shape(), 0.0),
                Err(e @ TryRecvError::Disconnected) => return Err(e.into()),
            };

            // If GPU-specific asynchronous commands are available, use them
            #[cfg(feature = "async-gpu")]
            {
                let steps =
                    simulation.prepare_steps(simulation.now(), &mut species, steps_per_image)?;
                species.access_result(|v, context| {
                    v.write_scalar_view_after(steps, context, image.view_mut())
                })?;
            }

            // Otherwise, use synchronous commands
            #[cfg(not(feature = "async-gpu"))]
            {
                simulation.perform_steps(&mut species, steps_per_image)?;
                species.write_result_view(image.view_mut())?
            }

            // Schedule writing the image
            image_send.send(image)?;
        }
        Ok::<_, anyhow::Error>(())
    })?;

    // Make sure output data is written correctly
    writer.close()?;
    Ok(())
}
