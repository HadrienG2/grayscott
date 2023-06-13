use clap::Parser;
use compute::{Simulate, SimulateBase};
use data::{
    hdf5::{self, Writer},
    parameters::Parameters,
    Precision,
};
use indicatif::{ProgressBar, ProgressFinish, ProgressIterator, ProgressStyle};
use log::LevelFilter;
use std::{num::NonZeroUsize, path::PathBuf, time::Duration};
use syslog::Facility;

/// Convert Gray-Scott simulation output to images
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Rate of the process which converts V into P
    #[arg(short, long)]
    killrate: Option<Precision>,

    /// Rate of the process which feeds U and drains U, V and P
    #[arg(short, long)]
    feedrate: Option<Precision>,

    /// Number of images to be created
    #[arg(short, long, default_value_t = 100)]
    nbimage: usize,

    /// Number of steps to be computed between images
    #[arg(short = 'e', long, default_value_t = NonZeroUsize::new(1).unwrap())]
    nbextrastep: NonZeroUsize,

    /// Number of rows of the images to be created
    #[arg(short = 'r', long, default_value_t = 100)]
    nbrow: usize,

    /// Number of columns of the images to be created
    #[arg(short = 'c', long, default_value_t = 200)]
    nbcol: usize,

    /// Time interval between two computations
    #[arg(short = 't', long)]
    deltat: Option<Precision>,

    /// Path to the results output file
    #[arg(short, long)]
    output: Option<PathBuf>,
}

// Use the best compute backend allowed by enabled crate features
cfg_if::cfg_if! {
    // TODO: Add more advanced and preferrable implementations above
    if #[cfg(feature = "compute_parallel")] {
        type Simulation = compute_parallel::Simulation;
    } else if #[cfg(feature = "compute_block")] {
        type Simulation = compute_block::Simulation;
    } else if #[cfg(feature = "compute_autovec")] {
        type Simulation = compute_autovec::Simulation;
    } else if #[cfg(feature = "compute_manualvec")] {
        type Simulation = compute_manualvec::Simulation;
    } else if #[cfg(feature = "compute_regular")] {
        type Simulation = compute_regular::Simulation;
    } else if #[cfg(any(feature = "compute_naive", test))] {
        type Simulation = compute_naive::Simulation;
    } else {
        // If no backend was specified, use a backend skeleton that throws a
        // minimal number of compiler errors.
        use data::concentration::{ScalarConcentration, Species};
        use std::convert::Infallible;
        //
        struct Simulation;
        //
        impl SimulateBase for Simulation {
            type Concentration = ScalarConcentration;

            type Error = Infallible;

            fn new(_params: Parameters) -> Result<Self, Infallible> {
                std::compile_error!("Please enable at least one compute backend via crate features")
            }

            fn make_species(&self, shape: [usize; 2]) -> Result<Species<ScalarConcentration>, Infallible> {
                Species::new((), shape)
            }
        }
        //
        impl Simulate for Simulation {
            fn perform_steps(
                &self,
                _species: &mut Species<ScalarConcentration>,
                _steps: usize
            ) -> Result<(), Infallible> {
                Ok(())
            }
        }
    }
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

    // Parse CLI arguments and handle unconventional defaults
    let args = Args::parse();
    let kill_rate = args.killrate.unwrap_or(Parameters::default().kill_rate);
    let feed_rate = args.feedrate.unwrap_or(Parameters::default().feed_rate);
    let time_step = args.deltat.unwrap_or(Parameters::default().time_step);
    let file_name = args.output.unwrap_or_else(|| "output.h5".into());
    let steps_per_image = usize::from(args.nbextrastep);

    // Set up the simulation
    let simulation = Simulation::new(Parameters {
        kill_rate,
        feed_rate,
        time_step,
        ..Default::default()
    })
    .expect("Failed to create simulation");

    // Set up chemical species concentration storage
    let mut species = simulation
        .make_species([args.nbrow, args.nbcol])
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

    // Run the simulation
    for _ in (0..args.nbimage).progress_with(progress) {
        // Move the simulation forward
        simulation
            .perform_steps(&mut species, steps_per_image)
            .expect("Failed to compute simulation steps");

        // Write a new image
        writer
            .write(
                species
                    .make_result_view()
                    .expect("Failed to extract result"),
            )
            .expect("Failed to write down results");
    }

    // Make sure output data is written correctly
    writer.close().expect("Failed to close output file");
}
