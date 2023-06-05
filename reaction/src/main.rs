use clap::Parser;
use compute::Simulate;
use data::{
    concentration::Species,
    hdf5::{self, Writer},
    parameters::Parameters,
    Precision,
};
use indicatif::{ProgressBar, ProgressFinish, ProgressIterator, ProgressStyle};
use std::{num::NonZeroUsize, path::PathBuf, time::Duration};

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
    if #[cfg(feature = "compute_block")] {
        use compute_block::Simulation;
    } else if #[cfg(feature = "compute_autovec")] {
        use compute_autovec::Simulation;
    } else if #[cfg(feature = "compute_manualvec")] {
        use compute_manualvec::Simulation;
    } else if #[cfg(feature = "compute_regular")] {
        use compute_regular::Simulation;
    } else if #[cfg(any(feature = "compute_naive", test))] {
        use compute_naive::Simulation;
    } else {
        // If no backend was specified, use a backend skeleton that shows what
        // the expected interface looks like and throws a compiler error.
        use data::concentration::ScalarConcentration;
        //
        struct Simulation;
        //
        impl Simulate for Simulation {
            type Concentration = ScalarConcentration;

            fn new(_params: Parameters) -> Self {
                std::compile_error!("Please enable at least one compute backend via crate features")
            }

            fn step(&self, _species: &mut Species<ScalarConcentration>) {}
        }
    }
}

fn main() {
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
    });

    // Set up chemical species concentration storage
    type Concentration = <Simulation as Simulate>::Concentration;
    let mut species = Species::<Concentration>::new([args.nbrow, args.nbcol]);
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
        for _ in 0..steps_per_image {
            simulation.step(&mut species);
            species.flip();
        }

        // Write a new image
        writer
            .write(&mut species)
            .expect("Failed to write down results");
    }

    // Make sure output data is written correctly
    writer.close().expect("Failed to close output file");
}
