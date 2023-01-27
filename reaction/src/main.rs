use clap::Parser;
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

// Select compute backend depending on enabled crate features
cfg_if::cfg_if! {
    // TODO: Add more advanced and preferrable implementations above
    if #[cfg(feature = "regular")] {
        use compute_regular::step;
    } else if #[cfg(any(feature = "naive", test))] {
        use compute_naive::step;
    } else {
        #[allow(non_upper_case_globals)]
        const step: fn(&mut Species, &Parameters) =
            panic!("Please enable at least one compute backend via crate features");
    }
}

fn main() {
    // Parse CLI arguments and handle unconventional defaults
    let args = Args::parse();
    let kill_rate = args.killrate.unwrap_or(Parameters::default().kill_rate);
    let feed_rate = args.feedrate.unwrap_or(Parameters::default().feed_rate);
    let time_step = args.deltat.unwrap_or(Parameters::default().time_step);
    let file_name = args.output.unwrap_or("output.h5".into());
    let steps_per_image = usize::from(args.nbextrastep);

    // Determine computation parameters
    let params = Parameters {
        kill_rate,
        feed_rate,
        time_step,
        ..Default::default()
    };

    // Set up chemical species concentration storage
    let mut species = Species::new([args.nbrow, args.nbcol]);
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
            ProgressStyle::with_template("{msg} {pos}/{len} {wide_bar} ~{eta} left")
                .expect("Failed to parse style"),
        )
        .with_finish(ProgressFinish::AndClear);
    progress.enable_steady_tick(Duration::from_millis(33));

    // Run the simulation
    for _ in (0..args.nbimage).progress_with(progress) {
        // Move the simulation forward
        for _ in 0..steps_per_image {
            step(&mut species, &params);
            species.flip();
        }

        // Write a new image
        writer
            .write(&species)
            .expect("Failed to write down results");
    }

    // Make sure output data is written correctly
    writer.close().expect("Failed to close output file");
}
