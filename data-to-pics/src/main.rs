use clap::Parser;
use data::{
    hdf5::{Config, Reader},
    Precision,
};
use image::{Rgb, RgbImage};
use indicatif::{ProgressBar, ProgressFinish, ProgressIterator, ProgressStyle};
use std::{path::PathBuf, time::Duration};

/// Convert Gray-Scott simulation output to images
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the input HDF5 file
    #[arg(short, long)]
    input: PathBuf,

    /// Directory where output images will be saved
    #[arg(short, long)]
    output_dir: Option<PathBuf>,
}

fn main() {
    // Parse CLI arguments
    let args = Args::parse();
    let output_dir = args.output_dir.unwrap_or_else(|| "./".into());

    // Open the HDF5 dataset
    let reader = Reader::open(Config {
        file_name: args.input,
        ..Default::default()
    })
    .expect("Failed to open input file");

    // Setup image rendering
    let gradient = colorous::INFERNO;
    let norm = 1.0 / 0.6 as Precision;
    let [rows, cols] = reader.image_shape();
    let mut image = RgbImage::new(cols as u32, rows as u32);

    // Set up progress reporting
    let progress = ProgressBar::new(reader.num_images() as u64)
        .with_message("Exporting image")
        .with_style(
            ProgressStyle::with_template("{msg} {pos}/{len} {wide_bar} {elapsed}/~{duration}")
                .expect("Failed to parse style"),
        )
        .with_finish(ProgressFinish::AndClear);
    progress.enable_steady_tick(Duration::from_millis(100));

    // Convert HDF5 data to images
    for (idx, input) in reader.enumerate().progress_with(progress) {
        // Load V species concentration matrix
        let input = input.expect("Failed to load image");

        // Generate image
        for (value, pixel) in input.iter().zip(image.pixels_mut()) {
            let color = gradient.eval_continuous((norm * value).into());
            *pixel = Rgb([color.r, color.g, color.b]);
        }

        // Save image
        image
            .save(output_dir.join(format!("{idx}.png")))
            .expect("Failed to save image");
    }
}
