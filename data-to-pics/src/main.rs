use clap::Parser;
use data::{
    hdf5::{Config, Reader},
    Precision,
};
use image::{ImageBuffer, Rgb, RgbImage};
use indicatif::{ProgressBar, ProgressFinish, ProgressStyle};
use ndarray::Axis;
use rayon::prelude::*;
use std::{
    num::NonZeroUsize,
    path::PathBuf,
    sync::mpsc::{self, TryRecvError},
    time::Duration,
};

/// Convert Gray-Scott simulation output to images
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the input HDF5 file
    #[arg(short, long)]
    input: Option<PathBuf>,

    /// Size of the image buffer between HDF5 I/O and the conversion threads
    ///
    /// A larger buffer enables better performance, at the cost of higher RAM
    /// utilization. 2 is the minimum to fully decouple compute and I/O, higher
    /// values may be beneficial if the I/O backend works in a batched fashion.
    #[arg(long, default_value_t = NonZeroUsize::new(2).unwrap())]
    input_buffer: NonZeroUsize,

    /// Directory where output images will be saved
    #[arg(short, long)]
    output_dir: PathBuf,

    /// Size of the image buffer between the conversion threads and image I/O
    ///
    /// A larger buffer enables better performance, at the cost of higher RAM
    /// utilization. 2 is the minimum to fully decouple compute and I/O, higher
    /// values may be beneficial if the I/O backend works in a batched fashion.
    #[arg(long, default_value_t = NonZeroUsize::new(2).unwrap())]
    output_buffer: NonZeroUsize,
}

fn main() {
    // Parse CLI arguments
    let args = Args::parse();

    // Open the HDF5 dataset
    let reader = Reader::open(Config {
        file_name: args.input.unwrap_or_else(|| "output.h5".into()),
        ..Default::default()
    })
    .expect("Failed to open input file");

    // Setup image rendering
    let gradient = colorous::INFERNO;
    let norm = 1.0 / 0.6 as Precision;
    let [rows, cols] = reader.image_shape();

    // Set up progress reporting
    let progress = ProgressBar::new(reader.num_images() as u64)
        .with_message("Exporting image")
        .with_style(
            ProgressStyle::with_template("{msg} {pos}/{len} {wide_bar} {elapsed}/~{duration}")
                .expect("Failed to parse style"),
        )
        .with_finish(ProgressFinish::AndClear);
    progress.enable_steady_tick(Duration::from_millis(100));

    // Set up the I/O threads
    std::thread::scope(|s| {
        // Input thread fetches HDF5 data and sends it to the main thread
        let (hdf5_send, hdf5_recv) = mpsc::sync_channel(args.input_buffer.into());
        s.spawn(move || {
            for input in reader {
                let input = input.expect("Failed to load image");
                hdf5_send.send(input).expect("Main thread has crashed");
            }
        });

        // Output thread writes down image data, then sends images back to the
        // main thread for recycling.
        // TODO: Parallelize image export, put plurals in above comment
        let (image_send, image_recv) = mpsc::sync_channel::<RgbImage>(args.output_buffer.into());
        let (image_recycle_send, image_recycle_recv) = mpsc::channel();
        s.spawn(move || {
            for (idx, image) in image_recv.into_iter().enumerate() {
                image
                    .save(args.output_dir.join(format!("{idx}.png")))
                    .expect("Failed to save image");
                let _ = image_recycle_send.send(image);
                progress.inc(1);
            }
        });

        // Main thread converts HDF5 tables to images
        for input in hdf5_recv {
            // Try to reuse a previously created image
            let mut image = match image_recycle_recv.try_recv() {
                Ok(image) => image,
                Err(TryRecvError::Empty) => RgbImage::new(cols as u32, rows as u32),
                Err(TryRecvError::Disconnected) => panic!("Output thread has crashed"),
            };

            // Generate image using multiple processing threads
            rayon::iter::split(
                (input.view(), image_view(&mut image)),
                |(subinput, subimage)| {
                    let num_rows = subinput.nrows();
                    if num_rows > 1 {
                        let midpoint = num_rows / 2;
                        let (input1, input2) = subinput.split_at(Axis(0), midpoint);
                        let [image1, image2] = vsplit_image(subimage, midpoint);
                        ((input1, image1), Some((input2, image2)))
                    } else {
                        ((subinput, subimage), None)
                    }
                },
            )
            .for_each(|(subinput, mut subimage)| {
                for (value, pixel) in subinput.iter().zip(subimage.pixels_mut()) {
                    let color = gradient.eval_continuous((norm * value).into());
                    *pixel = Rgb([color.r, color.g, color.b]);
                }
            });

            // Send it to the output thread so it's written down
            image_send.send(image).expect("Output thread has crashed");
        }
    });
}

// Unfortunately, the image crate does not currently have view types and
// splitting operations, so we need to implement these ourselves...

// Mutable view of an image
type RgbImageView<'a> = ImageBuffer<Rgb<u8>, &'a mut [u8]>;

// Produce a mutable view of an image
fn image_view(image: &mut RgbImage) -> RgbImageView {
    let width = image.width();
    let height = image.height();
    let subpixels: &mut [u8] = image;
    RgbImageView::from_raw(width, height, subpixels).unwrap()
}

// Vertically split an image view
fn vsplit_image(image: RgbImageView, row: usize) -> [RgbImageView; 2] {
    let width = image.width();
    let subpixels_per_row = image.sample_layout().height_stride;
    let subpixels = image.into_raw();
    let (subpixels1, subpixels2) = subpixels.split_at_mut(subpixels_per_row * row);
    [subpixels1, subpixels2].map(|subpixels| {
        let height = (subpixels.len() / subpixels_per_row) as u32;
        RgbImageView::from_raw(width, height, subpixels).unwrap()
    })
}
