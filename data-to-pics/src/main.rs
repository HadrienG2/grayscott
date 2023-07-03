use anyhow::Result;
use clap::Parser;
use data::hdf5::{Config, Reader};
use image::{ImageBuffer, Rgb, RgbImage};
use ndarray::Axis;
use rayon::prelude::*;
use std::{
    num::NonZeroUsize,
    path::PathBuf,
    sync::mpsc::{self, TryRecvError},
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

    /// Number of image I/O threads
    ///
    /// Adding more output threads can improve performance when all of the
    /// following is true:
    ///
    /// - The program is not processing images faster than HDF5 can read them.
    /// - The benefits of having more CPU time for PNG compression are not
    ///   compensated by the cost of oversubscribing the CPU, which slows
    ///   image rendering down.
    /// - Output storage has enough capacity to accomodate the extra workload
    ///   (seeks, writes) associated with writing more images in parallel.
    #[arg(long, default_value_t = NonZeroUsize::new(3).unwrap())]
    output_threads: NonZeroUsize,
}

fn main() -> Result<()> {
    // Enable logging to syslog
    ui::init_syslog();

    // Parse CLI arguments
    let args = Args::parse();

    // Open the HDF5 dataset
    let reader = Reader::open(Config {
        file_name: ui::simulation_output_path(args.input),
        ..Default::default()
    })?;
    let [rows, cols] = reader.image_shape();

    // Set up progress reporting
    let progress = ui::init_progress_reporting("Generating image", reader.num_images());

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

        // Output threads write down image data, then send image containers
        // back to the main thread for recycling
        let (image_send, image_recycle_recv) = {
            let (image_send, image_recv) =
                crossbeam_channel::bounded::<(usize, RgbImage)>(args.output_buffer.into());
            let (image_recycle_send, image_recycle_recv) = mpsc::channel();
            let output_dir = &args.output_dir;
            let progress = &progress;
            for _ in 0..args.output_threads.into() {
                let image_recv = image_recv.clone();
                let image_recycle_send = image_recycle_send.clone();
                s.spawn(move || {
                    for (idx, image) in image_recv {
                        image
                            .save(output_dir.join(format!("{idx}.png")))
                            .expect("Failed to save image");
                        let _ = image_recycle_send.send(image);
                        progress.inc(1);
                    }
                });
            }
            (image_send, image_recycle_recv)
        };

        // Main thread converts HDF5 tables to images
        for (idx, input) in hdf5_recv.into_iter().enumerate() {
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
                    let color = ui::GRADIENT.eval_continuous((ui::AMPLITUDE_SCALE * value).into());
                    *pixel = Rgb([color.r, color.g, color.b]);
                }
            });

            // Send it to the output thread so it's written down
            image_send.send((idx, image))?;
        }
        Ok::<_, anyhow::Error>(())
    })
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
    RgbImageView::from_raw(width, height, subpixels).expect("Should never fail")
}

// Vertically split an image view
fn vsplit_image(image: RgbImageView, row: usize) -> [RgbImageView; 2] {
    let width = image.width();
    let subpixels_per_row = image.sample_layout().height_stride;
    let subpixels = image.into_raw();
    let (subpixels1, subpixels2) = subpixels.split_at_mut(subpixels_per_row * row);
    [subpixels1, subpixels2].map(|subpixels| {
        let height = (subpixels.len() / subpixels_per_row) as u32;
        RgbImageView::from_raw(width, height, subpixels).expect("Should never fail")
    })
}
