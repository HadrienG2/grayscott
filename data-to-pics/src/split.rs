//! Image splitting operations
//!
//! Unfortunately, the image crate does not currently have view types and
//! splitting operations, so we need to implement these ourselves...

use image::{ImageBuffer, Rgb, RgbImage};

// Mutable view of an image
pub type RgbImageView<'a> = ImageBuffer<Rgb<u8>, &'a mut [u8]>;

// Produce a mutable view of an image
pub fn image_view(image: &mut RgbImage) -> RgbImageView {
    let width = image.width();
    let height = image.height();
    let subpixels: &mut [u8] = image;
    RgbImageView::from_raw(width, height, subpixels).expect("Should never fail")
}

// Vertically split an image view
pub fn vsplit_image(image: RgbImageView, row: usize) -> [RgbImageView; 2] {
    let width = image.width();
    let subpixels_per_row = image.sample_layout().height_stride;
    let subpixels = image.into_raw();
    let (subpixels1, subpixels2) = subpixels.split_at_mut(subpixels_per_row * row);
    [subpixels1, subpixels2].map(|subpixels| {
        let height = (subpixels.len() / subpixels_per_row) as u32;
        RgbImageView::from_raw(width, height, subpixels).expect("Should never fail")
    })
}
