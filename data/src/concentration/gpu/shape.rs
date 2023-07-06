//! Shape-related calculations

use crate::Precision;
use thiserror::Error;
use vulkano::{image::ImageDimensions, DeviceSize};

/// Domain shape converter
///
/// In GPU backends, we need a lot of quantities that are closely related to the
/// shape of the simulation domain in various ways, and it is tedious and
/// error-prone to convert between them all the time. This struct centralizes
/// all the conversions we need.
#[derive(Copy, Clone, Debug, Default, Eq, Hash, PartialEq)]
pub struct Shape([u32; 2]);
//
impl Shape {
    /// Construct from the [rows, cols] representation used elsewhere in `data`
    #[inline]
    pub const fn new([rows, cols]: [u32; 2]) -> Self {
        Self([cols, rows])
    }

    /// Construct from the GPU-favored [width, height] representation
    #[inline]
    pub const fn from_width_height([width, height]: [u32; 2]) -> Self {
        Self([width, height])
    }

    /// Image width
    #[inline]
    pub const fn width(&self) -> u32 {
        self.0[0]
    }

    /// Image height
    #[inline]
    pub const fn height(&self) -> u32 {
        self.0[1]
    }

    /// Number of rows
    #[inline]
    pub const fn rows(&self) -> u32 {
        self.height()
    }

    /// Number of columns
    #[inline]
    pub const fn cols(&self) -> u32 {
        self.width()
    }

    /// Domain shape in [width, height, depth] layout
    #[inline]
    pub const fn vulkan(&self) -> [u32; 3] {
        [self.width(), self.height(), 1]
    }

    /// Domain shape in [rows, cols] layout
    #[inline]
    pub const fn ndarray(&self) -> [usize; 2] {
        [self.rows() as usize, self.cols() as usize]
    }

    /// Size of Vulkan images dimensioned after the domain shape
    #[inline]
    pub const fn image(&self) -> ImageDimensions {
        ImageDimensions::Dim2d {
            width: self.width(),
            height: self.height(),
            array_layers: 1,
        }
    }

    /// Length of Vulkan buffers dimensioned after the domain shape
    #[inline]
    pub fn buffer_len(&self) -> Result<DeviceSize, ShapeConvertError> {
        (self.width() as DeviceSize)
            .checked_mul(self.height() as DeviceSize)
            .ok_or(ShapeConvertError::TooLarge)
    }

    /// Size of Vulkan buffers dimensioned after the domain shape
    #[inline]
    pub fn buffer_size(&self) -> Result<DeviceSize, ShapeConvertError> {
        self.buffer_len()?
            .checked_mul(std::mem::size_of::<Precision>() as DeviceSize)
            .ok_or(ShapeConvertError::TooLarge)
    }

    /// Number of shader invocations if each one processes one pixel
    #[inline]
    pub fn invocations(&self) -> Result<u32, ShapeConvertError> {
        self.width()
            .checked_mul(self.height())
            .ok_or(ShapeConvertError::TooLarge)
    }
}
//
impl From<[u32; 2]> for Shape {
    #[inline]
    fn from(shape: [u32; 2]) -> Self {
        Self::new(shape)
    }
}
//
impl TryFrom<[u32; 3]> for Shape {
    type Error = ShapeConvertError;

    #[inline]
    fn try_from([width, height, depth]: [u32; 3]) -> Result<Self, Self::Error> {
        if depth == 1 {
            Ok(Self::from_width_height([width, height]))
        } else {
            Err(ShapeConvertError::HasDepth)
        }
    }
}
//
impl TryFrom<[usize; 2]> for Shape {
    type Error = ShapeConvertError;

    #[inline]
    fn try_from([rows, cols]: [usize; 2]) -> Result<Self, Self::Error> {
        let convert = |us| u32::try_from(us).map_err(|_| ShapeConvertError::TooLarge);
        Ok(Self::new([convert(rows)?, convert(cols)?]))
    }
}
//
impl TryFrom<ImageDimensions> for Shape {
    type Error = ShapeConvertError;

    #[inline]
    fn try_from(value: ImageDimensions) -> Result<Self, Self::Error> {
        if let ImageDimensions::Dim2d {
            width,
            height,
            array_layers: 1,
        } = value
        {
            Ok(Self::from_width_height([width, height]))
        } else {
            Err(ShapeConvertError::BadImage)
        }
    }
}
//
/// Errors emitted by invalid shape conversions
#[derive(Copy, Clone, Debug, Eq, Error, Hash, PartialEq)]
pub enum ShapeConvertError {
    #[error("depth is not unity")]
    HasDepth,

    #[error("too large for GPU")]
    TooLarge,

    #[error("not a single 2D image")]
    BadImage,
}

/// GPU dispatch size assuming one shader invocation per domain element
#[inline]
pub fn full_dispatch_size(
    domain_shape: Shape,
    work_group_shape: Shape,
) -> Result<[u32; 3], PartialWorkGroupError> {
    let shape = domain_shape;
    let group = work_group_shape;
    let coord = |shape, group| {
        if shape % group == 0 {
            Ok(shape / group)
        } else {
            Err(PartialWorkGroupError)
        }
    };
    Ok([
        coord(shape.width(), group.width())?,
        coord(shape.height(), group.height())?,
        1,
    ])
}
//
/// Error emitted when the domain shape is not a multiple of the work group shape
#[derive(Copy, Clone, Debug, Default, Eq, Error, Hash, PartialEq)]
#[error("domain size is not a multiple of work group size")]
pub struct PartialWorkGroupError;
