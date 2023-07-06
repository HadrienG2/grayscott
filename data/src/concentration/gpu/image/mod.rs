//! Image-based GPU concentration storage

#![allow(clippy::result_large_err)]

pub mod context;

use self::context::{ImageContext, MemAlloc};
use crate::{
    concentration::{
        gpu::shape::{Shape, ShapeConvertError},
        AsScalars, Concentration,
    },
    Precision,
};
use ndarray::{s, ArrayView2, ArrayViewMut2};
use std::{ops::Range, sync::Arc};
use thiserror::Error;
use vulkano::{
    buffer::{
        subbuffer::BufferReadGuard, Buffer, BufferCreateInfo, BufferError, BufferUsage, Subbuffer,
    },
    command_buffer::{BuildError, CommandBufferBeginError, CommandBufferExecError, CopyError},
    device::DeviceOwned,
    format::{Format, FormatFeatures},
    image::{
        ImageAccess, ImageCreateFlags, ImageError, ImageFormatInfo, ImageTiling, ImageType,
        ImageUsage, StorageImage,
    },
    memory::allocator::{AllocationCreateInfo, MemoryAllocatePreference, MemoryUsage},
    sync::{future::NowFuture, FlushError, GpuFuture, Sharing},
    DeviceSize, OomError,
};

/// Image-based Concentration implementation
///
/// GPU images and samplers provide several useful features for our stencil
/// computation, that we would need to otherwise reimplement in shader code:
///
/// - Can index using 2D/3D coordinates, make array stacks
/// - Out-of-bound indexing returns a deterministic value
/// - Memory layout is optimized for 2D cache locality
///
/// The compromise we make by using them is that...
///
/// - We need a bit more host-side preparation than with buffer-based code
/// - Memory layout is an implementation detail that may vary from one device to
///   another, so further memory access optimizations are hard.
///
/// Overall, I think it's fair to say that images should provide a decent result
/// with minimal effort in device code, at the cost of making further
/// optimization harder. They are therefore a perfect fit for the naive code,
/// and best replaced by buffers in more advanced examples.
pub struct ImageConcentration {
    /// GPU version of the concentration
    gpu_image: Arc<StorageImage>,

    /// CPU version of the concentration
    cpu_buffer: CpuBuffer,

    /// Which device currently owns the data
    owner: Owner,
}
//
impl Concentration for ImageConcentration {
    type Context = ImageContext;

    type Error = Error;

    fn default(context: &mut ImageContext, shape: [usize; 2]) -> Result<Self> {
        Self::new(context, shape, Buffer::new_slice::<Precision>, Owner::Gpu)
    }

    fn zeros(context: &mut ImageContext, shape: [usize; 2]) -> Result<Self> {
        Self::constant(context, shape, 0.0)
    }

    fn ones(context: &mut ImageContext, shape: [usize; 2]) -> Result<Self> {
        Self::constant(context, shape, 1.0)
    }

    fn shape(&self) -> [usize; 2] {
        Shape::try_from(self.gpu_image.dimensions())
            .expect("Can't fail (checked at construction time)")
            .ndarray()
    }

    fn fill_slice(
        &mut self,
        context: &mut ImageContext,
        slice: [Range<usize>; 2],
        value: Precision,
    ) -> Result<()> {
        // If the image has already been used on GPU, download the new version
        if self.owner == Owner::Gpu {
            context.download_after(self.now(), self.gpu_image.clone(), self.cpu_buffer.clone())?;
            self.owner = Owner::Cpu;
        }

        // Perform the desired modification
        {
            let mut buffer_contents = self.cpu_buffer.write()?;
            let slice = s![slice[0].clone(), slice[1].clone()];
            ArrayViewMut2::from_shape(self.shape(), &mut buffer_contents)
                .expect("The shape should be right")
                .slice_mut(slice)
                .fill(value);
        }

        // Schedule an eventual upload to the GPU
        context.schedule_upload(&self.cpu_buffer, &self.gpu_image);
        Ok(())
    }

    fn finalize(&mut self, context: &mut Self::Context) -> Result<()> {
        context.upload_all()?;
        self.owner = Owner::Gpu;
        Ok(())
    }

    type ScalarView<'a> = ScalarView<'a>;

    fn make_scalar_view(&mut self, context: &mut ImageContext) -> Result<ScalarView> {
        self.make_scalar_view_after(self.now(), context)
    }

    fn write_scalar_view(
        &mut self,
        context: &mut ImageContext,
        target: ArrayViewMut2<Precision>,
    ) -> Result<()> {
        self.write_scalar_view_after(self.now(), context, target)
    }
}
//
impl ImageConcentration {
    /// Image format used by this concentration type
    pub fn format() -> Format {
        Self::image_format_info(ImageUsage::default())
            .format
            .unwrap()
    }

    /// Required image format features
    pub fn required_image_format_features() -> FormatFeatures {
        FormatFeatures::TRANSFER_SRC | FormatFeatures::TRANSFER_DST
    }

    /// Required buffer format features
    pub fn required_buffer_format_features() -> FormatFeatures {
        FormatFeatures::TRANSFER_SRC | FormatFeatures::TRANSFER_DST
    }

    /// Image configuration
    pub fn image_format_info(client_usage: ImageUsage) -> ImageFormatInfo {
        assert_eq!(
            std::mem::size_of::<Precision>(),
            4,
            "Must adjust image format"
        );
        ImageFormatInfo {
            flags: ImageCreateFlags::empty(),
            format: Some(Format::R32_SFLOAT),
            image_type: ImageType::Dim2d,
            tiling: ImageTiling::Optimal,
            usage: ImageUsage::TRANSFER_SRC | ImageUsage::TRANSFER_DST | client_usage,
            ..Default::default()
        }
    }

    /// Access the inner image for GPU work
    ///
    /// You must not cache this `Arc<StorageImage>` across simulation steps, as
    /// the actual image you are reading/writing will change between steps.
    pub fn access_image(&self) -> &Arc<StorageImage> {
        assert_eq!(
            self.owner,
            Owner::Gpu,
            "Some CPU-side data hasn't been uploaded yet. \
            Did you call finalize() after your last CPU-side modification?"
        );
        &self.gpu_image
    }

    /// Version of `Concentration::make_scalar_view()` that can take place right
    /// after simulation in a single GPU submission
    pub fn make_scalar_view_after(
        &mut self,
        after: impl GpuFuture + 'static,
        context: &mut ImageContext,
    ) -> Result<ScalarView> {
        context.download_after(after, self.gpu_image.clone(), self.cpu_buffer.clone())?;
        Ok(ScalarView::new(self.cpu_buffer.read()?, |buffer| {
            ArrayView2::from_shape(self.shape(), buffer).expect("The shape should be right")
        }))
    }

    /// Version of `Concentration::write_scalar_view()` that can take place right
    /// after simulation in a single GPU submission
    pub fn write_scalar_view_after(
        &mut self,
        after: impl GpuFuture + 'static,
        context: &mut ImageContext,
        mut target: ArrayViewMut2<Precision>,
    ) -> Result<()> {
        Self::validate_write(&self, &target);
        let view = self.make_scalar_view_after(after, context)?;
        target.assign(&view.as_scalars());
        Ok(())
    }

    /// Variation of new() that sets all buffer elements to the same value
    fn constant(context: &mut ImageContext, shape: [usize; 2], element: Precision) -> Result<Self> {
        let result = Self::new(
            context,
            shape,
            |allocator, buffer_info, allocation_info, len| {
                Buffer::from_iter(
                    allocator,
                    buffer_info,
                    allocation_info,
                    (0..len.try_into().unwrap()).map(|_| element),
                )
            },
            Owner::Cpu,
        )?;
        context.schedule_upload(&result.cpu_buffer, &result.gpu_image);
        Ok(result)
    }

    /// Set up an image of a certain shape, with certain initial contents
    ///
    /// The first 3 parameters of `make_buffer` have the same semantics as those
    /// of a Buffer constructor. The last parameter indicates how many elements
    /// the buffer will contain.
    fn new(
        context: &mut ImageContext,
        shape: [usize; 2],
        buffer_constructor: impl FnOnce(
            &MemAlloc,
            BufferCreateInfo,
            AllocationCreateInfo,
            DeviceSize,
        ) -> std::result::Result<CpuBuffer, BufferError>,
        owner: Owner,
    ) -> Result<Self> {
        let shape = Shape::try_from(shape)?;
        let gpu_image = Self::make_image(context, shape)?;
        let cpu_buffer = Self::make_buffer(context, shape, buffer_constructor)?;

        if cfg!(feature = "gpu-debug-utils") {
            let device = context.device();
            device
                .set_debug_utils_object_name(gpu_image.inner().image, Some("GPU concentration"))?;
            device.set_debug_utils_object_name(cpu_buffer.buffer(), Some("CPU concentration"))?;
        }

        Ok(Self {
            gpu_image,
            cpu_buffer,
            owner,
        })
    }

    /// Image construction logic
    fn make_image(context: &mut ImageContext, shape: Shape) -> Result<Arc<StorageImage>> {
        let image_format_info = Self::image_format_info(context.client_image_usage());
        let image = StorageImage::with_usage(
            context.memory_allocator(),
            shape.image(),
            Self::format(),
            image_format_info.usage,
            image_format_info.flags,
            context.gpu_queue_family_indices(),
        )?;
        Ok(image)
    }

    /// Buffer construction logic
    fn make_buffer(
        context: &mut ImageContext,
        shape: Shape,
        constructor: impl FnOnce(
            &MemAlloc,
            BufferCreateInfo,
            AllocationCreateInfo,
            DeviceSize,
        ) -> std::result::Result<CpuBuffer, BufferError>,
    ) -> Result<CpuBuffer> {
        let buffer = constructor(
            context.memory_allocator(),
            BufferCreateInfo {
                sharing: if context.cpu_queue_family_indices().count() > 1 {
                    Sharing::Concurrent(context.cpu_queue_family_indices().collect())
                } else {
                    Sharing::Exclusive
                },
                usage: BufferUsage::TRANSFER_SRC | BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                usage: MemoryUsage::Download,
                allocate_preference: MemoryAllocatePreference::Unknown,
                ..Default::default()
            },
            shape
                .buffer_size()
                .expect("Shouldn't happen if image allocation succeded"),
        )?;
        Ok(buffer)
    }

    /// Shortcut to `vulkano::sync::now()` on our device
    fn now(&self) -> NowFuture {
        vulkano::sync::now(self.gpu_image.device().clone())
    }
}

self_cell::self_cell!(
    /// Scalar CPU-side view of an ImageConcentration
    pub struct ScalarView<'concentration> {
        owner: BufferReadGuard<'concentration, [Precision]>,

        #[not_covariant]
        dependent: Dependent,
    }
);
//
type Dependent<'owner> = ArrayView2<'owner, Precision>;
//
impl AsScalars for ScalarView<'_> {
    fn as_scalars(&self) -> ArrayView2<Precision> {
        self.with_dependent(|_owner, dependent| dependent.reborrow())
    }
}

/// Who currently owns the data of an ImageConcentration
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Owner {
    Cpu,
    Gpu,
}

/// Errors that can occur while using images
#[derive(Clone, Debug, Error)]
pub enum Error {
    #[error("failed to start recording a command buffer")]
    CommandBufferBegin(#[from] CommandBufferBeginError),

    #[error("failed to build a command buffer")]
    CommandBufferBuild(#[from] BuildError),

    #[error("failed to execute a command buffer")]
    CommandBufferExec(#[from] CommandBufferExecError),

    #[error("failed to record a copy command")]
    Copy(#[from] CopyError),

    #[error("failed to create or manipulate a data buffer")]
    Buffer(#[from] BufferError),

    #[error("failed to submit commands to the GPU")]
    Flush(#[from] FlushError),

    #[error("failed to create or manipulate an image")]
    Image(#[from] ImageError),

    #[error("ran out of memory")]
    OutOfMemory(#[from] OomError),

    #[error("bad domain shape")]
    BadShape(#[from] ShapeConvertError),
}
//
pub type Result<T> = std::result::Result<T, Error>;

/// CPU-side data
type CpuBuffer = Subbuffer<[Precision]>;
