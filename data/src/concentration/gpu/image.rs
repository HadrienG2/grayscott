//! Image-based GPU concentration storage

use crate::{
    concentration::{AsScalars, Concentration},
    Precision,
};
use ndarray::{s, ArrayView2, ArrayViewMut2};
use std::{
    collections::{HashMap, HashSet},
    ops::Range,
    sync::Arc,
};
use thiserror::Error;
use vulkano::{
    buffer::{
        subbuffer::BufferReadGuard, Buffer, BufferCreateInfo, BufferError, BufferUsage, Subbuffer,
    },
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, BuildError,
        CommandBufferBeginError, CommandBufferExecError, CommandBufferUsage, CopyBufferToImageInfo,
        CopyError, CopyImageToBufferInfo, PrimaryAutoCommandBuffer,
    },
    descriptor_set::PersistentDescriptorSet,
    device::{DeviceOwned, Queue},
    format::{Format, FormatFeatures},
    image::{
        ImageAccess, ImageCreateFlags, ImageDimensions, ImageError, ImageFormatInfo, ImageTiling,
        ImageType, ImageUsage, StorageImage,
    },
    memory::allocator::{
        AllocationCreateInfo, MemoryAllocatePreference, MemoryUsage, StandardMemoryAllocator,
    },
    sync::{future::NowFuture, FlushError, GpuFuture, Sharing},
    DeviceSize,
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
        let ImageDimensions::Dim2d { width, height, .. } = self.gpu_image.dimensions() else {
             unreachable!()
        };
        [height.try_into().unwrap(), width.try_into().unwrap()]
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
        ImageFormatInfo {
            flags: ImageCreateFlags::empty(),
            format: Some(Format::R32_SFLOAT),
            image_type: ImageType::Dim2d,
            tiling: ImageTiling::Optimal,
            usage: ImageUsage::TRANSFER_SRC | ImageUsage::TRANSFER_DST | client_usage,
            ..Default::default()
        }
    }

    /// Set up an image of a certain shape, with certain initial contents
    ///
    /// The first 3 parameters of `make_buffer` have the same semantics as those
    /// of a Buffer constructor. The last parameter indicates how many elements
    /// the buffer will contain.
    fn new(
        context: &mut ImageContext,
        [rows, cols]: [usize; 2],
        make_buffer: impl FnOnce(
            &MemAlloc,
            BufferCreateInfo,
            AllocationCreateInfo,
            DeviceSize,
        ) -> std::result::Result<CpuBuffer, BufferError>,
        owner: Owner,
    ) -> Result<Self> {
        let num_texels = rows * cols;
        let texel_size = std::mem::size_of::<Precision>();
        assert_eq!(texel_size, 4, "Must adjust image format");

        let image_format_info = Self::image_format_info(context.client_image_usage);
        let gpu_image = StorageImage::with_usage(
            context.memory_allocator(),
            ImageDimensions::Dim2d {
                width: cols.try_into().unwrap(),
                height: rows.try_into().unwrap(),
                array_layers: 1,
            },
            Self::format(),
            image_format_info.usage,
            image_format_info.flags,
            context.queue_family_indices(),
        )?;

        let cpu_buffer = make_buffer(
            context.memory_allocator(),
            BufferCreateInfo {
                sharing: if context.queue_family_indices().count() > 1 {
                    Sharing::Concurrent(context.queue_family_indices().collect())
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
            num_texels.try_into().unwrap(),
        )?;

        Ok(Self {
            gpu_image,
            cpu_buffer,
            owner,
        })
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

    /// Shortcut to `vulkano::sync::now()` on our device
    fn now(&self) -> NowFuture {
        vulkano::sync::now(self.gpu_image.device().clone())
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
            ArrayView2::from_shape(self.shape(), &buffer).expect("The shape should be right")
        }))
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

/// External state needed to manipulate ImageConcentrations
pub struct ImageContext {
    /// Descriptor set cache
    ///
    /// This member is not used by ImageContext methods and purely provided for
    /// the benefit of the compute backend using these ImageConcentrations.
    ///
    /// Vulkan implementors recommend caching the DescriptorSets that are used
    /// to attach resources to shaders. In our case, one particular descriptor
    /// set to cache is the one used to bind quadruplets of images (input and
    /// output U and V) to compute shaders.
    ///
    /// We do not handle the creation of that descriptor set as the specifics
    /// depend on how the shader accesses the image (sampled or not, etc), which
    /// is where our line for compute backend specific code is drawn.
    pub descriptor_sets: HashMap<[Arc<StorageImage>; 4], Arc<PersistentDescriptorSet>>,

    /// Buffer/image allocator
    memory_allocator: Arc<MemAlloc>,

    /// Client-requested image usage
    client_image_usage: ImageUsage,

    /// Queue family indices of the upload and download queues
    queue_family_indices: HashSet<u32>,

    /// Command buffer allocator
    command_allocator: Arc<CommAlloc>,

    /// Modified CPU data that should eventually be uploaded to the GPU
    pending_uploads: HashMap<CpuBuffer, Arc<StorageImage>>,

    /// Queue to be used when GPU data is uploaded from the CPU
    upload_queue: Arc<Queue>,

    /// Queue to be used when GPU data is downloaded to the CPU
    download_queue: Arc<Queue>,
}
//
impl ImageContext {
    /// Prepare to set up image-based concentration
    pub fn new<'other_queues>(
        memory_allocator: Arc<MemAlloc>,
        command_allocator: Arc<CommAlloc>,
        upload_queue: Arc<Queue>,
        download_queue: Arc<Queue>,
        other_queues: impl IntoIterator<Item = &'other_queues Arc<Queue>>,
        client_image_usage: ImageUsage,
    ) -> Result<Self> {
        let queue_family_indices = [
            upload_queue.queue_family_index(),
            download_queue.queue_family_index(),
        ]
        .into_iter()
        .chain(
            other_queues
                .into_iter()
                .map(|queue| queue.queue_family_index()),
        )
        .collect();

        Ok(Self {
            descriptor_sets: HashMap::new(),
            memory_allocator,
            client_image_usage,
            queue_family_indices,
            command_allocator,
            pending_uploads: HashMap::new(),
            upload_queue,
            download_queue,
        })
    }

    /// Get access to the memory allocator
    fn memory_allocator(&self) -> &MemAlloc {
        &self.memory_allocator
    }

    /// Query which queue family indices images and buffers may be used with
    fn queue_family_indices(&self) -> impl Iterator<Item = u32> + '_ {
        self.queue_family_indices.iter().copied()
    }

    /// Signal that the CPU version of an image has been modified
    ///
    /// This does not trigger an immediate upload because this image could be
    /// modified further on the CPU side before we actually need it on the GPU,
    /// and other images may also be modified.
    ///
    /// Instead, the upload is delayed until the moment where we actually need
    /// the image to be on the GPU side, at which point upload_all is called.
    fn schedule_upload(&mut self, cpu: &CpuBuffer, gpu: &Arc<StorageImage>) {
        if let Some(target) = self.pending_uploads.get(cpu) {
            debug_assert_eq!(target, gpu);
        } else {
            self.pending_uploads.insert(cpu.clone(), gpu.clone());
        }
    }

    /// Update the GPU view of all modified image
    fn upload_all(&mut self) -> Result<()> {
        if !self.pending_uploads.is_empty() {
            let mut builder = CommandBufferBuilder::primary(
                &self.command_allocator,
                self.upload_queue.queue_family_index(),
                CommandBufferUsage::OneTimeSubmit,
            )?;
            for (src, dst) in self.pending_uploads.drain() {
                builder.copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(src, dst))?;
            }
            let commands = builder.build()?;
            vulkano::sync::now(self.upload_queue.device().clone())
                .then_execute(self.upload_queue.clone(), commands)?
                .then_signal_fence_and_flush()?
                .wait(None)?;
        }
        Ok(())
    }

    /// Update the CPU view of an image
    ///
    /// Unlike uploads, downloads are eager because we usually only need a
    /// single image on the CPU, so batching is a pessimization. Furthermore, it
    /// is hard to track when GPU images are modified, or even accessed.
    fn download_after(
        &mut self,
        after: impl GpuFuture + 'static,
        gpu: Arc<StorageImage>,
        cpu: CpuBuffer,
    ) -> Result<()> {
        assert!(
            !self.pending_uploads.contains_key(&cpu),
            "Attempting to download a stale image from the GPU. \
            Did you call finalize() after your last CPU-side modification?"
        );

        // Need a semaphore if the input future does not map to the same queue
        let after_queue = after.queue().unwrap_or(self.download_queue.clone());
        let after = if after_queue != self.download_queue {
            after.then_signal_semaphore_and_flush()?.boxed()
        } else {
            after.boxed()
        };

        let mut builder = CommandBufferBuilder::primary(
            &self.command_allocator,
            self.download_queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )?;
        builder.copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(gpu, cpu))?;
        let commands = builder.build()?;

        after
            .then_execute(self.download_queue.clone(), commands)?
            .then_signal_fence_and_flush()?
            .wait(None)?;

        Ok(())
    }
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
}
//
pub type Result<T> = std::result::Result<T, Error>;

/// Memory allocator (hardcoded for now)
type MemAlloc = StandardMemoryAllocator;

/// CPU-side data
type CpuBuffer = Subbuffer<[Precision]>;

/// Command buffer allocator (hardcoded for now)
type CommAlloc = StandardCommandBufferAllocator;

/// Command buffer builder
type CommandBufferBuilder<CommAlloc> =
    AutoCommandBufferBuilder<PrimaryAutoCommandBuffer, CommAlloc>;
