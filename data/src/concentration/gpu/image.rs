//! Image-based GPU concentration storage

use crate::{concentration::Concentration, Precision};
use ndarray::{s, ArrayView2, ArrayViewMut2};
use std::{
    collections::{HashMap, HashSet},
    ops::Range,
    sync::Arc,
};
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferError, BufferUsage, Subbuffer},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        CopyBufferToImageInfo, CopyImageToBufferInfo, PrimaryAutoCommandBuffer,
    },
    device::Queue,
    format::Format,
    image::{ImageAccess, ImageCreateFlags, ImageDimensions, ImageUsage, StorageImage},
    memory::allocator::{
        AllocationCreateInfo, MemoryAllocatePreference, MemoryUsage, StandardMemoryAllocator,
    },
    sync::GpuFuture,
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

    type Error = anyhow::Error;

    fn default(context: &mut ImageContext, shape: [usize; 2]) -> anyhow::Result<Self> {
        Self::new(context, shape, Buffer::new_slice::<Precision>, Owner::Gpu)
    }

    fn zeros(context: &mut ImageContext, shape: [usize; 2]) -> anyhow::Result<Self> {
        Self::constant(context, shape, 0.0)
    }

    fn ones(context: &mut ImageContext, shape: [usize; 2]) -> anyhow::Result<Self> {
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
    ) -> anyhow::Result<()> {
        // If the image has already been used on GPU, download the new version
        if self.owner == Owner::Gpu {
            context.download(self.gpu_image.clone(), self.cpu_buffer.clone())?;
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

    fn finalize(&mut self, context: &mut Self::Context) -> Result<(), Self::Error> {
        context.upload_all()?;
        self.owner = Owner::Gpu;
        Ok(())
    }

    // FIXME: This cannot literally be a ScalarConcentrationView, at best it can
    //        be an AsRef<ScalarConcentrationView>. Fix the trait definition.
    //        Also, I'll likely need self-ref types, try ouroboros.
    type ScalarView<'a> = ArrayView2<'a, Precision>;

    fn make_scalar_view(
        &mut self,
        context: &mut ImageContext,
    ) -> anyhow::Result<Self::ScalarView<'_>> {
        context.download(self.gpu_image.clone(), self.cpu_buffer.clone())?;
        let buffer_contents = self.cpu_buffer.read()?;
        Ok(ArrayView2::from_shape(self.shape(), &mut buffer_contents)
            .expect("The shape should be right"))
    }
}
//
impl ImageConcentration {
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
        ) -> Result<CpuBuffer, BufferError>,
        owner: Owner,
    ) -> anyhow::Result<Self> {
        let num_texels = rows * cols;
        let texel_size = std::mem::size_of::<Precision>();
        assert_eq!(texel_size, 4, "Must adjust image format");
        let gpu_image = StorageImage::with_usage(
            context.memory_allocator(),
            ImageDimensions::Dim2d {
                width: cols.try_into().unwrap(),
                height: rows.try_into().unwrap(),
                array_layers: 1,
            },
            Format::R32_SFLOAT,
            ImageUsage::TRANSFER_SRC
                | ImageUsage::TRANSFER_DST
                | ImageUsage::SAMPLED
                | ImageUsage::STORAGE,
            ImageCreateFlags::empty(),
            context.queue_family_indices(),
        )?;
        let cpu_buffer = make_buffer(
            context.memory_allocator(),
            BufferCreateInfo {
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
    fn constant(
        context: &mut ImageContext,
        shape: [usize; 2],
        element: Precision,
    ) -> anyhow::Result<Self> {
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
        &self.gpu_image
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
    /// Buffer/image allocator
    memory_allocator: Arc<MemAlloc>,

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
    pub fn new(
        memory_allocator: Arc<MemAlloc>,
        command_allocator: Arc<CommAlloc>,
        upload_queue: Arc<Queue>,
        download_queue: Arc<Queue>,
    ) -> anyhow::Result<Self> {
        let queue_family_indices = [
            upload_queue.queue_family_index(),
            download_queue.queue_family_index(),
        ]
        .into_iter()
        .collect();

        Ok(Self {
            memory_allocator,
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

    /// Query which queue family indices images may be used with
    fn queue_family_indices(&self) -> impl IntoIterator<Item = u32> + '_ {
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
    fn upload_all(&mut self) -> anyhow::Result<()> {
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
            .then_execute_same_queue(commands)?
            .then_signal_fence_and_flush()?
            .wait(None)?;
        Ok(())
    }

    /// Update the CPU view of an image
    ///
    /// Unlike uploads, downloads are eager because we usually only need a
    /// single image on the CPU, so batching is a pessimization. Furthermore, it
    /// is hard to track when GPU images are modified, or even accessed.
    fn download(&mut self, gpu: Arc<StorageImage>, cpu: CpuBuffer) -> anyhow::Result<()> {
        assert!(
            !self.pending_uploads.contains_key(&cpu),
            "Attempting to download a stale image from the GPU. \
            Did you call finalize() after your last CPU-side modification?"
        );
        let mut builder = CommandBufferBuilder::primary(
            &self.command_allocator,
            self.download_queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )?;
        builder.copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(gpu, cpu))?;
        let commands = builder.build()?;
        vulkano::sync::now(self.download_queue.device().clone())
            .then_execute_same_queue(commands)?
            .then_signal_fence_and_flush()?
            .wait(None)?;
        Ok(())
    }
}

/// Memory allocator (hardcoded for now)
type MemAlloc = StandardMemoryAllocator;

/// CPU-side data
type CpuBuffer = Subbuffer<[Precision]>;

/// Command buffer allocator (hardcoded for now)
type CommAlloc = StandardCommandBufferAllocator;

/// Command buffer builder
type CommandBufferBuilder<CommAlloc> =
    AutoCommandBufferBuilder<PrimaryAutoCommandBuffer, CommAlloc>;

/// Command buffer
type CommandBuffer<CommAlloc> = PrimaryAutoCommandBuffer<CommAlloc>;
