//! Image-based GPU concentration storage

use crate::{concentration::Concentration, Precision};
use ndarray::{s, ArrayView2, ArrayViewMut2};
use std::{
    collections::{hash_map::Entry, HashMap, HashSet},
    fmt::Debug,
    hash::Hash,
    ops::Range,
    sync::Arc,
};
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferError, BufferUsage, Subbuffer},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        CopyBufferToImageInfo, CopyError, CopyImageToBufferInfo, PrimaryAutoCommandBuffer,
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
}
//
impl Concentration for ImageConcentration {
    type Context = ImageContext;

    type Error = anyhow::Error;

    fn default(context: &mut ImageContext, shape: [usize; 2]) -> anyhow::Result<Self> {
        Self::new(context, shape, Buffer::new_slice::<Precision>)
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
        context.download(&self.gpu_image)?;
        {
            let mut buffer_contents = self.cpu_buffer.write()?;
            let slice = s![slice[0].clone(), slice[1].clone()];
            ArrayViewMut2::from_shape(self.shape(), &mut buffer_contents)
                .expect("The shape should be right")
                .slice_mut(slice)
                .fill(value);
        }
        context.touch_cpu(&self.cpu_buffer, &self.gpu_image);
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
        context.download(&self.gpu_image)?;
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
    ) -> anyhow::Result<Self> {
        let (gpu_image, cpu_buffer) =
            context.register_image(|allocator, queue_family_indices| {
                let num_texels = rows * cols;
                let texel_size = std::mem::size_of::<Precision>();
                assert_eq!(texel_size, 4, "Must adjust image format");
                let gpu_image = StorageImage::with_usage(
                    allocator,
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
                    queue_family_indices.iter().copied(),
                )?;
                let cpu_buffer = make_buffer(
                    allocator,
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
                Ok((gpu_image, cpu_buffer))
            })?;
        Ok(Self {
            gpu_image,
            cpu_buffer,
        })
    }

    /// Variation of new() that sets all buffer elements to the same value
    fn constant(
        context: &mut ImageContext,
        shape: [usize; 2],
        element: Precision,
    ) -> anyhow::Result<Self> {
        Self::new(
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
        )
    }

    /// Access the inner image for GPU work
    ///
    /// You must not cache this `Arc<StorageImage>` across simulation steps:
    ///
    /// - The actual image you are accessing will change between steps
    /// - CPU -> GPU uploads are performed lazily on image access, so it's
    ///   important for us to know when images are accessed.
    pub fn access_image(
        &mut self,
        context: &mut ImageContext,
    ) -> anyhow::Result<&Arc<StorageImage>> {
        context.upload(&self.cpu_buffer)?;
        context.touch_gpu(&self.gpu_image, &self.cpu_buffer);
        Ok(&self.gpu_image)
    }
}

/// External state needed to manipulate ImageConcentrations
pub struct ImageContext {
    /// Variable state
    state: ContextState,

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
            state: ContextState::Initialization {
                memory_allocator,
                queue_family_indices,
            },
            command_allocator,
            pending_uploads: HashMap::new(),
            upload_queue,
            download_queue,
        })
    }

    /// Register a new (image, buffer) pair
    fn register_image(
        &mut self,
        make_image: impl FnOnce(
            &MemAlloc,
            &HashSet<u32>,
        ) -> anyhow::Result<(Arc<StorageImage>, CpuBuffer)>,
    ) -> anyhow::Result<(Arc<StorageImage>, CpuBuffer)> {
        let ContextState::Initialization {
            memory_allocator,
            queue_family_indices,
        } = &mut self.state else {
            panic!("New images may not be registered after simulation has started!")
        };
        let (image, buffer) = make_image(&memory_allocator, &queue_family_indices)?;
        self.pending_uploads.insert(buffer.clone(), image.clone());
        Ok((image, buffer))
    }

    /// Signal that the CPU version of an image has been modified
    ///
    /// This does not trigger an immediate upload because this image could be
    /// modified further on the CPU side before we actually need it on the GPU.
    ///
    /// Instead, the upload is delayed until the moment where we actually need
    /// the image to be on the GPU side.
    fn touch_cpu(&mut self, cpu: &CpuBuffer, gpu: &Arc<StorageImage>) {
        Self::touch(&mut self.pending_uploads, cpu, gpu);
    }

    /// Signal that the GPU version of an image has been accessed
    ///
    /// We pessimistically treat accesses as modifications, and modifications
    /// on the GPU side are treated just like touch_cpu handles the CPU side.
    //
    // FIXME: Try to get away with specifying only the StorageImage and
    //        keeping a HashSet<StorageImage, CpuBuffer> around
    fn touch_gpu(&mut self, gpu: &Arc<StorageImage>, cpu: &CpuBuffer) {
        Self::touch(self.state.pending_downloads(), gpu, cpu);
        debug_assert!(
            !self.pending_uploads.contains_key(cpu),
            "Should not touch GPU data while a CPU upload is still pending!"
        );
    }

    /// Generic logic of the touch_xyz functions
    fn touch<Src: Clone + Eq + Hash, Dst: Clone + Debug + Eq>(
        map: &mut HashMap<Src, Dst>,
        src: &Src,
        dst: &Dst,
    ) {
        match map.entry(src.clone()) {
            Entry::Occupied(occupied) => debug_assert_eq!(occupied.get(), dst),
            Entry::Vacant(vacant) => {
                vacant.insert(dst.clone());
            }
        }
    }

    /// Update the GPU view of an image if the CPU side has changed
    fn upload(&mut self, cpu: &CpuBuffer) -> anyhow::Result<()> {
        Self::sync(
            cpu,
            &mut self.pending_uploads,
            &self.command_allocator,
            &self.upload_queue,
            |builder, src, dst| {
                builder.copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(src, dst))
            },
        )
    }

    /// Update the CPU view of an image if the GPU side has changed
    fn download(&mut self, gpu: &Arc<StorageImage>) -> anyhow::Result<()> {
        if self.state.gpu_state_exposed() {
            Self::sync(
                gpu,
                self.state.pending_downloads(),
                &self.command_allocator,
                &self.download_queue,
                |builder, src, dst| {
                    builder.copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(src, dst))
                },
            )
        } else {
            Ok(())
        }
    }

    /// Generic logic of the upload and download functions
    ///
    /// - `src` is the entity from which data is being transferred
    /// - `pending` records the list of similar data transfers that are pending
    /// - `allocator` is used to record command buffers
    /// - `queue` is where the command buffers will eventually be submitted
    /// - `record` tells how the underlying upload/download command is recorded
    //
    // FIXME: The current design will result in useless downloads in multi-step
    //        workflows, because we'll download to CPU data from GPU steps we
    //        do not care about. To fix this while retaining batched command
    //        submissions, we'll need a two-pass API: first collect needs, then
    //        start a batch download. Alternatively, we can just download
    //        images one by one for now, I think that's better.
    fn sync<Src: Eq + Hash, Dst>(
        src: &Src,
        pending: &mut HashMap<Src, Dst>,
        allocator: &CommAlloc,
        queue: &Queue,
        record: impl FnMut(
            &mut CommandBufferBuilder<CommAlloc>,
            Src,
            Dst,
        ) -> Result<&mut CommandBufferBuilder<CommAlloc>, CopyError>,
    ) -> anyhow::Result<()> {
        // Exit early if there is no need to update this particular image
        if !pending.contains_key(src) {
            return Ok(());
        }

        // Otherwise, update all images with pending changes
        let mut commands = CommandBufferBuilder::primary(
            allocator,
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )?;
        for (src, dst) in pending.drain() {
            record(&mut commands, src, dst)?;
        }
        let commands = commands.build()?;
        vulkano::sync::now(queue.device().clone())
            .then_execute_same_queue(commands)?
            .then_signal_fence_and_flush()?
            .wait(None)?;
        Ok(())
    }
}

/// Part of ImageContext that changes between initialization and operation
enum ContextState {
    /// Context is being initialized (new images are still being added)
    Initialization {
        /// Buffer/image allocator
        memory_allocator: Arc<MemAlloc>,

        /// Queue family indices of the upload and download queues
        queue_family_indices: HashSet<u32>,
    },

    /// Context is operational (all images have been added, ready for action)
    Operation {
        /// Modified GPU data that should eventually be downloaded to the CPU
        pending_downloads: HashMap<Arc<StorageImage>, CpuBuffer>,
    },
}
//
impl ContextState {
    /// Truth that GPU state has been exposed and there may be pending downloads
    fn gpu_state_exposed(&self) -> bool {
        if let ContextState::Operation { .. } = self {
            true
        } else {
            false
        }
    }

    /// Switch out of the initialization phase as needed, then access the
    /// pending downloads hashmap
    fn pending_downloads(&mut self) -> &mut HashMap<Arc<StorageImage>, CpuBuffer> {
        if let ContextState::Initialization { .. } = self {
            *self = ContextState::Operation {
                pending_downloads: HashMap::new(),
            }
        }
        let ContextState::Operation { pending_downloads } = self else {
            unreachable!()
        };
        pending_downloads
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
