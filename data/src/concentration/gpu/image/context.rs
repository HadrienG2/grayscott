//! Simulation-wide context for image-based concentrations

use super::{CpuBuffer, Result};
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};
use vulkano::{
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        CopyBufferToImageInfo, CopyImageToBufferInfo, PrimaryAutoCommandBuffer,
    },
    descriptor_set::PersistentDescriptorSet,
    device::{Device, DeviceOwned, Queue},
    image::{ImageUsage, StorageImage},
    memory::allocator::StandardMemoryAllocator,
    sync::GpuFuture,
};

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

    /// Queue family indices that CPU-side resources will be used with
    cpu_queue_family_indices: HashSet<u32>,

    /// Queue family indices that GPU-side resources will be used with
    gpu_queue_family_indices: HashSet<u32>,

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
        other_queues: impl IntoIterator<Item = &'other_queues Arc<Queue>> + Clone,
        client_image_usage: ImageUsage,
    ) -> Result<Self> {
        // Check for configuration consistency
        assert!(
            [
                command_allocator.device().clone(),
                upload_queue.device().clone(),
                download_queue.device().clone()
            ]
            .into_iter()
            .chain(other_queues.clone().into_iter().map(|q| q.device().clone()))
            .all(|other| other == *memory_allocator.device()),
            "All specified entities should map to the same Vulkan device"
        );

        // Collect the list of all queue family indices that CPU- and GPU-side
        // ressources will be used with
        let cpu_queue_family_indices = [
            upload_queue.queue_family_index(),
            download_queue.queue_family_index(),
        ]
        .into_iter()
        .collect::<HashSet<_>>();
        let gpu_queue_family_indices = cpu_queue_family_indices
            .iter()
            .copied()
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
            cpu_queue_family_indices,
            gpu_queue_family_indices,
            command_allocator,
            pending_uploads: HashMap::new(),
            upload_queue,
            download_queue,
        })
    }

    /// Get access to the underlying device
    pub(crate) fn device(&self) -> &Arc<Device> {
        self.download_queue.device()
    }

    /// Get access to the memory allocator
    pub(crate) fn memory_allocator(&self) -> &MemAlloc {
        &self.memory_allocator
    }

    /// Query how the end user will use the allocated images
    pub(crate) fn client_image_usage(&self) -> ImageUsage {
        self.client_image_usage
    }

    /// Query which queue family indices CPU-side resources may be used with
    pub(crate) fn cpu_queue_family_indices(&self) -> impl Iterator<Item = u32> + '_ {
        self.cpu_queue_family_indices.iter().copied()
    }

    /// Query which queue family indices GPU-side resources may be used with
    pub(crate) fn gpu_queue_family_indices(&self) -> impl Iterator<Item = u32> + '_ {
        self.gpu_queue_family_indices.iter().copied()
    }

    /// Signal that the CPU version of an image has been modified
    ///
    /// This does not trigger an immediate upload because this image could be
    /// modified further on the CPU side before we actually need it on the GPU,
    /// and other images may also be modified.
    ///
    /// Instead, the upload is delayed until the moment where we actually need
    /// the image to be on the GPU side, at which point upload_all is called.
    pub(crate) fn schedule_upload(&mut self, cpu: &CpuBuffer, gpu: &Arc<StorageImage>) {
        if let Some(target) = self.pending_uploads.get(cpu) {
            debug_assert_eq!(target, gpu);
        } else {
            self.pending_uploads.insert(cpu.clone(), gpu.clone());
        }
    }

    /// Update the GPU view of all modified images
    pub(crate) fn upload_all(&mut self) -> Result<()> {
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

            if cfg!(feature = "gpu-debug-utils") {
                let device = self.upload_queue.device();
                device.set_debug_utils_object_name(&commands, Some("Bulk concentration upload"))?;
            }

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
    pub(crate) fn download_after(
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

        if cfg!(feature = "gpu-debug-utils") {
            let device = self.download_queue.device();
            device.set_debug_utils_object_name(&commands, Some("Single concentration download"))?;
        }

        after
            .then_execute(self.download_queue.clone(), commands)?
            .then_signal_fence_and_flush()?
            .wait(None)?;

        Ok(())
    }
}

/// Memory allocator (hardcoded for now)
pub(crate) type MemAlloc = StandardMemoryAllocator;

/// Command buffer allocator (hardcoded for now)
type CommAlloc = StandardCommandBufferAllocator;

/// Command buffer builder
type CommandBufferBuilder<CommAlloc> =
    AutoCommandBufferBuilder<PrimaryAutoCommandBuffer, CommAlloc>;
