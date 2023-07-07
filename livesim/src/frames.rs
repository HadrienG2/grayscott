//! Per-frame state management

use crate::{
    context::SimulationContext,
    input::{self, Input},
    pipeline, surface, Result,
};
use compute::gpu::context::VulkanContext;
use data::concentration::gpu::shape::Shape;
use std::sync::Arc;
use vulkano::{
    command_buffer::{CommandBufferExecFuture, PrimaryAutoCommandBuffer},
    descriptor_set::PersistentDescriptorSet,
    pipeline::ComputePipeline,
    swapchain::{
        self, AcquireError, PresentFuture, Swapchain, SwapchainAcquireFuture, SwapchainPresentInfo,
    },
    sync::{future::FenceSignalFuture, FlushError, GpuFuture},
};

/// State associated with the processing of multiple on-screen frames
pub struct Frames {
    /// Swapchain that indicates which frame should be processed next
    swapchain: Arc<Swapchain>,

    /// Futures that indicate readiness of each frame's swapchain image
    frame_futures: Vec<Option<FrameFuture>>,

    /// Upload buffers for each frame
    upload_buffers: Vec<Input>,

    /// Descriptor sets for input + output of each frame
    inout_sets: Vec<Arc<PersistentDescriptorSet>>,

    /// Truth that the swapchain should be recreated
    recreate_swapchain: bool,
}
//
impl Frames {
    /// Set up per-frame state
    pub fn new(context: &SimulationContext, pipeline: &ComputePipeline) -> Result<Self> {
        // Set up a swapchain
        let (swapchain, swapchain_images) = surface::create_swapchain(&context)?;

        // Set up buffers to upload simulation results to the GPU
        // TODO: If the simulation backend is GPU-based, directly access simulation
        //       storage instead (will require a different rendering pipeline).
        let vulkan = context.vulkan();
        let shape = Shape::from_width_height(swapchain.image_extent());
        let frames_in_flight = swapchain_images.len();
        let upload_buffers = input::create_upload_buffers(vulkan, shape, frames_in_flight)?;

        // Set up input and output descriptor sets
        let inout_sets =
            pipeline::new_inout_sets(vulkan, &pipeline, &upload_buffers[..], swapchain_images)?;

        // Set up futures to track the rendering of each swapchain image
        let frame_futures = (0..frames_in_flight).map(|_| None).collect();

        Ok(Self {
            swapchain,
            frame_futures,
            upload_buffers,
            inout_sets,
            recreate_swapchain: false,
        })
    }

    /// Process the next frame, given a recipe for rendering commands
    pub fn process_frame(
        &mut self,
        context: &SimulationContext,
        pipeline: &ComputePipeline,
        record_commands: impl FnOnce(
            &mut Input,
            Arc<PersistentDescriptorSet>,
        ) -> Result<PrimaryAutoCommandBuffer>,
    ) -> Result<()> {
        // Recreate the swapchain and dependent state as needed
        let vulkan = context.vulkan();
        self.recreate_if_needed(vulkan, pipeline)?;

        // Acquire the next swapchain image, handle swapchain invalidation
        let (image_idx, suboptimal, acquire_future) =
            match swapchain::acquire_next_image(self.swapchain.clone(), None) {
                Ok(r) => r,
                Err(AcquireError::OutOfDate) => {
                    self.recreate_swapchain = true;
                    return Ok(());
                }
                Err(e) => return Err(e.into()),
            };
        if suboptimal {
            self.recreate_swapchain = true;
        }

        // Render the selected frame
        self.schedule_render(context, image_idx, acquire_future, record_commands)
    }

    /// Recreate swapchain and dependent state if need be
    fn recreate_if_needed(
        &mut self,
        vulkan: &VulkanContext,
        pipeline: &ComputePipeline,
    ) -> Result<()> {
        if self.recreate_swapchain {
            self.recreate_swapchain = false;
            let (new_swapchain, new_inout_sets) = surface::recreate_swapchain(
                vulkan,
                pipeline,
                &self.upload_buffers[..],
                &self.swapchain,
            )?;
            self.swapchain = new_swapchain;
            self.inout_sets = new_inout_sets;
        }
        Ok(())
    }

    /// Schedule a rendering command buffer for execution
    fn schedule_render(
        &mut self,
        context: &SimulationContext,
        image_idx_u32: u32,
        acquire_future: SwapchainAcquireFuture,
        record_commands: impl FnOnce(
            &mut Input,
            Arc<PersistentDescriptorSet>,
        ) -> Result<PrimaryAutoCommandBuffer>,
    ) -> Result<()> {
        // Wait for the previous render to this surface to terminate so we
        // can safely write to the upload buffer
        // TODO : In GPU mode, do not wait and just join the frame_future or
        //        now() with the acquire future in scheduling below.
        //        See tag `joined-futures` for reference.
        let image_idx = image_idx_u32 as usize;
        self.await_upload(image_idx)?;

        // Let the user build a command buffer for their rendering
        let commands = record_commands(
            &mut self.upload_buffers[image_idx],
            self.inout_sets[image_idx].clone(),
        )?;

        // Schedule rendering once swapchain image is ready
        let queue = context.queue();
        let schedule_result = acquire_future
            .then_execute(queue.clone(), commands)?
            .then_swapchain_present(
                queue.clone(),
                SwapchainPresentInfo::swapchain_image_index(self.swapchain.clone(), image_idx_u32),
            )
            .then_signal_fence_and_flush();

        // Handle swapchain invalidation during scheduling of render
        match schedule_result {
            Ok(future) => self.frame_futures[image_idx] = Some(future),
            Err(FlushError::OutOfDate) => self.recreate_swapchain = true,
            Err(e) => return Err(e.into()),
        }
        Ok(())
    }

    /// Wait for the GPU to be done with one of the upload buffers
    fn await_upload(&mut self, image_idx: usize) -> Result<()> {
        if let Some(future) = self.frame_futures[image_idx].take() {
            future.wait(None)?;
        }
        Ok(())
    }
}

/// Future of presenting a frame
type FrameFuture = FenceSignalFuture<PresentFuture<RenderFuture>>;

/// Future of rendering
type RenderFuture = CommandBufferExecFuture<AcquireFuture>;

/// Future of swapchain image acquisition
type AcquireFuture = SwapchainAcquireFuture;
