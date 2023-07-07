//! Simulation domain aka "Species"

use crate::{pipeline, Error, Result};
use compute::gpu::context::{config::requirements, VulkanContext};
use data::concentration::gpu::{
    image::{context::ImageContext, ImageConcentration},
    shape::Shape,
};
use std::{collections::hash_map::Entry, sync::Arc};
use vulkano::{
    descriptor_set::PersistentDescriptorSet, device::Queue, image::ImageUsage,
    pipeline::ComputePipeline,
};

/// Chosen concentration type
pub type Species = data::concentration::Species<ImageConcentration>;

/// Manner in which images are used, in the context of Species' multi-buffering
pub fn image_usage() -> ImageUsage {
    pipeline::input_usage() | pipeline::output_usage()
}

/// Reusable implementation of SimulateBase::make_species
pub fn make_species(
    context: &VulkanContext,
    domain_shape: [usize; 2],
    work_group_shape: Shape,
    queue: Arc<Queue>,
) -> Result<Species> {
    // Check if the pipelin dispatch size is OK
    let dispatch_size = pipeline::dispatch_size(
        Shape::try_from(domain_shape).map_err(|_| Error::UnsupportedShape)?,
        work_group_shape,
    )?;
    let properties = context.device.physical_device().properties();
    if !requirements::for_dispatch(properties, dispatch_size) {
        return Err(Error::UnsupportedShape);
    }

    // Create simulation data storage
    Ok(Species::new(
        ImageContext::new(
            context.memory_allocator.clone(),
            context.command_allocator.clone(),
            queue.clone(),
            queue,
            [],
            image_usage(),
        )?,
        domain_shape,
    )?)
}

/// Recover the dispatch size for a certain simulation domain
///
/// The validity of the dispatch is assumed to have been checked in
/// `make_species()`.
pub fn dispatch_size_for(species: &Species, work_group_shape: Shape) -> [u32; 3] {
    pipeline::dispatch_size(
        species
            .shape()
            .try_into()
            .expect("Cannot fail (checked at make_species time)"),
        work_group_shape,
    )
    .expect("Cannot fail (checked at make_species time)")
}

/// Acquire a descriptor set for the current (input, output) configuration
pub fn images_descriptor_set(
    context: &VulkanContext,
    pipeline: &ComputePipeline,
    species: &mut Species,
) -> Result<Arc<PersistentDescriptorSet>> {
    // Acquire access to the input and output images
    let (in_u, in_v, out_u, out_v) = species.in_out();
    let images = [in_u, in_v, out_u, out_v].map(|i| i.access_image().clone());

    // Have we seen this input + outpt images configuration before?
    let set = match species.context().descriptor_sets.entry(images) {
        // If so, reuse previously configured descriptor set
        Entry::Occupied(occupied) => occupied.get().clone(),

        // Otherwise, make a new descriptor set
        Entry::Vacant(vacant) => {
            let images = vacant.key().clone();
            vacant
                .insert(pipeline::new_images_set(context, pipeline, images)?)
                .clone()
        }
    };
    Ok(set)
}
