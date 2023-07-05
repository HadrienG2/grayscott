//! Resource setup related to image sampling and storage

use crate::{Result, Species};
use compute::gpu::VulkanContext;
use std::{collections::hash_map::Entry, sync::Arc};
use vulkano::{
    descriptor_set::{
        layout::{DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo},
        PersistentDescriptorSet, WriteDescriptorSet,
    },
    image::{
        view::{ImageView, ImageViewCreateInfo},
        ImageUsage, StorageImage,
    },
    pipeline::{ComputePipeline, Pipeline},
    sampler::{BorderColor, Sampler, SamplerAddressMode, SamplerCreateInfo},
};

/// Shader descriptor set to which input and output images are bound
pub const IMAGES_SET: u32 = 0;

/// Descriptor within `IMAGES_SET` for sampling of input U concentration
const IN_U: u32 = 0;

/// Descriptor within `IMAGES_SET` for sampling of input V concentration
const IN_V: u32 = 1;

/// Descriptor within `IMAGES_SET` for writing to output U concentration
const OUT_U: u32 = 2;

/// Descriptor within `IMAGES_SET` for writing to output V concentration
const OUT_V: u32 = 3;

/// Generate the callback used to configure image sampling during GPU compute
/// pipeline construction
pub fn sampler_setup_callback(
    context: &VulkanContext,
) -> Result<impl FnOnce(&mut [DescriptorSetLayoutCreateInfo])> {
    let input_sampler = Sampler::new(
        context.device.clone(),
        SamplerCreateInfo {
            address_mode: [SamplerAddressMode::ClampToBorder; 3],
            border_color: BorderColor::FloatOpaqueBlack,
            unnormalized_coordinates: true,
            ..Default::default()
        },
    )?;
    context.set_debug_utils_object_name(&input_sampler, || "Concentration sampler".into())?;
    Ok(
        move |descriptor_sets: &mut [DescriptorSetLayoutCreateInfo]| {
            fn binding(
                set: &mut DescriptorSetLayoutCreateInfo,
                idx: u32,
            ) -> &mut DescriptorSetLayoutBinding {
                set.bindings.get_mut(&idx).unwrap()
            }
            let images_set = &mut descriptor_sets[IMAGES_SET as usize];
            binding(images_set, IN_U).immutable_samplers = vec![input_sampler.clone()];
            binding(images_set, IN_V).immutable_samplers = vec![input_sampler];
        },
    )
}

/// Acquire a descriptor set to bind the proper images
pub fn descriptor_set(
    context: &VulkanContext,
    pipeline: &ComputePipeline,
    species: &mut Species,
) -> Result<Arc<PersistentDescriptorSet>> {
    // Acquire access to the input and output images
    let (in_u, in_v, out_u, out_v) = species.in_out();
    let images = [in_u, in_v, out_u, out_v].map(|i| i.access_image().clone());

    // Have we seen this input + outpt images configuration before?
    match species.context().descriptor_sets.entry(images) {
        // If so, reuse previously configured descriptor set
        Entry::Occupied(occupied) => Ok(occupied.get().clone()),

        // Otherwise, make a new descriptor set
        Entry::Vacant(vacant) => {
            let [in_u, in_v, out_u, out_v] = vacant.key();
            let binding =
                |binding, image: &Arc<StorageImage>, usage| -> Result<WriteDescriptorSet> {
                    Ok(WriteDescriptorSet::image_view(
                        binding,
                        ImageView::new(
                            image.clone(),
                            ImageViewCreateInfo {
                                usage,
                                ..ImageViewCreateInfo::from_image(&image)
                            },
                        )?,
                    ))
                };
            let input_binding = |idx, image| -> Result<WriteDescriptorSet> {
                binding(idx, image, ImageUsage::SAMPLED)
            };
            let output_binding = |idx, image| -> Result<WriteDescriptorSet> {
                binding(idx, image, ImageUsage::STORAGE)
            };
            let layout =
                pipeline.layout().set_layouts()[usize::try_from(IMAGES_SET).unwrap()].clone();
            let set = PersistentDescriptorSet::new(
                &context.descriptor_allocator,
                layout,
                [
                    input_binding(IN_U, in_u)?,
                    input_binding(IN_V, in_v)?,
                    output_binding(OUT_U, out_u)?,
                    output_binding(OUT_V, out_v)?,
                ],
            )?;
            // FIXME: Name this descriptor set once vulkano allows for it
            Ok(vacant.insert(set).clone())
        }
    }
}
