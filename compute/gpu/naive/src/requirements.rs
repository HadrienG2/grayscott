//! Requirements that the sampled image, thread-per-pixel simulation strategy
//! used by this backend impose on the GPU.

use crate::{Error, Result};
use data::{concentration::gpu::image::ImageConcentration, Precision};
use vulkano::{device::physical::PhysicalDevice, format::FormatFeatures, image::ImageUsage};

/// Device requirements
pub fn device_filter(device: &PhysicalDevice, work_group_size: [u32; 3]) -> bool {
    let properties = device.properties();
    let Ok(format_properties) = device.format_properties(ImageConcentration::format()) else {
        return false;
    };

    let num_samplers = 2;
    let num_storage_images = 2;

    properties.max_bound_descriptor_sets >= 1
        && properties.max_compute_work_group_invocations
            >= work_group_size.into_iter().product::<u32>()
        && (properties.max_compute_work_group_size.into_iter())
            .zip(work_group_size)
            .all(|(max, req)| max >= req)
        && properties.max_descriptor_set_samplers >= num_samplers
        && properties.max_per_stage_descriptor_samplers >= num_samplers
        && properties.max_descriptor_set_storage_images >= num_storage_images
        && properties.max_per_stage_descriptor_storage_images >= num_storage_images
        && properties.max_per_set_descriptors.unwrap_or(u32::MAX)
            >= num_samplers + num_storage_images
        && properties.max_per_stage_resources >= 2 * num_samplers + num_storage_images
        && properties.max_sampler_allocation_count >= num_samplers
        && format_properties.optimal_tiling_features.contains(
            FormatFeatures::SAMPLED_IMAGE
                | FormatFeatures::STORAGE_IMAGE
                | ImageConcentration::required_image_format_features(),
        )
}

/// Way in which we use images
pub fn image_usage() -> ImageUsage {
    ImageUsage::SAMPLED | ImageUsage::STORAGE
}

/// Problem size requirements that are present no matter how parameters are passed
pub fn check_image_shape(
    device: &PhysicalDevice,
    shape: [usize; 2],
    work_group_size: [u32; 3],
) -> Result<()> {
    // The simple shader can't accomodate a problem size that is not a
    // multiple of the work group size
    let global_size = crate::shape_to_global_size(shape);
    if (global_size.into_iter())
        .zip(work_group_size)
        .any(|(sh, wg)| sh % wg as usize != 0)
    {
        return Err(Error::UnsupportedShape);
    }
    let dispatch_size: [usize; 3] =
        std::array::from_fn(|i| global_size[i] / work_group_size[i] as usize);

    // Check device properties
    let num_texels = shape.into_iter().product::<usize>();
    let image_size = num_texels * std::mem::size_of::<Precision>();
    let properties = device.properties();
    if (properties.max_compute_work_group_count.into_iter())
        .zip(dispatch_size)
        .any(|(max, req)| (max as usize) < req)
        || (properties.max_image_dimension2_d as usize) < shape.into_iter().max().unwrap()
        || properties.max_memory_allocation_size.unwrap_or(u64::MAX) < image_size as u64
        || properties.max_buffer_size.unwrap_or(u64::MAX) < image_size as u64
    {
        return Err(Error::UnsupportedShape);
    }

    // Check image format properties
    let image_format_info = ImageConcentration::image_format_info(image_usage());
    let Some(image_format_properties) = device.image_format_properties(image_format_info)? else {
        return Err(Error::UnsupportedShape);
    };
    if (image_format_properties.max_extent.into_iter())
        .zip(global_size)
        .any(|(max, req)| (max as usize) < req)
        || image_format_properties.max_resource_size < image_size as u64
    {
        return Err(Error::UnsupportedShape);
    }
    Ok(())
}
