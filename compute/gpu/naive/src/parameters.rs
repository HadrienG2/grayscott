//! Simulation parameters

use crate::{pipeline, Result};
use compute::gpu::context::VulkanContext;
use crevice::std140::AsStd140;
use data::parameters::Parameters;
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, Subbuffer},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
};

/// GPU-readable version of the parameters
pub type GpuParameters = <Parameters as AsStd140>::Output;

/// Move parameters to GPU-accessible memory
pub fn expose(context: &VulkanContext, parameters: Parameters) -> Result<Subbuffer<GpuParameters>> {
    let parameters = Buffer::from_data(
        context.memory_allocator.clone(),
        BufferCreateInfo {
            usage: pipeline::parameters_usage(),
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        parameters.as_std140(),
    )?;
    context.set_debug_utils_object_name(parameters.buffer(), || "Simulation parameters".into())?;
    Ok(parameters)
}
