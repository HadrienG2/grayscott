//! Handling of input data from simulation

use crate::{pipeline, Result};
use compute::{gpu::VulkanContext, SimulateBase};
use compute_selector::Simulation;
use data::{concentration::Species, Precision};
use ndarray::ArrayViewMut2;
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, Subbuffer},
    memory::allocator::{AllocationCreateInfo, MemoryUsage},
};

/// GPU-accessible type from which simulation data is fetched
pub type Input = Subbuffer<[Precision]>;

/// Create buffers for upload of simulation output to the GPU
pub fn create_upload_buffers(
    vulkan: &VulkanContext,
    shape: [usize; 2],
    count: usize,
) -> Result<Vec<Input>> {
    let buffer_len = shape.into_iter().product::<usize>();
    (0..count)
        .map(|idx| {
            let sub_buffer = Buffer::new_slice::<Precision>(
                &vulkan.memory_allocator,
                BufferCreateInfo {
                    usage: pipeline::input_usage(),
                    ..Default::default()
                },
                AllocationCreateInfo {
                    usage: MemoryUsage::Upload,
                    ..Default::default()
                },
                buffer_len as u64,
            )?;
            vulkan.set_debug_utils_object_name(sub_buffer.buffer(), || {
                format!("Upload buffer #{idx}").into()
            })?;
            Ok(sub_buffer)
        })
        .collect()
}

/// Fill an upload buffer with simulation data
pub fn fill_upload_buffer(
    buffer: &mut Input,
    species: &mut Species<<Simulation as SimulateBase>::Concentration>,
) -> Result<()> {
    let mut upload_lock = buffer.write()?;
    let upload_scalars = ArrayViewMut2::from_shape(species.shape(), &mut upload_lock)
        .expect("Should not fail (shape should be right)");
    species.write_result_view(upload_scalars)?;
    Ok(())
}
