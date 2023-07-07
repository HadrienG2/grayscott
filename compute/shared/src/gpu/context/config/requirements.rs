//! Helpers for formulating device requirements

use data::concentration::gpu::shape::Shape;
use vulkano::device::Properties;

/// Device requirements when a particular work-group size is used
pub fn for_work_group(properties: &Properties, work_group_shape: Shape) -> bool {
    let Ok(invocations) = work_group_shape.invocations() else { return false };
    properties.max_compute_work_group_invocations >= invocations
        && (properties.max_compute_work_group_size.into_iter())
            .zip(work_group_shape.vulkan())
            .all(|(max, req)| max >= req)
}

/// Device requirements when a particular dispatch size is used
pub fn for_dispatch(properties: &Properties, dispatch_size: [u32; 3]) -> bool {
    (properties.max_compute_work_group_count.into_iter())
        .zip(dispatch_size)
        .all(|(max, req)| max >= req)
}
