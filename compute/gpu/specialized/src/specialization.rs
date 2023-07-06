//! Specialization constants

use crate::shader::SpecializationConstants;
use data::{
    concentration::gpu::shape::Shape,
    parameters::{Parameters, StencilWeights},
};

/// Generate GPU specialization constants
pub fn constants(parameters: Parameters, work_group_shape: Shape) -> SpecializationConstants {
    // By using struct patterns as done here, we ensure that many possible
    // mismatches between CPU and GPU expectations can be detected.
    let Parameters {
        weights:
            StencilWeights(
                [[weight00, weight10, weight20], [weight01, weight11, weight21], [weight02, weight12, weight22]],
            ),
        diffusion_rate_u,
        diffusion_rate_v,
        feed_rate,
        kill_rate,
        time_step,
    } = parameters;
    SpecializationConstants {
        weight00,
        weight01,
        weight02,
        weight10,
        weight11,
        weight12,
        weight20,
        weight21,
        weight22,
        diffusion_rate_u,
        diffusion_rate_v,
        feed_rate,
        kill_rate,
        time_step,
        constant_0: work_group_shape.width(),
        constant_1: work_group_shape.height(),
    }
}
