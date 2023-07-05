//! Specialization constants

use crate::shader::SpecializationConstants;
use data::parameters::{Parameters, StencilWeights};

/// Generate GPU specialization constants
pub fn constants(parameters: Parameters, work_group_size: [u32; 3]) -> SpecializationConstants {
    assert_eq!(work_group_size[2], 1, "This is not a 3D simulation!");

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
        constant_0: work_group_size[0],
        constant_1: work_group_size[1],
    }
}
