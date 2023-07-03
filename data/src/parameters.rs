//! Computation parameters

use crate::Precision;
#[cfg(feature = "gpu-parameters")]
use crevice::{
    glsl::{Glsl, GlslStruct},
    std140::{AsStd140, Mat3},
};

/// Computation parameters
#[derive(Copy, Clone, Debug, PartialEq)]
#[cfg_attr(feature = "gpu-parameters", derive(AsStd140, GlslStruct))]
pub struct Parameters {
    /// Matrix of weights to be applied in the stencil computation
    pub weights: StencilWeights,

    /// Diffusion rate of species U
    pub diffusion_rate_u: Precision,

    /// Diffusion rate of species V
    pub diffusion_rate_v: Precision,

    /// Speed of the chemical reaction that feeds U and kills V & P
    pub feed_rate: Precision,

    /// Rate of conversion from V to P
    pub kill_rate: Precision,

    /// Time step (make it shorter to increase precision)
    pub time_step: Precision,
}
//
impl Default for Parameters {
    fn default() -> Self {
        Self {
            // Stencil used by the C++ version
            weights: StencilWeights([[1.0; 3]; 3]),
            // More mathematically accurate but less fun-looking results
            /* weights: StencilWeights(
            [[0.05, 0.2, 0.05],
            [0.2, 0.0, 0.2],
            [0.05, 0.2, 0.05]]), */
            diffusion_rate_u: 0.1,
            diffusion_rate_v: 0.05,
            feed_rate: 0.014,
            kill_rate: 0.054,
            time_step: 1.0,
        }
    }
}

/// Computation stencil, as a row-major matrix
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct StencilWeights(pub [[Precision; STENCIL_SHAPE[1]]; STENCIL_SHAPE[0]]);

// Make StencilWeights uploadable to the GPU if support is enabled
cfg_if::cfg_if! {
    if #[cfg(feature = "gpu-parameters")] {
        impl AsStd140 for StencilWeights {
            type Output = Mat3;

            fn as_std140(&self) -> Mat3 {
                use bytemuck::Zeroable;
                use crevice::std140::Vec3;

                let [[w11, w12, w13], [w21, w22, w23], [w31, w32, w33]] = self.0;
                let vec3 = |x: Precision, y: Precision, z: Precision| -> Vec3 {
                    Vec3 { x, y, z }
                };
                Mat3 {
                    x: vec3(w11, w21, w31),
                    y: vec3(w12, w22, w32),
                    z: vec3(w13, w23, w33),
                    .. Zeroable::zeroed()
                }
            }

            fn from_std140(val: Mat3) -> Self {
                Self(
                    [
                        [val.x.x, val.y.x, val.z.x],
                        [val.x.y, val.y.y, val.z.y],
                        [val.x.z, val.y.z, val.z.z],
                    ]
                )
            }
        }

        // Safe because the AsStd140 impl does emit a Mat3
        unsafe impl Glsl for StencilWeights {
            const NAME: &'static str = "mat3";
        }
    }
}

/// Shape of CPUStencilWeights
pub const STENCIL_SHAPE: [usize; 2] = [3, 3];

/// Offset from the top-left corner of CPUStencilWeights to its center
#[inline]
pub fn stencil_offset() -> [usize; 2] {
    STENCIL_SHAPE.map(|dim| {
        debug_assert_eq!(dim % 2, 1);
        (dim - 1) / 2
    })
}
