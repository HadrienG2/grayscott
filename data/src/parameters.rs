//! Computation parameters

use std::debug_assert_eq;

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
    weights: StencilWeights,

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
impl Parameters {
    /// Matrix of weights to be applied in the naive
    /// sum(weight * (elem - center)) stencil computation
    #[inline]
    pub fn weights(&self) -> StencilWeights {
        #[cfg(feature = "weights-runtime")]
        {
            // Look up stencil weights at runtime, even though they are
            // actually known at compile time. The compiler may or may not
            // manage to figure things out (as of rustc 1.71, it doesn't).
            self.weights
        }
        #[cfg(not(feature = "weights-runtime"))]
        {
            // Take no chance and enforce use of the compile-time weights.
            debug_assert_eq!(self.weights, STENCIL_WEIGHTS);
            STENCIL_WEIGHTS
        }
    }

    /// Corrected weights which integrate the -center term above
    #[inline]
    pub fn corrected_weights(&self) -> StencilWeights {
        let mut weights = self.weights();
        let stencil_offset = stencil_offset();
        weights.0[stencil_offset[0]][stencil_offset[1]] -=
            weights.0.into_iter().flatten().sum::<Precision>();
        weights
    }

    /// -(feed_rate + kill_rate) prefactor for the dv computation
    #[inline]
    pub fn min_feed_kill(&self) -> Precision {
        -(self.feed_rate + self.kill_rate)
    }
}
//
impl Default for Parameters {
    fn default() -> Self {
        Self {
            weights: STENCIL_WEIGHTS,
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
//
/// Compile-time stencil selection
#[rustfmt::skip]
const STENCIL_WEIGHTS: StencilWeights =
    if cfg!(feature = "weights-pretty") {
        // Stencil used by the C++ version of the course
        //
        // Most likely based on the graph generalization of the Laplacian, which
        // assumes all neighbors to be at an equal distance.
        StencilWeights([[1.0; 3]; 3])
    } else if cfg!(feature = "weights-patrakarttunen") {
        // Patra-Karttunen stencil imposes rotational invariance, has smallest
        // error around the origin
        StencilWeights([
            [1.0/6.0, 4.0/6.0, 1.0/6.0],
            [4.0/6.0,   0.0,   4.0/6.0],
            [1.0/6.0, 4.0/6.0, 1.0/6.0]
        ])
    } else if cfg!(feature = "weights-5points") {
        // 5-points stencil is computationally simpler, but is intrinsically
        // anisotropic which will lead the simulated domain to take a cross-like
        // shape rather than the expected smooth circular shape
        StencilWeights([
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0]
        ])
    } else {
        // Oono-Puri stencil is the optimally isotropic form of discretization,
        // reduces overall error.
        StencilWeights([
            [0.25, 0.5, 0.25],
            [0.5,  0.0, 0.5],
            [0.25, 0.5, 0.25]
        ])
    };

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
