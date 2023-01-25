//! Computation parameters

use crate::Precision;

/// Computation parameters
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Parameters {
    /// Matrix of weights
    pub stencil: Stencil,

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
            stencil: [[1.0; 3]; 3],
            // More mathematically accurate but less fun-looking results
            /* stencil:
            [[0.05, 0.2, 0.05],
            [0.2, 0.0, 0.2],
            [0.05, 0.2, 0.05]]*/
            diffusion_rate_u: 0.1,
            diffusion_rate_v: 0.05,
            feed_rate: 0.014,
            kill_rate: 0.054,
            time_step: 1.0,
        }
    }
}

/// Computation stencil
pub type Stencil = [[Precision; 3]; 3];
