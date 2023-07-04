//! Auto-vectorized implementation of Gray-Scott simulation
//!
//! While compilers can automatically vectorize computations, said computations
//! must in all but simplest cases be shaped exactly like manually vectorized
//! code based on hardware intrinsics would be. This compute backend follows
//! this strategy, which should allow it to perform decently on hardware other
//! than the hardware it was written for (x86_64), with minimal porting effort
//! revolving around picking the right vector width.

use cfg_if::cfg_if;
use compute::{
    cpu::{CpuGrid, SimulateCpu},
    NoArgs, SimulateBase, SimulateCreate,
};
use data::{
    concentration::simd::SIMDConcentration,
    parameters::{stencil_offset, Parameters},
    Precision,
};
use slipstream::{vector::align, Vector};
use std::convert::Infallible;

/// Chosen concentration type (see below for vector size choice details)
type Concentration = SIMDConcentration<WIDTH, Values>;
type Species = data::concentration::Species<Concentration>;

/// Gray-Scott reaction simulation
#[derive(Debug)]
pub struct Simulation {
    /// Simulation parameters
    params: Parameters,
}
//
impl SimulateBase for Simulation {
    type CliArgs = NoArgs;

    type Concentration = Concentration;

    type Error = Infallible;

    fn make_species(&self, shape: [usize; 2]) -> Result<Species, Infallible> {
        Species::new((), shape)
    }
}
//
impl SimulateCreate for Simulation {
    fn new(params: Parameters, _args: NoArgs) -> Result<Self, Infallible> {
        Ok(Self { params })
    }
}
//
impl SimulateCpu for Simulation {
    type Values = Values;

    fn extract_grid(species: &mut Species) -> CpuGrid<Values> {
        let (in_u, in_v, out_u, out_v) = species.in_out();
        (
            [in_u.view(), in_v.view()],
            [out_u.simd_center_mut(), out_v.simd_center_mut()],
        )
    }

    #[inline]
    fn unchecked_step_impl(&self, grid: CpuGrid<Values>) {
        // Determine offset from the top-left corner of the stencil to its center
        let stencil_offset = stencil_offset();

        // Prepare vector versions of the scalar computation parameters
        let diffusion_rate_u = Values::splat(self.params.diffusion_rate_u);
        let diffusion_rate_v = Values::splat(self.params.diffusion_rate_v);
        let feed_rate = Values::splat(self.params.feed_rate);
        let kill_rate = Values::splat(self.params.kill_rate);
        let time_step = Values::splat(self.params.time_step);
        let ones = Values::splat(1.0);

        // Iterate over center pixels of the species concentration matrices
        for (out_u, out_v, win_u, win_v) in compute::cpu::fast_grid_iter(grid) {
            // Access center value of u
            let u = win_u[stencil_offset];
            let v = win_v[stencil_offset];

            // Compute diffusion gradient
            let [full_u, full_v] = (win_u.rows().into_iter())
                .zip(win_v.rows())
                .zip(self.params.weights.0)
                .flat_map(|((u_row, v_row), weights_row)| {
                    (u_row.into_iter().copied())
                        .zip(v_row.into_iter().copied())
                        .zip(weights_row)
                })
                .fold(
                    [Values::splat(0.); 2],
                    |[acc_u, acc_v], ((stencil_u, stencil_v), weight)| {
                        let weight = Values::splat(weight);
                        [
                            mul_add(weight, stencil_u - u, acc_u),
                            mul_add(weight, stencil_v - v, acc_v),
                        ]
                    },
                );

            // Deduce variation of U and V
            let uv_square = u * v * v;
            let du = mul_add(diffusion_rate_u, full_u, feed_rate * (ones - u) - uv_square);
            let dv = mul_add(
                diffusion_rate_v,
                full_v,
                uv_square - (feed_rate + kill_rate) * v,
            );
            *out_u = mul_add(du, time_step, u);
            *out_v = mul_add(dv, time_step, v);
        }
    }
}

// Pick vector size based on hardware support for vectorization of
// floating-point operations (which are the bulk of our SIMD workload)
const PRECISION_SIZE: usize = std::mem::size_of::<Precision>();
cfg_if! {
    if #[cfg(target_feature = "avx512f")] {
        pub const WIDTH: usize = 64 / PRECISION_SIZE;
        pub type Values = Vector<align::Align64, Precision, WIDTH>;
    } else if #[cfg(target_feature = "avx")] {
        pub const WIDTH: usize = 32 / PRECISION_SIZE;
        pub type Values = Vector<align::Align32, Precision, WIDTH>;
    } else {
        // NOTE: While most non-Intel CPUs use 128-bit vectorization, not all do.
        //       A benefit of autovectorization, however, is that supporting new
        //       hardware can just be a matter of adding cases in this cfg_if.
        pub const WIDTH: usize = 16 / PRECISION_SIZE;
        pub type Values = Vector<align::Align16, Precision, WIDTH>;
    }
}

// Use FMA if supported in hardware (unlike GCC, LLVM does not do it automatically)
cfg_if! {
    // NOTE: Extend this when porting to more CPU architectures
    if #[cfg(any(target_feature = "fma", target_feature = "vfp4"))] {
        #[inline]
        pub fn mul_add(x: Values, y: Values, z: Values) -> Values {
            x.mul_add(y, z)
        }
    } else {
        #[inline]
        pub fn mul_add(x: Values, y: Values, z: Values) -> Values {
            x * y + z
        }
   }
}
