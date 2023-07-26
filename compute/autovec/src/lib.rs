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
use itertools::Itertools;
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
        // Determine stencil weights and offset from the top-left corner of the stencil to its center
        let weights = self.params.corrected_weights();
        let stencil_offset = stencil_offset();

        // Prepare vector versions of the scalar computation parameters
        let diffusion_rate_u = Values::splat(self.params.diffusion_rate_u);
        let diffusion_rate_v = Values::splat(self.params.diffusion_rate_v);
        let feed_rate = Values::splat(self.params.feed_rate);
        let min_feed_kill = Values::splat(self.params.min_feed_kill());
        let time_step = Values::splat(self.params.time_step);
        let ones = Values::splat(1.0);

        // Iterate over center pixels of the species concentration matrices
        for (t1, t2) in compute::cpu::fast_grid_iter(grid).tuples::<(_, _)>() {
            let (out_u1, out_v1, win_u1, win_v1) = t1;
            let (out_u2, out_v2, win_u2, win_v2) = t2;

            // Access center value of u
            let u1 = win_u1[stencil_offset];
            let u2 = win_u2[stencil_offset];
            let v1 = win_v1[stencil_offset];
            let v2 = win_v2[stencil_offset];

            // Compute diffusion gradient
            let [full_u1, full_v1] = (win_u1.rows().into_iter())
                .zip(win_v1.rows())
                .zip(weights.0)
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
                            mul_add(weight, stencil_u, acc_u),
                            mul_add(weight, stencil_v, acc_v),
                        ]
                    },
                );
            let [full_u2, full_v2] = (win_u2.rows().into_iter())
                .zip(win_v2.rows())
                .zip(weights.0)
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
                            mul_add(weight, stencil_u, acc_u),
                            mul_add(weight, stencil_v, acc_v),
                        ]
                    },
                );

            // Deduce variation of U and V
            let uv_square_1 = u1 * v1 * v1;
            let uv_square_2 = u2 * v2 * v2;
            let du1 = mul_add(
                diffusion_rate_u,
                full_u1,
                feed_rate * (ones - u1) - uv_square_1,
            );
            let du2 = mul_add(
                diffusion_rate_u,
                full_u2,
                feed_rate * (ones - u2) - uv_square_2,
            );
            let dv1 = mul_add(
                diffusion_rate_v,
                full_v1,
                mul_add(min_feed_kill, v1, uv_square_1),
            );
            let dv2 = mul_add(
                diffusion_rate_v,
                full_v2,
                mul_add(min_feed_kill, v2, uv_square_2),
            );
            *out_u1 = mul_add(du1, time_step, u1);
            *out_u2 = mul_add(du2, time_step, u2);
            *out_v1 = mul_add(dv1, time_step, v1);
            *out_v2 = mul_add(dv2, time_step, v2);
        }
    }
}

// Pick vector size based on hardware support for vectorization of
// floating-point operations (which are the bulk of our SIMD workload)
const PRECISION_SIZE: usize = std::mem::size_of::<Precision>();
cfg_if! {
    // FIXME: AVX-512 is disabled because rustc does not use zmm registers
    //        and there is no way to force it to according to
    //        https://github.com/rust-lang/rust/issues/53312
    /* if #[cfg(target_feature = "avx512f")] {
        pub const WIDTH: usize = 64 / PRECISION_SIZE;
        pub type Values = Vector<align::Align64, Precision, WIDTH>;
    } else */ if #[cfg(target_feature = "avx")] {
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
