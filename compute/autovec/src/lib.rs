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
        for (out_u, out_v, win_u, win_v) in compute::cpu::fast_grid_iter(grid) {
            // Access center value of u
            let u = win_u[stencil_offset];
            let v = win_v[stencil_offset];

            // Compute SIMD versions of the stencil weights
            let weights = weights.0.map(|row| row.map(Values::splat));

            // Compute diffusion gradient
            let mut full_u_1 = win_u[[0, 0]] * weights[0][0];
            let mut full_v_1 = win_v[[0, 0]] * weights[0][0];
            let mut full_u_2 = win_u[[0, 1]] * weights[0][1];
            let mut full_v_2 = win_v[[0, 1]] * weights[0][1];
            let mut full_u_3 = win_u[[0, 2]] * weights[0][2];
            let mut full_v_3 = win_v[[0, 2]] * weights[0][2];
            full_u_1 = win_u[[1, 0]].mul_add(weights[1][0], full_u_1);
            full_v_1 = win_v[[1, 0]].mul_add(weights[1][0], full_v_1);
            full_u_2 = win_u[[1, 1]].mul_add(weights[1][1], full_u_2);
            full_v_2 = win_v[[1, 1]].mul_add(weights[1][1], full_v_2);
            full_u_3 = win_u[[1, 2]].mul_add(weights[1][2], full_u_3);
            full_v_3 = win_v[[1, 2]].mul_add(weights[1][2], full_v_3);
            full_u_1 = win_u[[2, 0]].mul_add(weights[2][0], full_u_1);
            full_v_1 = win_v[[2, 0]].mul_add(weights[2][0], full_v_1);
            full_u_2 = win_u[[2, 1]].mul_add(weights[2][1], full_u_2);
            full_v_2 = win_v[[2, 1]].mul_add(weights[2][1], full_v_2);
            full_u_3 = win_u[[2, 2]].mul_add(weights[2][2], full_u_3);
            full_v_3 = win_v[[2, 2]].mul_add(weights[2][2], full_v_3);
            let full_u = full_u_1 + full_u_2 + full_u_3;
            let full_v = full_v_1 + full_v_2 + full_v_3;

            // Deduce variation of U and V
            let uv_square = u * v * v;
            let du = diffusion_rate_u * full_u - uv_square + feed_rate * (ones - u);
            let dv = diffusion_rate_v * full_v + uv_square + min_feed_kill * v;
            *out_u = u + du * time_step;
            *out_v = u + dv * time_step;
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
