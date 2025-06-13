//! Manually vectorized implementation of Gray-Scott simulation
//!
//! This implementation is vectorized through direct use of hardware intrinsics.
//! Overall, it does not compare favorably with the autovectorized version from
//! the perspective of code complexity and portability, but its runtime
//! performance is comparable. This shows that compilers have become good enough
//! at autovectorizing stencil computations and there is no need to manually
//! vectorize those anymore.

mod simd;

use self::simd::{Scalar, Vector};
use compute::{
    cpu::{CpuGrid, SimulateCpu},
    NoArgs, SimulateBase, SimulateCreate,
};
use data::{
    concentration::simd::SIMDConcentration,
    parameters::{stencil_offset, Parameters},
    Precision,
};
use std::convert::Infallible;

/// Chosen concentration type (see below for vector size choice details)
pub type Values = <Precision as Scalar>::Vectorized;
type Concentration = SIMDConcentration<{ Values::WIDTH }, Values>;
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

    fn extract_grid(species: &mut Species) -> CpuGrid<'_, '_, Self::Values> {
        let (in_u, in_v, out_u, out_v) = species.in_out();
        (
            [in_u.view(), in_v.view()],
            [out_u.simd_center_mut(), out_v.simd_center_mut()],
        )
    }

    #[inline]
    fn unchecked_step_impl(&self, grid: CpuGrid<Self::Values>) {
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
            let mut full_u_1 = win_u[[0, 0]].mul(weights[0][0]);
            let mut full_v_1 = win_v[[0, 0]].mul(weights[0][0]);
            let mut full_u_2 = win_u[[0, 1]].mul(weights[0][1]);
            let mut full_v_2 = win_v[[0, 1]].mul(weights[0][1]);
            let mut full_u_3 = win_u[[0, 2]].mul(weights[0][2]);
            let mut full_v_3 = win_v[[0, 2]].mul(weights[0][2]);
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
            let uv_square = u.mul(v).mul(v);
            let du = diffusion_rate_u
                .mul(full_u)
                .add(feed_rate.mul(ones.sub(u)).sub(uv_square));
            let dv = diffusion_rate_v
                .mul(full_v)
                .add(min_feed_kill.mul(v).add(uv_square));
            *out_u = u.add(du.mul(time_step));
            *out_v = v.add(dv.mul(time_step));
        }
    }
}
