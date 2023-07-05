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

    fn extract_grid(species: &mut Species) -> CpuGrid<Self::Values> {
        let (in_u, in_v, out_u, out_v) = species.in_out();
        (
            [in_u.view(), in_v.view()],
            [out_u.simd_center_mut(), out_v.simd_center_mut()],
        )
    }

    #[inline]
    fn unchecked_step_impl(&self, grid: CpuGrid<Self::Values>) {
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
                            weight.mul_add(stencil_u.sub(u), acc_u),
                            weight.mul_add(stencil_v.sub(v), acc_v),
                        ]
                    },
                );

            // Deduce variation of U and V
            let uv_square = u.mul(v).mul(v);
            let du = diffusion_rate_u.mul_add(full_u, feed_rate.mul_sub(ones.sub(u), uv_square));
            let dv = diffusion_rate_v
                .mul_add(full_v, (feed_rate.add(kill_rate)).mul_neg_add(v, uv_square));
            *out_u = du.mul_add(time_step, u);
            *out_v = dv.mul_add(time_step, v);
        }
    }
}
