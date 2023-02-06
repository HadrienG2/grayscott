//! Cache blocking implementation of Gray-Scott simulation
//!
//! The `autovec` and `manualvec` versions are actually not compute bound but
//! memory bound. This version uses cache blocking techniques to improve the CPU
//! cache hit rate, getting us back into compute-bound territory.

use autovec::{mul_add, Values};
use compute::Simulate;
use data::{
    concentration::Species,
    parameters::{stencil_offset, Parameters},
};

/// Gray-Scott reaction simulation
pub struct Simulation {
    /// Simulation parameters
    params: Parameters,
}
//
impl Simulate for Simulation {
    type Concentration = <autovec::Simulation as Simulate>::Concentration;

    fn new(params: Parameters) -> Self {
        Self { params }
    }

    // TODO: Add the blocking
    // FIXME: Right now, this is a copypaste of autovec, set up a way to share more code
    fn step(&self, species: &mut Species<Self::Concentration>) {
        // Access species concentration matrices
        let (in_u, out_u) = species.u.in_out();
        let (in_v, out_v) = species.v.in_out();

        // Determine offset from the top-left corner of the stencil to its center
        let stencil_offset = stencil_offset();

        // Prepare vector versions of the scalar computation parameters
        let params = &self.params;
        let diffusion_rate_u = Values::splat(params.diffusion_rate_u);
        let diffusion_rate_v = Values::splat(params.diffusion_rate_v);
        let feed_rate = Values::splat(params.feed_rate);
        let kill_rate = Values::splat(params.kill_rate);
        let time_step = Values::splat(params.time_step);
        let ones = Values::splat(1.0);

        // Iterate over center pixels of the species concentration matrices
        for (((out_u, out_v), win_u), win_v) in (out_u.simd_center_mut().iter_mut())
            .zip(out_v.simd_center_mut().iter_mut())
            .zip(in_u.simd_stencil_windows())
            .zip(in_v.simd_stencil_windows())
        {
            // Access center value of u
            let u = win_u[stencil_offset];
            let v = win_v[stencil_offset];

            // Compute diffusion gradient
            let [full_u, full_v] = (win_u.iter())
                .zip(win_v.iter())
                .zip(params.weights.into_iter().flat_map(|row| row.into_iter()))
                .fold(
                    [Values::splat(0.); 2],
                    |[acc_u, acc_v], ((&stencil_u, &stencil_v), weight)| {
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
