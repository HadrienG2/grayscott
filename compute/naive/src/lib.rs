//! Naive implementation of Gray-Scott simulation
//!
//! This version follows the logic of the naive_propagation.cpp example from the
//! C++ tutorial, and is slow for the same reason.

use compute::{cpu::SimulateStep, NoArgs, SimulateBase, SimulateCreate};
use data::{
    array2,
    concentration::ScalarConcentration,
    parameters::{stencil_offset, Parameters},
};
use std::convert::Infallible;

/// Chosen concentration type
type Species = data::concentration::Species<ScalarConcentration>;

/// Gray-Scott reaction simulation
#[derive(Debug)]
pub struct Simulation {
    /// Simulation parameters
    params: Parameters,
}
//
impl SimulateBase for Simulation {
    type CliArgs = NoArgs;

    type Concentration = ScalarConcentration;

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
impl SimulateStep for Simulation {
    fn perform_step(&self, species: &mut Species) -> Result<(), Infallible> {
        // Access species concentration matrices
        let shape = species.shape();
        let (in_u, in_v, out_u, out_v) = species.in_out();

        // Determine stencil offsets
        let stencil_offset = stencil_offset();

        // Iterate over pixels of the species concentration matrices
        let params = &self.params;
        ndarray::azip!((index (out_row, out_col), out_u in out_u, out_v in out_v, &u in in_u, &v in in_v) {
            // Determine stencil input region
            let out_pos = [out_row, out_col];
            let stencil_start = array2(|i| out_pos[i].saturating_sub(stencil_offset[i]));
            let stencil_end = array2(|i| (out_pos[i] + stencil_offset[i] + 1).min(shape[i]));
            let stencil_range = array2(|i| stencil_start[i]..stencil_end[i]);
            let stencil_slice = ndarray::s![stencil_range[0].clone(), stencil_range[1].clone()];

            // Compute diffusion gradient for u and v
            let [full_u, full_v] = (in_u.slice(stencil_slice).indexed_iter())
                .zip(in_v.slice(stencil_slice))
                .fold(
                    [0.; 2],
                    |[acc_u, acc_v], (((in_row, in_col), &stencil_u), &stencil_v)| {
                        let weight = params.weights.0[in_row][in_col];
                        [acc_u + weight * (stencil_u - u), acc_v + weight * (stencil_v - v)]
                    },
                );

            // Deduce change in u and v
            let uv_square = u * v * v;
            let du = params.diffusion_rate_u * full_u - uv_square + params.feed_rate * (1.0 - u);
            let dv = params.diffusion_rate_v * full_v + uv_square
                - (params.feed_rate + params.kill_rate) * v;
            *out_u = u + du * params.time_step;
            *out_v = v + dv * params.time_step;
        });
        Ok(())
    }
}
