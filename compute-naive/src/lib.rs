//! Naive implementation of Gray-Scott simulation
//!
//! This version follows the logic of the naive_propagation.cpp example from the
//! C++ tutorial, and is slow for the same reason.

use data::{
    concentration::Species,
    parameters::{stencil_offset, Parameters},
};

/// Perform one simulation time step
pub fn step(species: &mut Species, params: &Parameters) {
    // Access species concentration matrices
    let shape = species.shape();
    let (in_u, out_u) = species.u.in_out();
    let (in_v, out_v) = species.v.in_out();

    // Determine stencil offsets
    let stencil_offset = stencil_offset();

    // Jointly iterate over cells of the species concentration matrices
    ndarray::azip!((index (out_row, out_col), out_u in out_u, out_v in out_v, &u in in_u, &v in in_v) {
        // Determine stencil input region
        let out_pos = [out_row, out_col];
        let stencil_start = array2(|i| out_pos[i].saturating_sub(stencil_offset[i]));
        let stencil_end = array2(|i| (out_pos[i] + stencil_offset[i] + 1).min(shape[i]));
        let stencil_range = array2(|i| stencil_start[i]..stencil_end[i]);
        let stencil_slice = ndarray::s![stencil_range[0].clone(), stencil_range[1].clone()];

        // Compute diffusion gradient for u and v
        let [full_u, full_v] = (in_u.slice(stencil_slice).indexed_iter())
            .zip(in_v.slice(stencil_slice).iter())
            .fold(
                [0.; 2],
                |[acc_u, acc_v], (((in_row, in_col), &stencil_u), &stencil_v)| {
                    let weight = params.weights[in_row][in_col];
                    [acc_u + weight * stencil_u, acc_v + weight * stencil_v]
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
}

/// Shorthand to create an array of 2 elements, knowing a function from index to value
fn array2<T>(f: impl FnMut(usize) -> T) -> [T; 2] {
    std::array::from_fn::<T, 2, _>(f)
}
