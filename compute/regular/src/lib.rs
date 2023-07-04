//! Regularized implementation of Gray-Scott simulation
//!
//! The naive propagation algorithm has variable stencil bounds due to the way
//! it handles edges. This variability adds complexity to the generated machine
//! code and reduces opportunities for compiler optimization. But it is actually
//! only needed for a few edge pixels of the image, and not for the bulk of the
//! computation. Better code generation can be obtained by discriminating these
//! two scenarios.

use compute::{cpu::SimulateStep, NoArgs, SimulateBase, SimulateCreate};
use data::{
    array2,
    concentration::ScalarConcentration,
    parameters::{stencil_offset, Parameters, STENCIL_SHAPE},
    Precision,
};
use ndarray::{s, ArrayView2};
use std::{convert::Infallible, ops::Range};

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
        // Locate the regular center of the species concentration matrix
        let shape = species.shape();
        let stencil_offset = stencil_offset();
        let near_edge_end = array2(|i| stencil_offset[i].min(shape[i]));
        let far_edge_start =
            array2(|i| (shape[i].saturating_sub(stencil_offset[i])).max(near_edge_end[i]));
        let center_range = array2(|i| near_edge_end[i]..far_edge_start[i]);

        // Process the center and the edge of the matrix separately
        self.step_center(species, center_range.clone());
        self.step_edge(species, center_range);
        Ok(())
    }
}
//
impl Simulation {
    /// Compute pixels in the center of the image, where the full stencil is always used
    fn step_center(&self, species: &mut Species, center_range: [Range<usize>; 2]) {
        // Access species concentration matrices
        let (in_u, in_v, out_u, out_v) = species.in_out();

        // Determine offset from the top-left corner of the stencil to its center
        let stencil_offset = stencil_offset();

        // Iterate over center pixels of the species concentration matrices
        // and corresponding windows of input data
        let center_slice = s![center_range[0].clone(), center_range[1].clone()];
        for (out_u, out_v, win_u, win_v) in compute::cpu::fast_grid_iter((
            [in_u.view(), in_v.view()],
            [out_u.slice_mut(center_slice), out_v.slice_mut(center_slice)],
        )) {
            // Compute pixel
            let (new_u, new_v) = self.compute_pixel(win_u, win_v, STENCIL_SHAPE, stencil_offset);
            *out_u = new_u;
            *out_v = new_v;
        }
    }

    /// Compute pixels on the edges of the image, where the complicated stencil
    /// formula is truly needed
    fn step_edge(&self, species: &mut Species, center_range: [Range<usize>; 2]) {
        // Access species concentration matrices
        let shape = species.shape();
        let (in_u, in_v, out_u, out_v) = species.in_out();

        // Determine offset from the top-left corner of the stencil to its center
        let stencil_offset = stencil_offset();

        // Iterate over edges of the species concentration matrices
        for (out_slice, in_offset) in [
            // Top edge
            (s![..center_range[0].start, ..], [0, 0]),
            // Bottom edge excluding part covered by top edge
            (s![center_range[0].end.., ..], [center_range[0].end, 0]),
            // Left edge excluding part covered by top and bottom edges
            (
                s![center_range[0].clone(), ..center_range[1].start],
                [center_range[0].start, 0],
            ),
            // Right edge excluding part covered by top, bottom and left edges
            (
                s![center_range[0].clone(), center_range[1].end..],
                [center_range[0].start, center_range[1].end],
            ),
        ] {
            // Iterate over edge pixels of the species concentration matrices
            ndarray::azip!((
                index (out_row, out_col),
                out_u in out_u.slice_mut(out_slice),
                out_v in out_v.slice_mut(out_slice),
            ) {
                // Determine stencil input region
                let in_pos = [out_row + in_offset[0], out_col + in_offset[1]];
                let in_start = array2(|i| in_pos[i].saturating_sub(stencil_offset[i]));
                let in_end = array2(|i| (in_pos[i] + stencil_offset[i] + 1).min(shape[i]));
                let in_range = array2(|i| in_start[i]..in_end[i]);
                let in_slice = ndarray::s![in_range[0].clone(), in_range[1].clone()];

                // Deduce stencil shape and center offset
                let stencil_shape = array2(|i| in_end[i] - in_start[i]);
                let stencil_offset = array2(|i| in_pos[i] - in_start[i]);

                // Compute pixel
                let (new_u, new_v) = self.compute_pixel(
                    in_u.slice(in_slice),
                    in_v.slice(in_slice),
                    stencil_shape,
                    stencil_offset
                );
                *out_u = new_u;
                *out_v = new_v;
            });
        }
    }

    /// Compute a pixel of the species concentration matrices
    ///
    /// This function must be inlined for the center pixel special-casing to work
    #[inline]
    fn compute_pixel(
        &self,
        win_u: ArrayView2<'_, Precision>,
        win_v: ArrayView2<'_, Precision>,
        stencil_shape: [usize; 2],
        stencil_offset: [usize; 2],
    ) -> (Precision, Precision) {
        // Check that everything's alright in debug mode
        debug_assert_eq!(win_u.shape(), win_v.shape());
        debug_assert_eq!(win_u.shape(), stencil_shape);
        debug_assert!((stencil_offset.into_iter())
            .zip(stencil_shape)
            .all(|(offset, shape)| offset < shape));

        // Access parameters and center value of u
        let u = win_u[stencil_offset];
        let v = win_v[stencil_offset];
        let params = &self.params;

        // Compute diffusion gradient
        // NOTE: Right now, this matches the computation performed by the
        //       C++ version, where no matter what happens, we start from the
        //       top-left edge of the stencil. But this is dubious, since the
        //       stencil should arguably stay centered on the target pixel. Then
        //       again, it's just going to result in a few weird edge pixels...
        let [full_u, full_v] = (win_u.rows().into_iter())
            .zip(win_v.rows())
            .zip(self.params.weights.0)
            .flat_map(|((u_row, v_row), weights_row)| {
                (u_row.into_iter().copied())
                    .zip(v_row.into_iter().copied())
                    .zip(weights_row)
            })
            .fold(
                [0.; 2],
                |[acc_u, acc_v], ((stencil_u, stencil_v), weight)| {
                    [
                        acc_u + weight * (stencil_u - u),
                        acc_v + weight * (stencil_v - v),
                    ]
                },
            );

        // Deduce variation of U and V
        let uv_square = u * v * v;
        let du = params.diffusion_rate_u * full_u - uv_square + params.feed_rate * (1.0 - u);
        let dv = params.diffusion_rate_v * full_v + uv_square
            - (params.feed_rate + params.kill_rate) * v;
        let u = u + du * params.time_step;
        let v = v + dv * params.time_step;
        (u, v)
    }
}
