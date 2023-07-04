//! Facilities that are specific to CPU implementations

use crate::{Simulate, SimulateBase, SimulateCreate};
use data::{
    array2,
    concentration::Species,
    parameters::{stencil_offset, STENCIL_SHAPE},
};
use ndarray::{ArrayBase, ArrayView2, ArrayViewMut2, Axis, Dimension, RawData, ShapeBuilder};

/// Simplified version of Simulate that simulates a single time step at a time
///
/// If you implement this, then a [`Simulate`] implementation that loops while
/// flipping the concentration arrays will be automatically provided.
///
/// This is good enough for single- and multi-core CPU computations, but GPU
/// and distributed computations may benefit from the batching of simulation
/// time steps that is enabled by direct implementation of the `Simulate` trait.
pub trait SimulateStep: SimulateBase + SimulateCreate {
    /// Perform a single simulation time step
    ///
    /// At the end of the simulation, the output concentrations of `species`
    /// will contain the final simulation results. It is the job of the caller
    /// to flip the concentrations if they want the result to be their input.
    fn perform_step(&self, species: &mut Species<Self::Concentration>) -> Result<(), Self::Error>;
}
//
impl<T: SimulateStep> Simulate for T {
    fn perform_steps(
        &self,
        species: &mut Species<Self::Concentration>,
        steps: usize,
    ) -> Result<(), Self::Error> {
        for _ in 0..steps {
            self.perform_step(species)?;
            species.flip()?;
        }
        Ok(())
    }
}

/// Lower-level grid-based interface to a CPU compute backend
///
/// Some CPU compute backends expose a lower-level interface based on
/// computations on grids of concentration values.
///
/// This is used by the block-based backends (`block`, `parallel`, ...) to
/// slice the original step computations into smaller sub-computations for
/// cache locality and parallelization purposes.
///
/// If you implement this, then `Simulate` will be implemented automatically
pub trait SimulateCpu: SimulateBase + SimulateCreate {
    /// Concentration values at a point on the simulation's grid
    ///
    /// Can be a single value or some SIMD type containing multiple values
    type Values;

    /// Extract a view of the full grid from the concentrations array
    fn extract_grid(species: &mut Species<Self::Concentration>) -> CpuGrid<Self::Values>;

    /// Perform one simulation time step on the full grid or a subset thereof
    ///
    /// This method does not check the grid for consistency, but is used to
    /// implement `step_impl` that does perform some sanity checks.
    fn unchecked_step_impl(&self, grid: CpuGrid<Self::Values>);

    /// Check that the CpuGrid seems correct
    ///
    /// Note that full correctness checking would involve making sure that the
    /// input and output array views point to the same region of the full grid,
    /// which cannot be done. Therefore, this is only a partial sanity check.
    fn check_grid(([in_u, in_v], [out_u_center, out_v_center]): &CpuGrid<Self::Values>) {
        debug_assert_eq!(in_u.shape(), in_v.shape());
        debug_assert_eq!(out_u_center.shape(), out_v_center.shape());

        let stencil_offset = data::parameters::stencil_offset();
        debug_assert_eq!(in_u.nrows(), out_u_center.nrows() + 2 * stencil_offset[0]);
        debug_assert_eq!(in_u.ncols(), out_u_center.ncols() + 2 * stencil_offset[1]);
    }

    /// Like `unchecked_step_impl()`, but with some sanity checks
    #[inline]
    fn step_impl(&self, grid: CpuGrid<Self::Values>) {
        Self::check_grid(&grid);
        self.unchecked_step_impl(grid);
    }

    /// Count the total number of grid elements which `step_impl()` would manipulate
    #[inline]
    fn grid_len(grid: &CpuGrid<Self::Values>) -> usize {
        Self::check_grid(grid);
        let ([in_u, in_v], [out_u_center, out_v_center]) = grid;
        in_u.len() + in_v.len() + out_u_center.len() + out_v_center.len()
    }

    /// Count the number of grid elements which `step_impl()` would manipulate
    /// over the course of processing a single line of output elements
    #[inline]
    fn grid_line_len(grid: &CpuGrid<Self::Values>) -> usize {
        Self::check_grid(grid);
        let ([in_u, in_v], [out_u_center, out_v_center]) = grid;
        3 * in_u.ncols() + 3 * in_v.ncols() + out_u_center.ncols() + out_v_center.ncols()
    }

    /// Split the grid on which `step_impl()` operates into two parts
    ///
    /// If no split axis is specified, the split is performed across the longest
    /// axis in order to maximize the number of shared elements.
    #[inline]
    fn split_grid<'input, 'output>(
        grid: CpuGrid<'input, 'output, Self::Values>,
        axis_idx: Option<usize>,
    ) -> [CpuGrid<'input, 'output, Self::Values>; 2] {
        Self::check_grid(&grid);
        let ([in_u, in_v], [out_u_center, out_v_center]) = grid;

        // Split across the longest grid axis
        let out_shape = out_u_center.shape();
        let (axis_idx, out_length) =
            axis_idx
                .map(|idx| (idx, out_shape[idx]))
                .unwrap_or_else(|| {
                    out_shape
                        .iter()
                        .copied()
                        .enumerate()
                        .max_by_key(|(_idx, length)| *length)
                        .unwrap()
                });
        let axis = Axis(axis_idx);
        let stencil_offset = data::parameters::stencil_offset()[axis_idx];

        // Splitting the output slice is easy
        let out_split_point = out_length / 2;
        let (out_u_1, out_u_2) = out_u_center.split_at(axis, out_split_point);
        let (out_v_1, out_v_2) = out_v_center.split_at(axis, out_split_point);

        // On the input side, we must mind the edge elements
        let in_split_point = out_split_point + stencil_offset;
        //
        let in_end_1 = in_split_point + stencil_offset;
        let in_u_1 = in_u.split_at(axis, in_end_1).0;
        let in_v_1 = in_v.split_at(axis, in_end_1).0;
        let result_1 = ([in_u_1, in_v_1], [out_u_1, out_v_1]);
        //
        let in_start_2 = in_split_point - stencil_offset;
        let in_u_2 = in_u.split_at(axis, in_start_2).1;
        let in_v_2 = in_v.split_at(axis, in_start_2).1;
        let result_2 = ([in_u_2, in_v_2], [out_u_2, out_v_2]);
        //
        [result_1, result_2]
    }
}
//
/// Low-level representation of the simulation grid used by SimulateCpu
///
/// Composed of the input and output concentrations of species U and V. Note
/// that the input concentrations include a neighborhood of size
/// [`data::parameters::stencil_offset()`] around the region of interest.
pub type CpuGrid<'input, 'output, Values> = (
    [ArrayView2<'input, Values>; 2],
    [ArrayViewMut2<'output, Values>; 2],
);
//
impl<T: SimulateCpu> SimulateStep for T {
    fn perform_step(&self, species: &mut Species<Self::Concentration>) -> Result<(), Self::Error> {
        self.step_impl(Self::extract_grid(species));
        Ok(())
    }
}

/// Optimized iteration over (a regular chunk of) the simulation grid
///
/// In an ideal world, this would be just...
///
/// ```
/// (out_u_center.iter_mut())
///     .zip(&mut out_v_center)
///     .zip(in_u.windows(STENCIL_SHAPE))
///     .zip(in_v.windows(STENCIL_SHAPE))
/// ```
///
/// But at present time, rustc/LLVM cannot optimize the zipped iterator as well
/// as a specialized explicit joint iterator...
#[inline]
pub fn fast_grid_iter<'grid, 'input: 'grid, 'output: 'grid, Values>(
    ([in_u, in_v], [mut out_u_center, mut out_v_center]): CpuGrid<'input, 'output, Values>,
) -> impl Iterator<
    Item = (
        &'output mut Values,
        &'output mut Values,
        ArrayView2<'input, Values>,
        ArrayView2<'input, Values>,
    ),
> {
    // Assert that the sub-grids have the expected shape
    let stencil_offset = stencil_offset();
    let out_shape = [out_u_center.nrows(), out_u_center.ncols()];
    assert_eq!(out_v_center.shape(), &out_shape[..]);
    let in_shape = array2(|i| out_shape[i] + 2 * stencil_offset[i]);
    assert_eq!(in_u.shape(), &in_shape[..]);
    assert_eq!(in_v.shape(), &in_shape[..]);

    // Assert that the sub-grids have the expected strides
    fn checked_row_stride<A, S, D>(arrays: [&ArrayBase<S, D>; 2]) -> usize
    where
        S: RawData<Elem = A>,
        D: Dimension,
    {
        let strides = arrays[0].strides();
        assert_eq!(strides, arrays[1].strides());
        let &[row_stride, 1] = strides else {
            unreachable!()
        };
        row_stride as usize
    }
    let out_row_stride = checked_row_stride([&out_u_center, &out_v_center]);
    let in_row_stride = checked_row_stride([&in_u, &in_v]);

    // Prepare a way to access input windows and output refs by output position
    // The safety of the closures below is actually asserted on the caller's
    // side, but sadly unsafe closures aren't a thing in Rust yet.
    let window_shape = (STENCIL_SHAPE[0], STENCIL_SHAPE[1]).strides((in_row_stride, 1));
    let offset = |position: [usize; 2], row_stride: usize| -> usize {
        position[0] * row_stride + position[1]
    };
    let unchecked_output = move |out_ptr: *mut Values, out_pos| unsafe {
        &mut *out_ptr.add(offset(out_pos, out_row_stride))
    };
    let unchecked_input_window = move |in_ptr: *const Values, out_pos| unsafe {
        let win_ptr = in_ptr.add(offset(out_pos, in_row_stride));
        ArrayView2::from_shape_ptr(window_shape, win_ptr)
    };

    // Start iteration
    let in_u_ptr = in_u.as_ptr();
    let in_v_ptr = in_v.as_ptr();
    let out_u_ptr = out_u_center.as_mut_ptr();
    let out_v_ptr = out_v_center.as_mut_ptr();
    let mut out_pos = [0, 0];
    std::iter::from_fn(move || {
        // Handle end of iteration
        if out_pos[0] == out_shape[0] {
            return None;
        }

        // Produce current result
        // Safe because it will only be called on valid window positions
        let out_u = unchecked_output(out_u_ptr, out_pos);
        let out_v = unchecked_output(out_v_ptr, out_pos);
        let win_u = unchecked_input_window(in_u_ptr, out_pos);
        let win_v = unchecked_input_window(in_v_ptr, out_pos);

        // Advance iterator
        out_pos[1] += 1;
        if out_pos[1] == out_shape[1] {
            out_pos[0] += 1;
            out_pos[1] = 0;
        }

        // Emit result
        Some((out_u, out_v, win_u, win_v))
    })
}
