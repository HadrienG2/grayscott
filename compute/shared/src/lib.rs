//! Common facilities shared by all compute backends

#[cfg(feature = "gpu")]
pub mod gpu;

use clap::Args;
#[cfg(feature = "criterion")]
use criterion::{BenchmarkId, Criterion, Throughput};
use data::{
    array2,
    concentration::{Concentration, Species},
    parameters::{stencil_offset, Parameters, STENCIL_SHAPE},
};
use ndarray::{ArrayBase, ArrayView2, ArrayViewMut2, Axis, Dimension, RawData, ShapeBuilder};
use std::{assert_eq, error::Error, fmt::Debug};

/// Commonalities between all ways to implement a simulation
pub trait SimulateBase: Sized {
    /// Supplementary CLI arguments allowing fine-tuning of this backend
    ///
    /// To honor the principle of least surprise and make criterion
    /// microbenchmarks work smoothly, any argument you add must have a default
    /// value and should also be configurable through environment variables.
    type CliArgs: Args + Debug;

    /// Concentration type
    type Concentration: Concentration;

    /// Error type used by simulation operations
    type Error: Error + From<<Self::Concentration as Concentration>::Error> + Send + Sync;

    /// Set up a species concentration grid
    fn make_species(&self, shape: [usize; 2]) -> Result<Species<Self::Concentration>, Self::Error>;
}

/// Commonalities between all ways to set up a simulation
pub trait SimulateCreate : SimulateBase {
    /// Set up the simulation
    fn new(params: Parameters, args: Self::CliArgs) -> Result<Self, Self::Error>;
}

/// No CLI parameters
#[derive(Args, Copy, Clone, Debug, Default, Eq, Hash, PartialEq)]
pub struct NoArgs;

/// Simulation compute backend interface expected by the binaries
pub trait Simulate: SimulateBase + SimulateCreate {
    /// Perform `steps` simulation time steps on the specified grid
    ///
    /// At the end of the simulation, the input concentrations of `species` will
    /// contain the final simulation results.
    fn perform_steps(
        &self,
        species: &mut Species<Self::Concentration>,
        steps: usize,
    ) -> Result<(), Self::Error>;
}

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
    fn unchecked_step_impl(
        &self,
        grid: CpuGrid<Self::Values>,
    );

    /// Check that the CpuGrid seems correct
    ///
    /// Note that full correctness checking would involve making sure that the
    /// input and output array views point to the same region of the full grid,
    /// which cannot be done. Therefore, this is only a partial sanity check.
    fn check_grid(([in_u, in_v], [out_u_center, out_v_center]): &CpuGrid<Self::Values>) {
        debug_assert_eq!(in_u.shape(), in_v.shape());
        debug_assert_eq!(out_u_center.shape(), out_v_center.shape());

        let stencil_offset = data::parameters::stencil_offset();
        debug_assert_eq!(out_u_center.nrows(), in_u.nrows() + 2 * stencil_offset[0]);
        debug_assert_eq!(out_u_center.ncols(), in_u.ncols() + 2 * stencil_offset[1]);
    }

    /// Like `unchecked_step_impl()`, but with some sanity checks
    #[inline(always)]
    fn step_impl(
        &self,
        grid: CpuGrid<Self::Values>,
    ) {
        Self::check_grid(&grid);
        self.unchecked_step_impl(grid);
    }

    /// Count the number of grid elements which `step_impl()` would manipulate
    #[inline(always)]
    fn grid_len(grid: &CpuGrid<Self::Values>) -> usize {
        Self::check_grid(grid);
        let ([in_u, in_v], [out_u_center, out_v_center]) = grid;
        in_u.len() + in_v.len() + out_u_center.len() + out_v_center.len()
    }

    /// Split the grid on which `step_impl()` operates into two parts
    ///
    /// The split is performed on the longest axis so that given enough
    /// splitting iterations, the processed grid fragment becomes close to
    /// square, which is optimal from the point of view of cache locality.
    #[inline(always)]
    fn split_grid<'input, 'output>(
        grid: CpuGrid<'input, 'output, Self::Values>,
    ) -> [CpuGrid<'input, 'output, Self::Values>; 2] {
        Self::check_grid(&grid);
        let ([in_u, in_v], [out_u_center, out_v_center]) = grid;

        // Split across the longest grid axis
        let out_shape = out_u_center.shape();
        let (axis_idx, out_length) = out_shape
            .iter()
            .enumerate()
            .max_by_key(|(_idx, length)| *length)
            .unwrap();
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
        let in_u_1 = in_u.clone().split_at(axis, in_end_1).0;
        let in_v_1 = in_v.clone().split_at(axis, in_end_1).0;
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
pub type CpuGrid<'input, 'output, Values> = ([ArrayView2<'input, Values>; 2], [ArrayViewMut2<'output, Values>; 2]);
//
impl<T: SimulateCpu> SimulateStep for T {
    fn perform_step(&self, species: &mut Species<Self::Concentration>) -> Result<(), Self::Error> {
        Ok(self.step_impl(Self::extract_grid(species)))
    }
}

/// Re-export criterion for the criterion_benchmark macro
#[cfg(feature = "criterion")]
pub use criterion;

/// Macro that generates a complete criterion benchmark harness for you
#[macro_export]
#[cfg(feature = "criterion")]
macro_rules! criterion_benchmark {
    ($backend:ident) => {
        fn criterion_benchmark(c: &mut $crate::criterion::Criterion) {
            $crate::criterion_benchmark::<$backend::Simulation>(c, stringify!($backend))
        }
        $crate::criterion::criterion_group!(benches, criterion_benchmark);
        $crate::criterion::criterion_main!(benches);
    };
}

/// Common criterion benchmark for all Gray-Scott reaction computations
/// Use via the criterion_benchmark macro
#[doc(hidden)]
#[cfg(feature = "criterion")]
pub fn criterion_benchmark<Simulation: Simulate>(c: &mut Criterion, backend_name: &str) {
    use clap::{Command, FromArgMatches};
    use std::hint::black_box;

    env_logger::init();

    let args = Simulation::CliArgs::from_arg_matches(
        &Simulation::CliArgs::augment_args(Command::default().no_binary_name(true))
            .get_matches_from(None::<&str>),
    )
    .expect("Failed to parse arguments from defaults & environment");

    let sim = Simulation::new(black_box(Parameters::default()), black_box(args)).unwrap();
    let mut group = c.benchmark_group(format!("{backend_name}"));
    for num_steps_pow2 in 0..=8 {
        let num_steps = 2u64.pow(num_steps_pow2);
        for size_pow2 in 3..=11 {
            let size = 2usize.pow(size_pow2);
            let shape = [size, 2 * size];
            let num_elems = (shape[0] * shape[1]) as u64;

            let mut species = sim.make_species(black_box(shape)).unwrap();

            group.throughput(Throughput::Elements(num_elems * num_steps));
            group.bench_function(
                BenchmarkId::from_parameter(format!(
                    "{}x{}elems,{}steps",
                    shape[1], shape[0], num_steps
                )),
                |b| {
                    b.iter(|| {
                        sim.perform_steps(&mut species, num_steps as usize).unwrap();
                        species.make_result_view().unwrap();
                    });
                },
            );
        }
    }
    group.finish();
}

/// Optimized iteration over (a regular chunk of) the simulation grid
///
/// In an ideal world, this would be just...
///
/// ```
/// (out_u_center.iter_mut())
///     .zip(out_v_center.iter_mut())
///     .zip(in_u.windows(STENCIL_SHAPE))
///     .zip(in_v.windows(STENCIL_SHAPE))
/// ```
///
/// But at present time, rustc/LLVM cannot optimize the zipped iterator as well
/// as a specialized explicit joint iterator...
#[inline(always)]
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
        let &[row_stride, 1] = strides else { unreachable!() };
        row_stride as usize
    }
    let out_row_stride = checked_row_stride([&out_u_center, &out_v_center]);
    let in_row_stride = checked_row_stride([&in_u, &in_v]);

    // Prepare a way to access input windows and output refs by output position
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
