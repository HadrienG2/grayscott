//! Common facilities shared by all compute backends

#[cfg(feature = "criterion")]
use criterion::{BenchmarkId, Criterion, Throughput};
use data::{concentration::Species, parameters::Parameters};
use ndarray::{Axis, ArrayView2, ArrayViewMut2};

/// Simulation compute backend interface expected by the "reaction" CLI program
pub trait Simulate {
    /// Concentration type
    type Concentration: data::concentration::Concentration;

    /// Set up the simulation
    fn new(params: Parameters) -> Self;

    /// Perform one simulation time step
    fn step(&self, species: &mut Species<Self::Concentration>);
}

/// Lower-level grid-based interface to a simulation's compute backend
///
/// Some compute backends expose a lower-level interface based on computations
/// on grids of concentration values.
///
/// This is used by the block-based backends (`block`, `parallel`, ...) to
/// slice the original step computations into smaller sub-computations for
/// cache locality and parallelization purposes.
pub trait SimulateImpl: Simulate {
    /// Concentration values at a point on the simulation's grid
    ///
    /// Can be a single value or some SIMD type containing multiple values
    type Values;

    /// Perform one simulation time step on the full grid, or a subset thereof
    ///
    /// - `in_u_v` should contain the initial concentrations of species U and V,
    ///   including a neighborhood of size
    ///   [`data::parameters::stencil_offset()`] around the region of interest.
    ///   Both concentration array views should point to the same subsets of
    ///   the full U and V concentration arrays.
    /// - `out_u_v_centers` will receive the final concentrations. Its position
    ///   in the output array should match that of the central region of
    ///   `in_u_v`, without the neighborhood (which will be updated automatically)
    ///
    /// This method checks none of the above properties, but is used to
    /// implement method `step_impl` which performs some sanity checks.
    fn unchecked_step_impl(
        &self,
        in_u_v: [ArrayView2<Self::Values>; 2],
        out_u_v_centers: [ArrayViewMut2<Self::Values>; 2],
    );

    /// Like `unchecked_step_impl()`, but with some sanity checks
    #[inline(always)]
    fn step_impl(
        &self,
        [in_u, in_v]: [ArrayView2<Self::Values>; 2],
        [out_u_center, out_v_center]: [ArrayViewMut2<Self::Values>; 2],
    ) {
        debug_assert_eq!(in_u.shape(), in_v.shape());
        debug_assert_eq!(out_u_center.shape(), out_v_center.shape());

        let stencil_offset = data::parameters::stencil_offset();
        debug_assert_eq!(out_u_center.nrows(), in_u.nrows() + 2 * stencil_offset[0]);
        debug_assert_eq!(out_u_center.ncols(), in_u.ncols() + 2 * stencil_offset[1]);

        self.unchecked_step_impl([in_u, in_v], [out_u_center, out_v_center]);
    }

    /// Split the grid on which `step_impl()` operates into two parts
    ///
    /// The split is performed on the longest axis so that given enough
    /// splitting iterations, the processed grid fragment becomes close to
    /// square, which is optimal from the point of view of cache locality.
    #[inline(always)]
    fn split_grid<'input, 'output>(
        [in_u, in_v]: [ArrayView2<'input, Self::Values>; 2],
        [out_u_center, out_v_center]: [ArrayViewMut2<'output, Self::Values>; 2],
    ) -> [([ArrayView2<'input, Self::Values>; 2], [ArrayViewMut2<'output, Self::Values>; 2]); 2] {
        // Split across the longest grid axis
        let out_shape = out_u_center.shape();
        let (axis_idx, out_length) = out_shape.iter().enumerate().max_by_key(|(_idx, length)| *length).unwrap();
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

/// Macro that generates a complete criterion benchmark harness for you
#[macro_export]
#[cfg(feature = "criterion")]
macro_rules! criterion_benchmark {
    ($backend:ident) => {
        fn criterion_benchmark(c: &mut criterion::Criterion) {
            $crate::criterion_benchmark::<$backend::Simulation>(c, stringify!($backend))
        }
        criterion::criterion_group!(benches, criterion_benchmark);
        criterion::criterion_main!(benches);
    };
}

/// Common criterion benchmark for all Gray-Scott reaction computations
/// Use via the criterion_benchmark macro
#[doc(hidden)]
#[cfg(feature = "criterion")]
pub fn criterion_benchmark<Simulation: Simulate>(c: &mut Criterion, backend_name: &str) {
    use std::hint::black_box;

    let sim = Simulation::new(black_box(Parameters::default()));
    let mut group = c.benchmark_group(format!("{backend_name}::step"));
    for size_pow2 in 3..=9 {
        let size = 2usize.pow(size_pow2);
        let shape = [size, 2 * size];
        let num_elems = (shape[0] * shape[1]) as u64;

        let mut species = Species::<Simulation::Concentration>::new(black_box(shape));

        group.throughput(Throughput::Elements(num_elems));
        group.bench_function(BenchmarkId::from_parameter(num_elems), |b| {
            b.iter(|| {
                sim.step(&mut species);
                species.flip();
            });
        });
    }
    group.finish();
}
