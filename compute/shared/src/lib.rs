//! Common facilities shared by all compute backends

#[cfg(feature = "criterion")]
use criterion::{BenchmarkId, Criterion, Throughput};
use data::{concentration::Species, parameters::Parameters};
use ndarray::{ArrayView2, ArrayViewMut2};

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
