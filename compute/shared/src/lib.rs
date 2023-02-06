//! Common facilities shared by all compute backends

#[cfg(feature = "criterion")]
use criterion::{BenchmarkId, Criterion, Throughput};
use data::{concentration::Species, parameters::Parameters};

/// Expected reaction simulation backend interface
pub trait Simulate {
    /// Concentration type
    type Concentration: data::concentration::Concentration;

    /// Set up the simulation
    fn new(params: Parameters) -> Self;

    /// Perform one simulation time step
    fn step(&self, species: &mut Species<Self::Concentration>);
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
