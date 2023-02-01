//! Common microbenchmarking code for all compute crates

use criterion::{BenchmarkId, Criterion, Throughput};
use data::{
    concentration::{Concentration, Species},
    parameters::Parameters,
};
use std::hint::black_box;

/// Common criterion benchmark for all Gray-Scott reaction computations
pub fn criterion_benchmark<C: Concentration>(
    c: &mut Criterion,
    step: impl Fn(&mut Species<C>, &Parameters),
    crate_name: &str,
) {
    let params = black_box(Parameters::default());
    let mut group = c.benchmark_group(format!("{crate_name}::step"));
    for size_pow2 in 3..=9 {
        let size = 2usize.pow(size_pow2);
        let shape = [size, 2 * size];
        let num_elems = (shape[0] * shape[1]) as u64;

        let mut species = Species::<C>::new(black_box(shape));

        group.throughput(Throughput::Elements(num_elems));
        group.bench_function(BenchmarkId::from_parameter(num_elems), |b| {
            b.iter(|| {
                step(&mut species, &params);
                species.flip();
            });
        });
    }
    group.finish();
}

/// Macro that generates the whole criterion benchmark harness for you
#[macro_export]
macro_rules! criterion_benchmark {
    ($crate_name:ident) => {
        fn criterion_benchmark(c: &mut criterion::Criterion) {
            $crate::criterion_benchmark(c, $crate_name::step, stringify!($crate_name))
        }
        criterion::criterion_group!(benches, criterion_benchmark);
        criterion::criterion_main!(benches);
    };
}
