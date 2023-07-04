//! Benchmarking utilities
//!
//! Please consider using the macros provided by this crate instead of calling
//! these implementation details directly.

use crate::Simulate;
use clap::{Args, Command, FromArgMatches};
use criterion::{BenchmarkId, Criterion, Throughput};
#[cfg(feature = "gpu")]
use data::concentration::gpu::image::ImageConcentration;
use data::{concentration::Species, parameters::Parameters};
use std::{hint::black_box, sync::Once};
#[cfg(feature = "gpu")]
use vulkano::sync::FlushError;

/// Re-export criterion for the criterion_benchmark macro
#[cfg(feature = "criterion")]
pub use criterion;

// Make sure env_logger is only initialized once
fn init_logger() {
    static INIT_LOGGER: Once = Once::new();
    INIT_LOGGER.call_once(env_logger::init);
}

/// Common criterion benchmark for all Gray-Scott reaction computations
/// Use via the criterion_benchmark macro
pub fn criterion_benchmark<Simulation: Simulate>(
    c: &mut Criterion,
    backend_name: &str,
    mut workload: impl FnMut(&Simulation, &mut Species<Simulation::Concentration>, usize),
    workload_name: &str,
) {
    init_logger();

    let args = Simulation::CliArgs::from_arg_matches(
        &Simulation::CliArgs::augment_args(Command::default().no_binary_name(true))
            .get_matches_from(None::<&str>),
    )
    .expect("Failed to parse arguments from defaults & environment");

    let parameter_base = if workload_name.is_empty() {
        String::new()
    } else {
        format!("{workload_name},")
    };

    let sim = Simulation::new(black_box(Parameters::default()), black_box(args)).unwrap();
    let mut group = c.benchmark_group(backend_name.to_owned());
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
                    "{parameter_base}{}x{}elems,{num_steps}steps",
                    shape[1], shape[0]
                )),
                |b| {
                    b.iter(|| workload(&sim, &mut species, num_steps as usize));
                },
            );
            black_box(species);
        }
    }
    group.finish();
}

// Workload for performing a few simulation steps, without building the result
pub fn compute_workload<Simulation: Simulate>(
    sim: &Simulation,
    species: &mut Species<Simulation::Concentration>,
    num_steps: usize,
) {
    sim.perform_steps(species, num_steps).unwrap();
}

// Full simulation workload, each step being synchronous
pub fn full_sync_workload<Simulation: Simulate>(
    sim: &Simulation,
    species: &mut Species<Simulation::Concentration>,
    num_steps: usize,
) {
    sim.perform_steps(species, num_steps).unwrap();
    black_box(species.make_result_view().unwrap());
}

// Full GPU simulation workload, as a single synchronous transaction
// TODO: Expand beyond ImageConcentration once GpuConcentration is a thing
#[cfg(feature = "gpu")]
pub fn full_gpu_future_workload<
    Simulation: crate::gpu::SimulateGpu<Concentration = ImageConcentration>,
>(
    sim: &Simulation,
    species: &mut Species<ImageConcentration>,
    num_steps: usize,
) where
    <Simulation as crate::SimulateBase>::Error: From<FlushError>,
{
    let steps = sim.prepare_steps(sim.now(), species, num_steps).unwrap();
    black_box(
        species
            .access_result(|v, ctx| v.make_scalar_view_after(steps, ctx))
            .unwrap(),
    );
}
