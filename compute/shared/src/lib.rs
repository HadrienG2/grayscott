//! Common facilities shared by all compute backends

#[cfg(feature = "criterion")]
#[doc(hidden)]
pub mod benchmark;
#[cfg(feature = "cpu")]
pub mod cpu;
#[cfg(feature = "gpu")]
pub mod gpu;

use clap::Args;
use data::{
    concentration::{Concentration, Species},
    parameters::Parameters,
};
use std::{error::Error, fmt::Debug};

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

/// No CLI parameters
#[derive(Args, Copy, Clone, Debug, Default, Eq, Hash, PartialEq)]
pub struct NoArgs;

/// Commonalities between all ways to set up a simulation
pub trait SimulateCreate: SimulateBase {
    /// Set up the simulation
    fn new(params: Parameters, args: Self::CliArgs) -> Result<Self, Self::Error>;
}

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

/// Macro that generates a complete criterion benchmark harness for you
/// Consider using the higher-level cpu_benchmark and gpu_benchmark macros below
#[doc(hidden)]
#[macro_export]
#[cfg(feature = "criterion")]
macro_rules! criterion_benchmark {
    ($backend:ident, $($workload:ident => $workload_name:expr),*) => {
        $(
            fn $workload(c: &mut $crate::benchmark::criterion::Criterion) {
                $crate::benchmark::criterion_benchmark::<$backend::Simulation>(
                    c,
                    stringify!($backend),
                    $crate::benchmark::$workload,
                    $workload_name,
                )
            }
        )*
        $crate::benchmark::criterion::criterion_group!(
            benches,
            $($workload),*
        );
        $crate::benchmark::criterion::criterion_main!(benches);
    };
}

/// Macro that generates a CPU criterion benchmark harness for you
#[macro_export]
#[cfg(feature = "criterion")]
macro_rules! cpu_benchmark {
    ($backend:ident) => {
        $crate::criterion_benchmark!(
            $backend,
            compute_workload => "compute",
            full_sync_workload => "full"
        );
    };
}

/// Macro that generates a GPU criterion benchmark harness for you
#[macro_export]
#[cfg(feature = "criterion")]
macro_rules! gpu_benchmark {
    ($backend:ident) => {
        $crate::criterion_benchmark!(
            $backend,
            compute_workload => "compute",
            full_sync_workload => "full_sync",
            full_gpu_future_workload => "full_future"
        );
    };
}
