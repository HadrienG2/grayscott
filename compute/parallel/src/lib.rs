//! Parallel implementation of Gray-Scott simulation
//!
//! This crates implements a parallel version of the Gray-Scott simulation based
//! on domain decomposition and fork-join parallelism.

use clap::Args;
use compute::{
    cpu::{CpuGrid, SimulateCpu},
    SimulateBase, SimulateCreate,
};
use compute_block::{DefaultBlockSize, SingleCore};
use data::{
    concentration::{Concentration, Species},
    parameters::Parameters,
};
use hwlocality::{errors::RawHwlocError, Topology};
use rayon::{prelude::*, ThreadPool, ThreadPoolBuildError, ThreadPoolBuilder};
use std::num::NonZeroUsize;
use thiserror::Error;

/// Gray-Scott reaction simulation
pub type Simulation = ParallelSimulation<compute_block::Simulation>;

/// Parameters are tunable via CLI args and environment variables
#[derive(Args, Copy, Clone, Debug, Eq, Hash, PartialEq)]
pub struct CliArgs<BackendArgs: Args> {
    /// Number of processing threads
    #[arg(short = 'j', long, env)]
    num_threads: Option<NonZeroUsize>,

    /// Number of processed bytes per parallel task
    ///
    /// This should be tuned somewhere between half the L1 cache size
    /// (default for optimal cache locality in the HT regime) and the L3 per-core
    /// cache size (will reduce work distribution overhead if the backend can
    /// live with the reduced bandwidth and increased latency of the L3 cache).
    #[arg(long, env)]
    seq_block_size: Option<NonZeroUsize>,

    /// Expose backend arguments too
    #[command(flatten)]
    backend: BackendArgs,
}

/// Gray-Scott simulation wrapper that enforces parallel iteration
#[derive(Debug)]
pub struct ParallelSimulation<Backend: SimulateCpu + Sync>
where
    Backend::Values: Send + Sync,
{
    /// Thread pool
    thread_pool: ThreadPool,

    /// Number of grid elements (scalars or SIMD blocks) below which
    /// parallelism is not considered worthwhile
    sequential_len_threshold: usize,

    /// Underlying sequential compute backend
    backend: Backend,
}
//
impl<Backend: SimulateCpu + Sync> SimulateBase for ParallelSimulation<Backend>
where
    Backend::Values: Send + Sync,
{
    type CliArgs = CliArgs<Backend::CliArgs>;

    type Concentration = <Backend as SimulateBase>::Concentration;

    type Error =
        Error<<Backend as SimulateBase>::Error, <Self::Concentration as Concentration>::Error>;

    fn make_species(&self, shape: [usize; 2]) -> Result<Species<Self::Concentration>, Self::Error> {
        self.backend.make_species(shape).map_err(Error::Backend)
    }
}
//
impl<Backend: SimulateCpu + Sync> SimulateCreate for ParallelSimulation<Backend>
where
    Backend::Values: Send + Sync,
{
    fn new(params: Parameters, args: Self::CliArgs) -> Result<Self, Self::Error> {
        // Set up the thread pool
        let thread_pool = ThreadPoolBuilder::new()
            .num_threads(args.num_threads.map(usize::from).unwrap_or(0))
            .build()
            .map_err(Error::ThreadPool)?;

        // Define the granularity of sequential processing
        let seq_block_size = args
            .seq_block_size
            .map(|bs| Ok::<_, Self::Error>(usize::from(bs)))
            .unwrap_or_else(|| {
                let topology = Topology::new().map_err(Error::Hwloc)?;
                let defaults = SingleCore::new(&topology);
                // FIXME: Expose cache-per-thread in hwlocality so I don't need
                //        this hardcoded factor of 2 for hyperthreading.
                Ok(defaults.l2_block_size() / 2)
            })?;
        let sequential_len_threshold = seq_block_size / std::mem::size_of::<Backend::Values>();

        Ok(Self {
            thread_pool,
            sequential_len_threshold,
            backend: Backend::new(params, args.backend).map_err(Error::Backend)?,
        })
    }
}
//
impl<Backend: SimulateCpu + Sync> SimulateCpu for ParallelSimulation<Backend>
where
    Backend::Values: Send + Sync,
{
    type Values = Backend::Values;

    fn extract_grid(species: &mut Species<Self::Concentration>) -> CpuGrid<Self::Values> {
        Backend::extract_grid(species)
    }

    fn unchecked_step_impl(&self, grid: CpuGrid<Self::Values>) {
        self.thread_pool.install(|| {
            rayon::iter::split(grid, |subgrid| {
                if Self::grid_len(&subgrid) <= self.sequential_len_threshold {
                    (subgrid, None)
                } else {
                    let [half1, half2] = Self::split_grid(subgrid, None);
                    (half1, Some(half2))
                }
            })
            .for_each(|subgrid| {
                self.backend.step_impl(subgrid);
            });
        });
    }
}

/// Things that can go wrong when performing parallel simulation
#[derive(Debug, Error)]
pub enum Error<BackendError: std::error::Error, ConcentrationError: std::error::Error> {
    /// Error from the underlying compute backend
    #[error(transparent)]
    Backend(BackendError),

    /// Error from hwloc
    #[error("failed to query hardware topology")]
    Hwloc(RawHwlocError),

    /// Failed to configure thread pool
    #[error("failed to configure thread pool")]
    ThreadPool(ThreadPoolBuildError),

    /// Error from the Concentration implementation
    ///
    /// In an ideal world, this error kind wouldn't be needed, as Backend can
    /// cover this case. But Rust is not yet smart enough to treat From as
    /// a transitive operation (if `T: From<U>` and `U: From<V>`, we do not yet
    /// get `T: From<V>` for free).
    #[doc(hidden)]
    #[error(transparent)]
    Concentration(#[from] ConcentrationError),
}
