//! Cache blocking implementation of Gray-Scott simulation
//!
//! This demonstrates how the simulation grid can be sliced into blocks in order
//! to improve cache locality.

use clap::Args;
use compute::{
    cpu::{CpuGrid, SimulateCpu},
    SimulateBase, SimulateCreate,
};
use data::{
    concentration::{Concentration, Species},
    parameters::Parameters,
};
use hwlocality::{errors::RawHwlocError, Topology};
use std::{marker::PhantomData, num::NonZeroUsize};
use thiserror::Error;

/// Gray-Scott reaction simulation
pub type Simulation = BlockWiseSimulation<compute_autovec::Simulation, SingleCore>;

/// Parameters are tunable via CLI args and environment variables
#[derive(Args, Copy, Clone, Debug, Eq, Hash, PartialEq)]
pub struct CliArgs<BackendArgs: Args> {
    /// Block size in bytes
    #[arg(long, env)]
    block_size: Option<NonZeroUsize>,

    /// Expose backend arguments too
    #[command(flatten)]
    backend: BackendArgs,
}

/// Gray-Scott simulation wrapper that enforces block-wise iteration
#[derive(Debug)]
pub struct BlockWiseSimulation<Backend: SimulateCpu, BlockSize: DefaultBlockSize> {
    /// Maximal number of grid elements (scalars or SIMD blocks) to be
    /// manipulated in one processing batch for optimal cache locality
    max_values_per_block: usize,

    /// Underlying compute backend
    backend: Backend,

    /// Block size selector
    block_size: PhantomData<BlockSize>,
}
//
impl<Backend: SimulateCpu, BlockSize: DefaultBlockSize> SimulateBase
    for BlockWiseSimulation<Backend, BlockSize>
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
impl<Backend: SimulateCpu, BlockSize: DefaultBlockSize> SimulateCreate
    for BlockWiseSimulation<Backend, BlockSize>
{
    fn new(params: Parameters, args: Self::CliArgs) -> Result<Self, Self::Error> {
        // Determine the desired block size in bytes
        let max_bytes_per_block = args
            .block_size
            .map(|bs| Ok::<_, Self::Error>(usize::from(bs)))
            .unwrap_or_else(|| {
                let topology = Topology::new().map_err(Error::Hwloc)?;
                Ok(BlockSize::block_size(&topology))
            })?;

        // Translate that into a number of SIMD vectors
        Ok(Self {
            max_values_per_block: max_bytes_per_block / std::mem::size_of::<Backend::Values>(),
            backend: Backend::new(params, args.backend).map_err(Error::Backend)?,
            block_size: PhantomData,
        })
    }
}
//
impl<Backend: SimulateCpu, BlockSize: DefaultBlockSize> SimulateCpu
    for BlockWiseSimulation<Backend, BlockSize>
{
    type Values = Backend::Values;

    fn extract_grid(species: &mut Species<Self::Concentration>) -> CpuGrid<Self::Values> {
        Backend::extract_grid(species)
    }

    fn unchecked_step_impl(&self, grid: CpuGrid<Self::Values>) {
        // Is the current grid fragment small enough?
        if Self::grid_len(&grid) < self.max_values_per_block {
            // If so, process it as is
            self.backend.step_impl(grid);
        } else {
            // Otherwise, split it and process the two halves sequentially
            for half_grid in Self::split_grid(grid) {
                self.step_impl(half_grid)
            }
        }
    }
}

/// Default block size selection policy, in absence of explicit user setting
pub trait DefaultBlockSize {
    /// Knowing the hardware topology, pick a good block size for the
    /// computation of interest (results are in bytes)
    fn block_size(topology: &Topology) -> usize;
}

/// Single-core block size selection policy
///
/// Single-core computations should process data in blocks that fit in the L1
/// cache of any single core, and that's what this policy enforces.
#[derive(Debug)]
pub struct SingleCore;
//
impl DefaultBlockSize for SingleCore {
    fn block_size(topology: &Topology) -> usize {
        topology.cpu_cache_stats().smallest_data_cache_sizes()[0] as usize
    }
}

/// Things that can go wrong when performing block-wise simulation
#[derive(Clone, Debug, Error)]
pub enum Error<BackendError: std::error::Error, ConcentrationError: std::error::Error> {
    /// Error from the underlying compute backend
    #[error(transparent)]
    Backend(BackendError),

    /// Error from hwloc
    #[error("failed to query hardware topology")]
    Hwloc(RawHwlocError),

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
