//! Cache blocking implementation of Gray-Scott simulation
//!
//! The `autovec` and `manualvec` versions are actually not compute bound but
//! memory bound. This version uses cache blocking techniques to improve the CPU
//! cache hit rate, getting us back into compute-bound territory.

use compute::{CpuGrid, SimulateBase, SimulateCpu};
use data::{
    concentration::{Concentration, Species},
    parameters::Parameters,
};
use hwlocality::{errors::RawHwlocError, Topology};
use std::marker::PhantomData;
use thiserror::Error;

/// Gray-Scott reaction simulation
pub type Simulation = BlockWiseSimulation<compute_autovec::Simulation, SingleCore>;

/// Gray-Scott simulation wrapper that enforces block-wise iteration
pub struct BlockWiseSimulation<Backend: SimulateCpu, BlockSize: BlockSizeSelector> {
    /// Maximal number of grid elements (scalars or SIMD blocks) to be
    /// manipulated in one processing batch for optimal cache locality
    max_values_per_block: usize,

    /// Underlying compute backend
    backend: Backend,

    /// Block size selector
    block_size: PhantomData<BlockSize>,
}
//
impl<Backend: SimulateCpu, BlockSize: BlockSizeSelector> SimulateBase
    for BlockWiseSimulation<Backend, BlockSize>
{
    type Concentration = <Backend as SimulateBase>::Concentration;

    type Error =
        Error<<Backend as SimulateBase>::Error, <Self::Concentration as Concentration>::Error>;

    fn new(params: Parameters) -> Result<Self, Self::Error> {
        // Determine the desired block size in bytes
        let topology = Topology::new().map_err(Error::Hwloc)?;
        let max_bytes_per_block = BlockSize::block_size(&topology);

        // Translate that into a number of SIMD vectors
        Ok(Self {
            max_values_per_block: max_bytes_per_block / std::mem::size_of::<Backend::Values>(),
            backend: Backend::new(params).map_err(Error::Backend)?,
            block_size: PhantomData,
        })
    }

    fn make_species(&self, shape: [usize; 2]) -> Result<Species<Self::Concentration>, Self::Error> {
        self.backend.make_species(shape).map_err(Error::Backend)
    }
}
//
impl<Backend: SimulateCpu, BlockSize: BlockSizeSelector> SimulateCpu
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

/// Block size selection policy
pub trait BlockSizeSelector {
    /// Knowing the hardware topology, pick a good block size for the
    /// computation of interest (results are in bytes)
    fn block_size(topology: &Topology) -> usize;
}

/// Single-core block size selection policy
///
/// Single-core computations should process data in blocks that fit in the L1
/// cache of any single core, and that's what this policy enforces.
pub struct SingleCore;
//
impl BlockSizeSelector for SingleCore {
    fn block_size(topology: &Topology) -> usize {
        topology.cpu_cache_stats().smallest_data_cache_sizes()[0] as usize
    }
}

/// Things that can go wrong when performing block-wise simulation
#[derive(Clone, Debug, Error)]
pub enum Error<BackendError: std::error::Error, ConcentrationError: std::error::Error> {
    /// Error from the backend
    #[error(transparent)]
    Backend(BackendError),

    /// Error from hwloc
    #[error("failed to query hardware topology")]
    Hwloc(RawHwlocError),

    /// Error from the Concentration implementation
    ///
    /// In an ideal world, this error kind wouldn't be needed, as Backend can
    /// cover this case. But Rust is not yet smart enough to treat From as
    /// a transitive operation (if T: From<U> and U: From<V>, then T: From<V>).
    #[doc(hidden)]
    #[error(transparent)]
    Concentration(#[from] ConcentrationError),
}
