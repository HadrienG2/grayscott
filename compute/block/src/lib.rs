//! Cache blocking implementation of Gray-Scott simulation
//!
//! The `autovec` and `manualvec` versions are actually not compute bound but
//! memory bound. This version uses cache blocking techniques to improve the CPU
//! cache hit rate, getting us back into compute-bound territory.

use compute::{Simulate, SimulateCpu, SimulationGrid};
use data::{concentration::Species, parameters::Parameters};
use hwlocality::Topology;
use std::marker::PhantomData;

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
impl<Backend: SimulateCpu, BlockSize: BlockSizeSelector> Simulate
    for BlockWiseSimulation<Backend, BlockSize>
{
    type Concentration = <Backend as Simulate>::Concentration;

    fn new(params: Parameters) -> Self {
        // Determine the desired block size in bytes
        let topology = Topology::new().expect("Failed to probe hwloc topology");
        let max_bytes_per_block = BlockSize::block_size(&topology);

        // Translate that into a number of SIMD vectors
        Self {
            max_values_per_block: max_bytes_per_block / std::mem::size_of::<Backend::Values>(),
            backend: Backend::new(params),
            block_size: PhantomData,
        }
    }

    fn step(&self, species: &mut Species<Self::Concentration>) {
        self.step_impl(Self::extract_grid(species));
    }
}
//
impl<Backend: SimulateCpu, BlockSize: BlockSizeSelector> SimulateCpu
    for BlockWiseSimulation<Backend, BlockSize>
{
    type Values = Backend::Values;

    fn extract_grid(species: &mut Species<Self::Concentration>) -> SimulationGrid<Self::Values> {
        Backend::extract_grid(species)
    }

    fn unchecked_step_impl(&self, grid: SimulationGrid<Self::Values>) {
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
