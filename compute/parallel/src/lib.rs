//! Parallel implementation of Gray-Scott simulation
//!
//! This crates implements a parallel version of the Gray-Scott simulation based
//! on domain decomposition and fork-join parallelism.

use compute::{Simulate, SimulateCpu, SimulationGrid};
use compute_block::{BlockSizeSelector, SingleCore};
use data::{concentration::Species, parameters::Parameters};
use hwlocality::Topology;
use rayon::prelude::*;

/// Gray-Scott reaction simulation
// TODO: Add cache blocking with MultiCore outer granularity and SingleCore
//       inner granularity, study its effect
pub type Simulation = ParallelSimulation<compute_autovec::Simulation>;

/// Gray-Scott simulation wrapper that enforces parallel iteration
pub struct ParallelSimulation<Backend: SimulateCpu + Sync>
where
    Backend::Values: Send + Sync,
{
    /// Number of grid elements (scalars or SIMD blocks) below which
    /// parallelism is not considered worthwhile
    sequential_len_threshold: usize,

    /// Underlying sequential compute backend
    backend: Backend,
}
//
impl<Backend: SimulateCpu + Sync> Simulate for ParallelSimulation<Backend>
where
    Backend::Values: Send + Sync,
{
    type Concentration = <Backend as Simulate>::Concentration;

    fn new(params: Parameters) -> Self {
        let topology = Topology::new().expect("Failed to probe hwloc topology");
        let sequential_len_threshold =
            SingleCore::block_size(&topology) / std::mem::size_of::<Backend::Values>();
        Self {
            sequential_len_threshold,
            backend: Backend::new(params),
        }
    }

    fn step(&self, species: &mut Species<Self::Concentration>) {
        self.step_impl(Self::extract_grid(species));
    }
}
//
impl<Backend: SimulateCpu + Sync> SimulateCpu for ParallelSimulation<Backend>
where
    Backend::Values: Send + Sync,
{
    type Values = Backend::Values;

    fn extract_grid(species: &mut Species<Self::Concentration>) -> SimulationGrid<Self::Values> {
        Backend::extract_grid(species)
    }

    fn unchecked_step_impl(&self, grid: SimulationGrid<Self::Values>) {
        rayon::iter::split(grid, |subgrid| {
            if Self::grid_len(&subgrid) <= self.sequential_len_threshold {
                (subgrid, None)
            } else {
                let [half1, half2] = Self::split_grid(subgrid);
                (half1, Some(half2))
            }
        })
        .for_each(|subgrid| {
            self.backend.step_impl(subgrid);
        });
    }
}

/// Multi-core block size selection policy
///
/// Multi-core computations should process data in blocks that fit in the sum of
/// the last-level caches of all cores, and that's what this policy enforces.
pub struct MultiCore;
//
impl BlockSizeSelector for MultiCore {
    fn block_size(topology: &Topology) -> usize {
        *topology
            .cpu_cache_stats()
            .total_data_cache_sizes()
            .last()
            .unwrap() as usize
    }
}
