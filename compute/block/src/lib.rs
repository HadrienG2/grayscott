//! Cache blocking implementation of Gray-Scott simulation
//!
//! The `autovec` and `manualvec` versions are actually not compute bound but
//! memory bound. This version uses cache blocking techniques to improve the CPU
//! cache hit rate, getting us back into compute-bound territory.

use std::marker::PhantomData;

use compute::{Simulate, SimulateImpl};
use data::{concentration::Species, parameters::Parameters};
use hwlocality::Topology;
use ndarray::{ArrayView2, ArrayViewMut2};

/// Gray-Scott reaction simulation
pub struct Simulation<
    Backend: SimulateImpl = compute_autovec::Simulation,
    BlockSize: BlockSizeSelector = SingleCore,
> {
    /// Maximal number of grid elements (scalars or SIMD blocks) to be
    /// manipulated in one processing batch for optimal cache locality
    max_values_per_block: usize,

    /// Underlying compute backend
    backend: Backend,

    /// Block size selector
    block_size: PhantomData<BlockSize>,
}
//
impl<Backend: SimulateImpl, BlockSize: BlockSizeSelector> Simulate
    for Simulation<Backend, BlockSize>
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
        let (in_u_v, out_u_v) = Self::step_impl_input(species);
        self.step_impl(in_u_v, out_u_v);
    }
}
//
impl<Backend: SimulateImpl, BlockSize: BlockSizeSelector> SimulateImpl
    for Simulation<Backend, BlockSize>
{
    type Values = Backend::Values;

    fn step_impl_input(
        species: &mut Species<Self::Concentration>,
    ) -> (
        [ArrayView2<Self::Values>; 2],
        [ArrayViewMut2<Self::Values>; 2],
    ) {
        Backend::step_impl_input(species)
    }

    fn unchecked_step_impl(
        &self,
        [in_u, in_v]: [ArrayView2<Self::Values>; 2],
        [out_u_center, out_v_center]: [ArrayViewMut2<Self::Values>; 2],
    ) {
        // Is the current grid fragment small enough?
        if 2 * (in_u.len() + out_u_center.len()) < self.max_values_per_block {
            // If so, process it as is
            self.backend
                .step_impl([in_u, in_v], [out_u_center, out_v_center]);
        } else {
            // Otherwise, split it and process the two halves sequentially
            for (in_u_v, out_u_v_center) in
                Self::split_grid([in_u, in_v], [out_u_center, out_v_center])
            {
                self.step_impl(in_u_v, out_u_v_center)
            }
        }
    }
}

/// Block size selection policy
pub trait BlockSizeSelector {
    /// Knowing the hardware topology, pick a good block size for the
    /// computation of interest
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
