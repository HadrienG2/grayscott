//! Cache blocking implementation of Gray-Scott simulation
//!
//! This demonstrates how the simulation grid can be sliced into blocks in order
//! to improve cache locality.

mod args;
pub mod default;

use self::{
    args::BlockArgs,
    default::{DefaultBlockSize, SingleCore},
};
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

/// Gray-Scott simulation wrapper that enforces block-wise iteration
#[derive(Debug)]
pub struct BlockWiseSimulation<Backend: SimulateCpu, BlockSize: DefaultBlockSize> {
    /// Underlying compute backend
    backend: Backend,

    /// Maximal number of values processed per line of the simulation grid
    max_values_per_line: usize,

    /// Maximal number of values processed per recursively split grid block
    max_values_per_block: usize,

    /// Block size selector
    block_size: PhantomData<BlockSize>,
}
//
impl<Backend: SimulateCpu, BlockSize: DefaultBlockSize> SimulateBase
    for BlockWiseSimulation<Backend, BlockSize>
{
    type CliArgs = BlockArgs<Backend::CliArgs>;

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
        // Determine the desired block sizes in unit of processed values
        let topology = Topology::new().map_err(Error::Hwloc)?;
        let block_values = |arg: Option<NonZeroUsize>, default| {
            arg.map(usize::from).unwrap_or(default) / std::mem::size_of::<Backend::Values>()
        };
        let defaults = BlockSize::new(&topology);
        let max_values_per_line = block_values(args.l1_block_size, defaults.l1_block_size());
        let max_values_per_block_pair = block_values(args.l2_block_size, defaults.l2_block_size());
        let max_values_per_block = max_values_per_block_pair / 2;

        // Translate that into a number of SIMD vectors
        Ok(Self {
            max_values_per_line,
            max_values_per_block,
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
        let line_len_ok = Self::grid_line_len(&grid) <= self.max_values_per_line;
        if line_len_ok {
            // Hand over sufficiently small blocks to the backend
            self.backend.step_impl(grid);
        } else if Self::grid_len(&grid) > self.max_values_per_block {
            // Recursively split simulation grid by longest axis (this maximizes
            // the number of boundary elements that will be hot in cache when
            // switching from the first half to the second half) until we
            // reach the desired L2 block size.
            for half_grid in Self::split_grid(grid, None) {
                self.step_impl(half_grid)
            }
        } else {
            // Then, if needed, recursively split grid lines for L1 locality
            for half_grid in Self::split_grid(grid, Some(1)) {
                self.step_impl(half_grid)
            }
        }
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
