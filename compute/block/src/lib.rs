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
    /// Level 1 block size in bytes
    ///
    /// Owing to the stencil nature of the computation, in order to generate one
    /// line of the output grid, we need to read three lines of the input grid
    /// (matching line, line before, line after).
    ///
    /// The following ASCII art demonstrates the naive input data access
    /// pattern without any cache blocking:
    ///
    /// ```text
    /// Input:                         Output:
    /// ------------------------       ------------------------
    /// ==###===================       ------------------------
    /// ==###===================       ===#====================
    /// ==###===================       ------------------------
    /// ------------------------       ------------------------
    /// ------------------------       ------------------------
    /// ------------------------       ------------------------
    /// ------------------------       ------------------------
    /// ```
    ///
    /// Here's what the various symbols represent:
    ///
    /// - `#` represents data which is directly being accessed over the course
    ///   of processing one output data point
    /// - `=` represents data which is being accessed over the course of
    ///   processing one line of output data points
    /// - `-` represents other data which is accessed before and after the
    ///   active lines of inputs/outputs.
    ///
    /// This CLI parameter controls a first layer of cache blocking which cuts
    /// lines in segments to ensure that three consecutive segments fit in
    /// cache. This way, by the time we start processing a new line, the input
    /// line before will still resident in cache.
    ///
    /// If we applied only this layer of cache blocking, the data access
    /// pattern would become this...
    ///
    /// ```text
    /// Input:                          Output
    /// --------................        --------................
    /// ==###===................        --------................
    /// ==###===................        ===#====................
    /// ==###===................        --------................
    /// --------................        --------................
    /// --------................        --------................
    /// --------................        --------................
    /// --------................        --------................
    /// ```
    ///
    /// ...where data is now accessed in columnar blocks. In this new schematic,
    /// `-` represents of data line segments that will be accessed before or
    /// after the active segment inside of the active columnar block, and `.`
    /// represents data points which will be accessed later on.
    ///
    /// By default, the block size is tuned to the size of the CPU's L1 cache.
    /// Lines will be cut so that that 3 line segments fit into the block size.
    #[arg(long, env)]
    l1_block_size: Option<NonZeroUsize>,

    /// Level 2 block size in bytes
    ///
    /// The first layer of cache blocking cuts the grid into columnar blocks,
    /// which ensures cache locality between consecutive data lines. However,
    /// it does not ensure cache locality is not guaranteed between consecutive
    /// data columns: by the time a new columnar data block starts being
    /// processed, all memory accesses to the line of input data on the left of
    /// the column may cache-miss all the way to RAM.
    ///
    /// To address this, a second layer of cache blocking is used, which further
    /// subdivides the columnar blocks produced by L1 blocking into rectangles:
    ///
    /// ```text
    /// Input:                          Output
    /// --------................        --------................
    /// ==###===................        --------................
    /// ==###===................        ===#====................
    /// ==###===................        --------................
    /// --------................        --------................
    /// --------................        --------................
    /// ........................        ........................
    /// ........................        ........................
    /// ```
    ///
    /// These rectangular blocks are then accessed along a fractal space-filling
    /// curve, ensuring that elements on the boundary between two consecutive
    /// blocks fit in in the highest level of cache possible. Here is an
    /// example of block access order with the Z-shaped Morton curve:
    ///
    /// ```text
    ///  1  2  5  6
    ///  3  4  7  8
    ///  9 10 13 14
    /// 11 12 15 16
    /// ```
    ///
    /// By default, the block size is tuned to the size of the CPU's L2 cache.
    /// Rectangular blocks will be cut to ensure that 2 consecutive blocks fit
    /// into the block size.
    #[arg(long, env)]
    l2_block_size: Option<NonZeroUsize>,

    /// Expose backend arguments too
    #[command(flatten)]
    backend: BackendArgs,
}

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
        // Determine the desired block sizes in unit of processed values
        let topology = Topology::new().map_err(Error::Hwloc)?;
        let block_values = |arg: Option<NonZeroUsize>, default| {
            arg.map(usize::from).unwrap_or(default) / std::mem::size_of::<Backend::Values>()
        };
        let defaults = BlockSize::new(&topology);
        let max_values_per_line = block_values(args.l1_block_size, defaults.l1_block_size());
        let l2_block_values = block_values(args.l2_block_size, defaults.l2_block_size());

        // Translate that into a number of SIMD vectors
        Ok(Self {
            max_values_per_line,
            // 2 blocks must fit in L2 cache for good cache locality
            max_values_per_block: l2_block_values / 2,
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
        if Self::grid_len(&grid) > self.max_values_per_block {
            // Recursively split simulation grid until L2 locality is achieved
            for half_grid in Self::split_grid(grid, None) {
                self.step_impl(half_grid)
            }
        } else if Self::grid_line_len(&grid) > self.max_values_per_line {
            // Recursively split grid lines until L1 locality is achieved
            for half_grid in Self::split_grid(grid, Some(1)) {
                self.step_impl(half_grid)
            }
        } else {
            // Hand over sufficiently small blocks to the backend
            self.backend.step_impl(grid);
        }
    }
}

/// Default block size selection policy, in absence of explicit user setting
pub trait DefaultBlockSize {
    /// Acquire required data from the hwloc topology
    fn new(topology: &Topology) -> Self;

    /// Suggested level 1 block size in bytes
    fn l1_block_size(&self) -> usize;

    /// Suggested level 2 block size in bytes
    fn l2_block_size(&self) -> usize;
}

/// Single-core block size selection policy
///
/// Single-core computations can use the full L1 and L2 cache and need not
/// concern themselves with another hyperthread using part of it.
#[derive(Debug)]
pub struct SingleCore {
    /// Level 1 block size in bytes
    l1_block_size: usize,

    /// Level 2 block size in bytes
    l2_block_size: usize,
}
//
impl DefaultBlockSize for SingleCore {
    fn new(topology: &Topology) -> Self {
        let cache_stats = topology.cpu_cache_stats();
        let cache_sizes = cache_stats.smallest_data_cache_sizes();

        let l1_block_size = cache_sizes.get(0).copied().unwrap_or(32 * 1024) as usize;
        let l2_block_size = if cache_sizes.len() > 1 {
            cache_sizes[1] as usize
        } else {
            l1_block_size
        };

        Self {
            l1_block_size,
            l2_block_size,
        }
    }

    fn l1_block_size(&self) -> usize {
        self.l1_block_size
    }

    fn l2_block_size(&self) -> usize {
        self.l2_block_size
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
