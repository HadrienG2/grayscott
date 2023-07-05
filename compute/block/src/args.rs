//! Command-line arguments

use clap::Args;
use std::num::NonZeroUsize;

/// CLI parameters for the cache blocking implementation
#[derive(Args, Copy, Clone, Debug, Eq, Hash, PartialEq)]
pub struct BlockArgs<BackendArgs: Args> {
    /// Level 1 block size in bytes
    ///
    /// Owing to the stencil nature of the computation, in order to generate one
    /// line of the output grid, we need to read three lines of the input grid
    /// (matching line, line before, line after).
    ///
    // The following ASCII art demonstrates the naive input data access
    // pattern without any cache blocking:
    //
    // ```text
    // Input:                         Output:
    // ------------------------       ------------------------
    // ==###===================       ------------------------
    // ==###===================       ===#====================
    // ==###===================       ------------------------
    // ------------------------       ------------------------
    // ------------------------       ------------------------
    // ------------------------       ------------------------
    // ------------------------       ------------------------
    // ```
    //
    // Here's what the various symbols represent:
    //
    // - `#` represents data which is directly being accessed over the course
    //   of processing one output data point
    // - `=` represents data which is being accessed over the course of
    //   processing one line of output data points
    // - `-` represents other data which is accessed before and after the
    //   active lines of inputs/outputs.
    //
    /// This CLI parameter controls a first layer of cache blocking which cuts
    /// lines in segments to ensure that three consecutive input segments and
    /// one output segment fit in cache. This way, by the time we start
    /// processing a new line, the input line before should still fit in cache.
    ///
    // If we applied only this layer of cache blocking, the data access
    // pattern would become this...
    //
    // ```text
    // Input:                          Output
    // --------................        --------................
    // ==###===................        --------................
    // ==###===................        ===#====................
    // ==###===................        --------................
    // --------................        --------................
    // --------................        --------................
    // --------................        --------................
    // --------................        --------................
    // ```
    //
    // ...where data is now accessed in columnar blocks. In this new schematic,
    // `-` represents of data line segments that will be accessed before or
    // after the active segment inside of the active columnar block, and `.`
    // represents data points which will be accessed later on.
    //
    /// By default, the block size is tuned to half the CPU's L1 cache size.
    #[arg(long, env)]
    pub(crate) l1_block_size: Option<NonZeroUsize>,

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
    // ```text
    // Input:                          Output
    // --------................        --------................
    // ==###===................        --------................
    // ==###===................        ===#====................
    // ==###===................        --------................
    // --------................        --------................
    // --------................        --------................
    // ........................        ........................
    // ........................        ........................
    // ```
    //
    /// These rectangular blocks are then accessed along a fractal space-filling
    /// curve, ensuring that elements on the boundary between two consecutive
    /// blocks fit in in the highest level of cache possible.
    ///
    // Here is an example of block access order with the Z-shaped Morton
    // curve:
    //
    // ```text
    //  1  2  5  6
    //  3  4  7  8
    //  9 10 13 14
    // 11 12 15 16
    // ```
    //
    /// By default, the block size is tuned to half the CPU's L2 cache size.
    #[arg(long, env)]
    pub(crate) l2_block_size: Option<NonZeroUsize>,

    /// Expose backend arguments too
    #[command(flatten)]
    pub(crate) backend: BackendArgs,
}
