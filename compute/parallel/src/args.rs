//! CLI arguments

use clap::Args;
use std::num::NonZeroUsize;

/// CLI parameters for the multithreaded implementation
#[derive(Args, Copy, Clone, Debug, Eq, Hash, PartialEq)]
pub struct ParallelArgs<BackendArgs: Args> {
    /// Number of processing threads
    #[arg(short = 'j', long, env)]
    pub(crate) num_threads: Option<NonZeroUsize>,

    /// Number of processed bytes per parallel task
    ///
    /// There is a granularity compromise between exposing opportunities for
    /// parallelism and keeping individual sequential tasks efficient. This
    /// block size is the tuning knob that lets you fine-tune this compromise.
    ///
    /// On x86 CPUs, the suggested tuning range is from half the per-thread L1
    /// cache capacity to the per-thread L3 cache capacity. By default, we tune
    /// to the per-thread L2 capacity.
    #[arg(long, env)]
    pub(crate) seq_block_size: Option<NonZeroUsize>,

    /// Expose backend arguments too
    #[command(flatten)]
    pub(crate) backend: BackendArgs,
}
