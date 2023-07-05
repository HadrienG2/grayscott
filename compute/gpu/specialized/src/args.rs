//! CLI parameters

use clap::Args;
use std::num::NonZeroU32;

/// CLI parameters of the specialized GPU compute backend
#[derive(Args, Copy, Clone, Debug, Eq, Hash, PartialEq)]
pub struct GpuSpecializedArgs {
    /// Number of rows processed by each GPU work group during simulation
    ///
    /// The number of simulated rows must be a multiple of this.
    #[arg(long, env, default_value_t = NonZeroU32::new(8).unwrap())]
    pub(crate) compute_work_group_rows: NonZeroU32,

    /// Number of columns processed by each GPU work group during simulation
    ///
    /// The number of simulated columns must be a multiple of this.
    #[arg(long, env, default_value_t = NonZeroU32::new(8).unwrap())]
    pub(crate) compute_work_group_cols: NonZeroU32,
}
