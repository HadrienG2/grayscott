//! Default block size selection policy

use hwlocality::Topology;

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

        let l1_block_size = cache_sizes.first().copied().unwrap_or(32 * 1024) as usize / 2;
        let l2_block_size = if cache_sizes.len() > 1 {
            cache_sizes[1] as usize / 2
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
