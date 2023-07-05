//! Block size policy of the multicore computation

use compute_block::default::DefaultBlockSize;
use hwlocality::Topology;

/// Multi-core block size selection policy
///
/// Multi-core computations should mind fact that in threads can share certain
/// cache levels, including L1 in the presence of SMT / Hyperthreading. Memory
/// budgets should be adjusted accordingly.
#[derive(Debug)]
pub struct MultiCore {
    /// Level 1 block size in bytes
    l1_block_size: usize,

    /// Level 2 block size in bytes
    l2_block_size: usize,
}
//
impl DefaultBlockSize for MultiCore {
    fn new(topology: &Topology) -> Self {
        let cache_stats = topology.cpu_cache_stats();
        let cache_sizes = cache_stats.smallest_data_cache_sizes_per_thread();

        let l1_block_size = cache_sizes.first().copied().unwrap_or(16 * 1024) as usize / 2;
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
