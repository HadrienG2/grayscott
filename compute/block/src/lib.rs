//! Cache blocking implementation of Gray-Scott simulation
//!
//! The `autovec` and `manualvec` versions are actually not compute bound but
//! memory bound. This version uses cache blocking techniques to improve the CPU
//! cache hit rate, getting us back into compute-bound territory.

use compute::{Simulate, SimulateImpl};
use data::{concentration::Species, parameters::Parameters};
use hwlocality::Topology;
use ndarray::{ArrayView2, ArrayViewMut2};

// SIMD compute backend (can be compute_autovec or compute_manualvec)
use compute_autovec as compute_backend;
use compute_backend::Values;

/// Gray-Scott reaction simulation
pub struct Simulation {
    /// Maximal number of SIMD vectors to be manipulated in one processing batch
    max_vecs_per_block: usize,

    /// SIMD compute backend
    backend: compute_backend::Simulation,
}
//
impl Simulate for Simulation {
    type Concentration = <compute_backend::Simulation as Simulate>::Concentration;

    fn new(params: Parameters) -> Self {
        // Check minimal CPU L1 cache size in bytes
        let topology = Topology::new().expect("Failed to probe hwloc topology");
        let min_l1_size = topology.cpu_cache_stats().smallest_data_cache_sizes()[0] as usize;

        // Translate that into a number of SIMD vectors
        Self {
            max_vecs_per_block: min_l1_size / std::mem::size_of::<Values>(),
            backend: compute_backend::Simulation::new(params),
        }
    }

    fn step(&self, species: &mut Species<Self::Concentration>) {
        let (in_u, out_u) = species.u.in_out();
        let (in_v, out_v) = species.v.in_out();
        self.step_impl(
            [in_u.view(), in_v.view()],
            [out_u.simd_center_mut(), out_v.simd_center_mut()],
        );
    }
}
//
impl SimulateImpl for Simulation {
    type Values = Values;

    fn unchecked_step_impl(
        &self,
        [in_u, in_v]: [ArrayView2<Values>; 2],
        [out_u_center, out_v_center]: [ArrayViewMut2<Values>; 2],
    ) {
        // Is the current grid fragment small enough to fit in L1 cache ?
        if 2 * (in_u.len() + out_u_center.len()) < self.max_vecs_per_block {
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
