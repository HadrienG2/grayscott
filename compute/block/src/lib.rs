//! Cache blocking implementation of Gray-Scott simulation
//!
//! The `autovec` and `manualvec` versions are actually not compute bound but
//! memory bound. This version uses cache blocking techniques to improve the CPU
//! cache hit rate, getting us back into compute-bound territory.

use compute::{Simulate, SimulateImpl};
use data::{
    concentration::Species,
    parameters::{stencil_offset, Parameters},
};
use hwlocality::Topology;
use ndarray::{s, ArrayView2, ArrayViewMut2, Axis};

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

    #[inline]
    fn unchecked_step_impl(
        &self,
        [in_u, in_v]: [ArrayView2<Values>; 2],
        [out_u_center, out_v_center]: [ArrayViewMut2<Values>; 2],
    ) {
        // If the problem has become small enough for the cache, run it
        if 2 * (in_u.len() + out_u_center.len()) < self.max_vecs_per_block {
            self.backend
                .step_impl([in_u, in_v], [out_u_center, out_v_center]);
            return;
        }

        // Otherwise, split the problem in two across its longest dimension
        // and process the two halves.
        let stencil_offset = stencil_offset();
        if out_u_center.nrows() > out_u_center.ncols() {
            // Splitting the output slice is easy
            let out_split_point = out_u_center.nrows() / 2;
            let (out_u_1, out_u_2) = out_u_center.split_at(Axis(0), out_split_point);
            let (out_v_1, out_v_2) = out_v_center.split_at(Axis(0), out_split_point);

            // On the input side, we must mind the edge elements
            let in_split_point = out_split_point + stencil_offset[0];
            let in_slice_1 = s![..in_split_point + stencil_offset[0], ..];
            let in_u_1 = in_u.slice(in_slice_1);
            let in_v_1 = in_v.slice(in_slice_1);
            self.step_impl([in_u_1, in_v_1], [out_u_1, out_v_1]);
            //
            let in_slice_2 = s![in_split_point - stencil_offset[0].., ..];
            let in_u_2 = in_u.slice(in_slice_2);
            let in_v_2 = in_v.slice(in_slice_2);
            self.step_impl([in_u_2, in_v_2], [out_u_2, out_v_2]);
        } else {
            // Splitting the output slice is easy
            let out_split_point = out_u_center.ncols() / 2;
            let (out_u_1, out_u_2) = out_u_center.split_at(Axis(1), out_split_point);
            let (out_v_1, out_v_2) = out_v_center.split_at(Axis(1), out_split_point);

            // On the input side, we must mind the edge elements
            let in_split_point = out_split_point + stencil_offset[1];
            let in_slice_1 = s![.., ..in_split_point + stencil_offset[1]];
            let in_u_1 = in_u.slice(in_slice_1);
            let in_v_1 = in_v.slice(in_slice_1);
            self.step_impl([in_u_1, in_v_1], [out_u_1, out_v_1]);
            //
            let in_slice_2 = s![.., in_split_point - stencil_offset[1]..];
            let in_u_2 = in_u.slice(in_slice_2);
            let in_v_2 = in_v.slice(in_slice_2);
            self.step_impl([in_u_2, in_v_2], [out_u_2, out_v_2]);
        }
    }
}
