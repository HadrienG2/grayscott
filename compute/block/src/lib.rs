//! Cache blocking implementation of Gray-Scott simulation
//!
//! The `autovec` and `manualvec` versions are actually not compute bound but
//! memory bound. This version uses cache blocking techniques to improve the CPU
//! cache hit rate, getting us back into compute-bound territory.

use autovec::Values;
use compute::Simulate;
use data::{
    concentration::Species,
    parameters::{stencil_offset, Parameters},
};
use hwloc2::{ObjectType, Topology, TopologyObject};
use ndarray::{s, ArrayView2, ArrayViewMut2, Axis};

/// Gray-Scott reaction simulation
pub struct Simulation {
    /// Maximal number of SIMD vectors in one processing batch
    max_block_len: usize,

    /// Simulation parameters
    params: Parameters,
}
//
impl Simulate for Simulation {
    type Concentration = <autovec::Simulation as Simulate>::Concentration;

    fn new(params: Parameters) -> Self {
        let topology = Topology::new().expect("Failed to probe hwloc topology");
        Self {
            max_block_len: l1_block_len(&topology),
            params,
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
impl Simulation {
    fn step_impl(
        &self,
        [in_u, in_v]: [ArrayView2<Values>; 2],
        [out_u, out_v]: [ArrayViewMut2<Values>; 2],
    ) {
        // Check that everything is alright in debug mode
        debug_assert_eq!(in_u.shape(), in_v.shape());
        debug_assert_eq!(out_u.shape(), out_v.shape());
        let stencil_offset = stencil_offset();
        debug_assert_eq!(out_u.nrows(), in_u.nrows() + 2 * stencil_offset[0]);
        debug_assert_eq!(out_u.ncols(), in_u.ncols() + 2 * stencil_offset[1]);

        // If the problem has become small enough for the cache, run it
        if 2 * (in_u.len() + out_u.len()) < self.max_block_len {
            autovec::step_impl([in_u, in_v], [out_u, out_v], &self.params);
            return;
        }

        // Otherwise, split the problem in two across its longest dimension
        // and process the two halves.
        if out_u.nrows() > out_u.ncols() {
            // Splitting the output slice is easy
            let out_split_point = out_u.nrows() / 2;
            let (out_u_1, out_u_2) = out_u.split_at(Axis(0), out_split_point);
            let (out_v_1, out_v_2) = out_v.split_at(Axis(0), out_split_point);

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
            let out_split_point = out_u.ncols() / 2;
            let (out_u_1, out_u_2) = out_u.split_at(Axis(1), out_split_point);
            let (out_v_1, out_v_2) = out_v.split_at(Axis(1), out_split_point);

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

/// Determine the maximal block size that provides optimal cache locality in
/// a single-threaded workflow.
pub fn l1_block_len(topology: &Topology) -> usize {
    // Use hwloc to find the smallest L1 cache on this system
    fn min_l1_size(object: &TopologyObject) -> Option<usize> {
        // If this is an L1 cache, return how big it is
        if object.object_type() == ObjectType::L1Cache {
            // FIXME: Due to a bug in hwloc2-rs, must parse object display :'(
            let display = object.to_string();
            let trailer = display.strip_prefix("L1d (").unwrap();
            let capacity_kb = trailer.strip_suffix("KB)").unwrap();
            let capacity = capacity_kb.parse::<usize>().unwrap() * 1024;
            return Some(capacity);
        }

        // Otherwise, look at children caches
        let mut child_opt = object.first_child();
        let mut cache_size: Option<usize> = None;
        while let Some(child) = child_opt {
            match (cache_size, min_l1_size(child)) {
                (Some(c1), Some(c2)) => cache_size = Some(c1.min(c2)),
                (None, Some(c2)) => cache_size = Some(c2),
                (Some(_), None) | (None, None) => {}
            }
            child_opt = child.next_sibling();
        }
        cache_size
    }
    let min_l1_size =
        min_l1_size(topology.object_at_root()).expect("Failed to probe smallest cache size");

    // Translate cache size into a block length in units of SIMD elements
    min_l1_size / std::mem::size_of::<Values>()
}
