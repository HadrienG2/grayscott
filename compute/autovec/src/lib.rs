//! Auto-vectorized implementation of Gray-Scott simulation
//!
//! While compilers can automatically vectorize computations, said computations
//! must in all but simplest cases be shaped exactly like manually vectorized
//! code based on hardware intrinsics would be. This compute backend follows
//! this strategy, which should allow it to perform decently on hardware other
//! than the hardware it was written for (x86_64), with minimal porting effort
//! revolving around picking the right vector width.

use cfg_if::cfg_if;
use compute::{Simulate, SimulateImpl, SimulationGrid};
use data::{
    concentration::{simd::SIMDConcentration, Species},
    parameters::{stencil_offset, Parameters, STENCIL_SHAPE},
    Precision,
};
use slipstream::{vector::align, Vector};

// Pick vector size based on hardware support for vectorization of
// floating-point operations (which are the bulk of our SIMD workload)
const PRECISION_SIZE: usize = std::mem::size_of::<Precision>();
cfg_if! {
    if #[cfg(target_feature = "avx512f")] {
        pub const WIDTH: usize = 64 / PRECISION_SIZE;
        pub type Values = Vector<align::Align64, Precision, WIDTH>;
    } else if #[cfg(target_feature = "avx")] {
        pub const WIDTH: usize = 32 / PRECISION_SIZE;
        pub type Values = Vector<align::Align32, Precision, WIDTH>;
    } else {
        // NOTE: While most non-Intel CPUs use 128-bit vectorization, not all do.
        //       A benefit of autovectorization, however, is that supporting new
        //       hardware can just be a matter of adding cases in this cfg_if.
        pub const WIDTH: usize = 16 / PRECISION_SIZE;
        pub type Values = Vector<align::Align16, Precision, WIDTH>;
    }
}

// Use FMA if supported in hardware (unlike GCC, LLVM does not do it automatically)
cfg_if! {
    // NOTE: Extend this when porting to more CPU architectures
    if #[cfg(any(target_feature = "fma", target_feature = "vfp4"))] {
        #[inline(always)]
        pub fn mul_add(x: Values, y: Values, z: Values) -> Values {
            x.mul_add(y, z)
        }
    } else {
        #[inline(always)]
        pub fn mul_add(x: Values, y: Values, z: Values) -> Values {
            x * y + z
        }
   }
}

/// Gray-Scott reaction simulation
pub struct Simulation {
    /// Simulation parameters
    params: Parameters,
}
//
impl Simulate for Simulation {
    type Concentration = SIMDConcentration<WIDTH, Values>;

    fn new(params: Parameters) -> Self {
        Self { params }
    }

    fn step(&self, species: &mut Species<Self::Concentration>) {
        self.step_impl(Self::extract_grid(species));
    }
}
//
impl SimulateImpl for Simulation {
    type Values = Values;

    fn extract_grid(species: &mut Species<Self::Concentration>) -> SimulationGrid<Self::Values> {
        let (in_u, out_u) = species.u.in_out();
        let (in_v, out_v) = species.v.in_out();
        (
            [in_u.view(), in_v.view()],
            [out_u.simd_center_mut(), out_v.simd_center_mut()],
        )
    }

    fn unchecked_step_impl(
        &self,
        ([in_u, in_v], [mut out_u_center, mut out_v_center]): SimulationGrid<Self::Values>,
    ) {
        // Determine offset from the top-left corner of the stencil to its center
        let stencil_offset = stencil_offset();

        // Prepare vector versions of the scalar computation parameters
        let diffusion_rate_u = Values::splat(self.params.diffusion_rate_u);
        let diffusion_rate_v = Values::splat(self.params.diffusion_rate_v);
        let feed_rate = Values::splat(self.params.feed_rate);
        let kill_rate = Values::splat(self.params.kill_rate);
        let time_step = Values::splat(self.params.time_step);
        let ones = Values::splat(1.0);

        // Iterate over center pixels of the species concentration matrices
        for (((out_u, out_v), win_u), win_v) in (out_u_center.iter_mut())
            .zip(out_v_center.iter_mut())
            .zip(in_u.windows(STENCIL_SHAPE))
            .zip(in_v.windows(STENCIL_SHAPE))
        {
            // Access center value of u
            let u = win_u[stencil_offset];
            let v = win_v[stencil_offset];

            // Compute diffusion gradient
            let [full_u, full_v] = (win_u.iter())
                .zip(win_v.iter())
                .zip(
                    self.params
                        .weights
                        .into_iter()
                        .flat_map(|row| row.into_iter()),
                )
                .fold(
                    [Values::splat(0.); 2],
                    |[acc_u, acc_v], ((&stencil_u, &stencil_v), weight)| {
                        let weight = Values::splat(weight);
                        [
                            mul_add(weight, stencil_u - u, acc_u),
                            mul_add(weight, stencil_v - v, acc_v),
                        ]
                    },
                );

            // Deduce variation of U and V
            let uv_square = u * v * v;
            let du = mul_add(diffusion_rate_u, full_u, feed_rate * (ones - u) - uv_square);
            let dv = mul_add(
                diffusion_rate_v,
                full_v,
                uv_square - (feed_rate + kill_rate) * v,
            );
            *out_u = mul_add(du, time_step, u);
            *out_v = mul_add(dv, time_step, v);
        }
    }
}
