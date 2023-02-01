//! Auto-vectorized implementation of Gray-Scott simulation
//!
//! While compilers can automatically vectorize computations, said computations
//! must in all but simplest cases be shaped exactly like manually vectorized
//! code based on hardware intrinsics would be. This compute backend follows
//! this strategy, which should allow it to perform decently on hardware other
//! than the hardware it was written for (x86_64), with minimal porting effort
//! revolving around picking the right vector width.

use cfg_if::cfg_if;
use data::{
    concentration::simd::SIMDConcentration,
    parameters::{stencil_offset, Parameters},
    Precision,
};
use slipstream::{vector::align, Vector};

// Pick vector size based on hardware support for vectorization of
// floating-point operations (which are the bulk of our SIMD workload)
const PRECISION_SIZE: usize = std::mem::size_of::<Precision>();
cfg_if! {
    if #[cfg(target_feature = "avx512f")] {
        const WIDTH: usize = 64 / PRECISION_SIZE;
        type Values = Vector<align::Align64, Precision, WIDTH>;
    } else if #[cfg(target_feature = "avx")] {
        const WIDTH: usize = 32 / PRECISION_SIZE;
        type Values = Vector<align::Align32, Precision, WIDTH>;
    } else {
        // NOTE: While most non-Intel CPUs use 128-bit vectorization, not all do.
        //       A benefit of autovectorization, however, is that supporting new
        //       hardware can just be a matter of adding cases in this cfg_if.
        const WIDTH: usize = 16 / PRECISION_SIZE;
        type Values = Vector<align::Align16, Precision, WIDTH>;
    }
}

// Use FMA if supported in hardware (unlike GCC, LLVM does not do it automatically)
// NOTE: This is the other part that would need to change when porting to more HW
// FIXME: Currently disabled due to lack of slipstream support
/* cfg_if! {
if #[cfg(any(target_feature = "fma", target_feature = "vfp4"))] {
    #[inline(always)]
    fn mul_add(x: Values, y: Values, z: Values) -> Values {
        x.mul_add(y, z)
    }
} else { */
#[inline(always)]
fn mul_add(x: Values, y: Values, z: Values) -> Values {
    x * y + z
}
/*    }
} */

/// Chosen concentration type
pub type Species = data::concentration::Species<SIMDConcentration<WIDTH, Values>>;

/// Perform one simulation time step
pub fn step(species: &mut Species, params: &Parameters) {
    // Access species concentration matrices
    let (in_u, out_u) = species.u.in_out();
    let (in_v, out_v) = species.v.in_out();

    // Determine offset from the top-left corner of the stencil to its center
    let stencil_offset = stencil_offset();

    // Prepare vector versions of the scalar computation parameters
    let diffusion_rate_u = Values::splat(params.diffusion_rate_u);
    let diffusion_rate_v = Values::splat(params.diffusion_rate_v);
    let feed_rate = Values::splat(params.feed_rate);
    let kill_rate = Values::splat(params.kill_rate);
    let time_step = Values::splat(params.time_step);
    let ones = Values::splat(1.0);

    // Iterate over center pixels of the species concentration matrices
    for (((out_u, out_v), win_u), win_v) in (out_u.simd_center_mut().iter_mut())
        .zip(out_v.simd_center_mut().iter_mut())
        .zip(in_u.simd_stencil_windows())
        .zip(in_v.simd_stencil_windows())
    {
        // Access center value of u
        let u = win_u[stencil_offset];
        let v = win_v[stencil_offset];

        // Compute diffusion gradient
        let [full_u, full_v] = (win_u.iter())
            .zip(win_v.iter())
            .zip(params.weights.into_iter().flat_map(|row| row.into_iter()))
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
        let du = mul_add(
            diffusion_rate_u,
            full_u,
            mul_add(feed_rate, ones - u, -uv_square),
        );
        let dv = mul_add(
            diffusion_rate_v,
            full_v,
            mul_add(-(feed_rate + kill_rate), v, uv_square),
        );
        *out_u = mul_add(du, time_step, u);
        *out_v = mul_add(dv, time_step, v);
    }
}
