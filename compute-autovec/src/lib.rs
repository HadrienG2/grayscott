//! Auto-vectorized implementation of Gray-Scott simulation
//!
//! While compilers can automatically vectorize computations, said computations
//! must in all but simplest cases be shaped exactly like manually vectorized
//! code based on hardware intrinsics would be. This compute backend follows
//! this strategy, which should allow it to perform decently on hardware other
//! than the hardware it was written for (x86_64), with minimal porting effort
//! revolving around picking the right vector width.

use data::{
    concentration::simd::SIMDConcentration,
    parameters::{stencil_offset, Parameters},
    Precision,
};
use slipstream::{vector::align, Vector};

/// Chosen SIMD vector type
const PRECISION_SIZE: usize = std::mem::size_of::<Precision>();
cfg_if::cfg_if! {
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
                        acc_u + weight * (stencil_u - u),
                        acc_v + weight * (stencil_v - v),
                    ]
                },
            );

        // Deduce variation of U and V
        let uv_square = u * v * v;
        let du = diffusion_rate_u * full_u - uv_square + feed_rate * (ones - u);
        let dv = diffusion_rate_v * full_v + uv_square - (feed_rate + kill_rate) * v;
        *out_u = u + du * time_step;
        *out_v = v + dv * time_step;
    }
}
