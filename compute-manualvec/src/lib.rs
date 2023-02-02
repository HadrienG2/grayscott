//! Manually vectorized implementation of Gray-Scott simulation
//!
//! This implementation is vectorized through direct use of hardware intrinsics.
//! Overall, it does not compare favorably with the autovectorized version from
//! the perspective of code complexity, portability and runtime performance,
//! which shows that compilers have good enough at autovectorizing stencil
//! computations and one can't really justify manually vectorizing those anymore.

use cfg_if::cfg_if;
use data::{
    concentration::simd::SIMDConcentration,
    parameters::{stencil_offset, Parameters},
    Precision,
};

/// Chosen concentration type
pub type Values = <Precision as Scalar>::Vectorized;
pub type Species = data::concentration::Species<SIMDConcentration<{ Values::WIDTH }, Values>>;

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
                        weight.mul_add(stencil_u.sub(u), acc_u),
                        weight.mul_add(stencil_v.sub(v), acc_v),
                    ]
                },
            );

        // Deduce variation of U and V
        let uv_square = u.mul(v).mul(v);
        let du = diffusion_rate_u.mul_add(full_u, feed_rate.mul_sub(ones.sub(u), uv_square));
        let dv =
            diffusion_rate_v.mul_add(full_v, (feed_rate.add(kill_rate)).mul_neg_add(v, uv_square));
        *out_u = du.mul_add(time_step, u);
        *out_v = dv.mul_add(time_step, v);
    }
}

/// SIMD abstraction layer
pub trait Vector: Copy + Sized {
    /// Vector width
    const WIDTH: usize;

    /// Element type
    type Element;

    /// Broadcasting constructor
    fn splat(x: Self::Element) -> Self;

    /// Addition
    fn add(self, other: Self) -> Self;

    /// Subtraction
    fn sub(self, other: Self) -> Self;

    /// Multiplication
    fn mul(self, other: Self) -> Self;

    /// Multiply-add
    fn mul_add(self, mul: Self, add: Self) -> Self {
        self.mul(mul).add(add)
    }

    /// Multiply-subtract
    fn mul_sub(self, mul: Self, sub: Self) -> Self {
        self.mul(mul).sub(sub)
    }

    /// Negated multiply-add
    fn mul_neg_add(self, min_mul: Self, add: Self) -> Self {
        add.sub(self.mul(min_mul))
    }
}

/// Mapping from scalar types to SIMD types
pub trait Scalar {
    /// SIMD vector type to use for this data type
    type Vectorized: Vector<Element = Self>;
}

// Pick vector size based on hardware support for vectorization of
// floating-point operations (which are the bulk of our SIMD workload).
//
// Notice that the code will not vectorize at all on non-x86 hardware.
//
// Also, if you think this is long, do check out data::concentration::safe_arch,
// which is the support code used by Species, though that comparison is a bit on
// the unfair side as I wrote that code a little more general than it needs to
// be so I can reuse it in other contexts later on.
//
cfg_if! {
    if #[cfg(target_feature = "avx")] {
        // Use AVX if available
        use safe_arch::{m256, m256d};

        impl Vector for m256 {
            const WIDTH: usize = 8;

            type Element = f32;

            fn splat(x: Self::Element) -> Self {
                safe_arch::set_splat_m256(x)
            }

            fn add(self, other: Self) -> Self {
                safe_arch::add_m256(self, other)
            }

            fn sub(self, other: Self) -> Self {
                safe_arch::sub_m256(self, other)
            }

            fn mul(self, other: Self) -> Self {
                safe_arch::mul_m256(self, other)
            }

            #[cfg(target_feature = "fma")]
            fn mul_add(self, mul: Self, add: Self) -> Self {
                safe_arch::fused_mul_add_m256(self, mul, add)
            }

            #[cfg(target_feature = "fma")]
            fn mul_sub(self, mul: Self, sub: Self) -> Self {
                safe_arch::fused_mul_sub_m256(self, mul, sub)
            }

            #[cfg(target_feature = "fma")]
            fn mul_neg_add(self, min_mul: Self, add: Self) -> Self {
                safe_arch::fused_mul_neg_add_m256(self, min_mul, add)
            }
        }

        impl Scalar for f32 {
            type Vectorized = m256;
        }

        impl Vector for m256d {
            const WIDTH: usize = 4;

            type Element = f64;

            fn splat(x: Self::Element) -> Self {
                safe_arch::set_splat_m256d(x)
            }

            fn add(self, other: Self) -> Self {
                safe_arch::add_m256d(self, other)
            }

            fn sub(self, other: Self) -> Self {
                safe_arch::sub_m256d(self, other)
            }

            fn mul(self, other: Self) -> Self {
                safe_arch::mul_m256d(self, other)
            }

            #[cfg(target_feature = "fma")]
            fn mul_add(self, mul: Self, add: Self) -> Self {
                safe_arch::fused_mul_add_m256d(self, mul, add)
            }

            #[cfg(target_feature = "fma")]
            fn mul_sub(self, mul: Self, sub: Self) -> Self {
                safe_arch::fused_mul_sub_m256d(self, mul, sub)
            }

            #[cfg(target_feature = "fma")]
            fn mul_neg_add(self, min_mul: Self, add: Self) -> Self {
                safe_arch::fused_mul_neg_add_m256d(self, min_mul, add)
            }
        }

        impl Scalar for f64 {
            type Vectorized = m256d;
        }
    } else if #[cfg(target_feature = "sse2")] {
        // If there is no AVX, use SSE if available
        use safe_arch::{m128, m128d};

        impl Vector for m128 {
            const WIDTH: usize = 4;

            type Element = f32;

            fn splat(x: Self::Element) -> Self {
                safe_arch::set_splat_m128(x)
            }

            fn add(self, other: Self) -> Self {
                safe_arch::add_m128(self, other)
            }

            fn sub(self, other: Self) -> Self {
                safe_arch::sub_m128(self, other)
            }

            fn mul(self, other: Self) -> Self {
                safe_arch::mul_m128(self, other)
            }

            #[cfg(target_feature = "fma")]
            fn mul_add(self, mul: Self, add: Self) -> Self {
                safe_arch::fused_mul_add_m128(self, mul, add)
            }

            #[cfg(target_feature = "fma")]
            fn mul_sub(self, mul: Self, sub: Self) -> Self {
                safe_arch::fused_mul_sub_m128(self, mul, sub)
            }

            #[cfg(target_feature = "fma")]
            fn mul_neg_add(self, min_mul: Self, add: Self) -> Self {
                safe_arch::fused_mul_neg_add_m128(self, min_mul, add)
            }
        }

        impl Scalar for f32 {
            type Vectorized = m128;
        }

        impl Vector for m128d {
            const WIDTH: usize = 2;

            type Element = f64;

            fn splat(x: Self::Element) -> Self {
                safe_arch::set_splat_m128d(x)
            }

            fn add(self, other: Self) -> Self {
                safe_arch::add_m128d(self, other)
            }

            fn sub(self, other: Self) -> Self {
                safe_arch::sub_m128d(self, other)
            }

            fn mul(self, other: Self) -> Self {
                safe_arch::mul_m128d(self, other)
            }

            #[cfg(target_feature = "fma")]
            fn mul_add(self, mul: Self, add: Self) -> Self {
                safe_arch::fused_mul_add_m128d(self, mul, add)
            }

            #[cfg(target_feature = "fma")]
            fn mul_sub(self, mul: Self, sub: Self) -> Self {
                safe_arch::fused_mul_sub_m128d(self, mul, sub)
            }

            #[cfg(target_feature = "fma")]
            fn mul_neg_add(self, min_mul: Self, add: Self) -> Self {
                safe_arch::fused_mul_neg_add_m128d(self, min_mul, add)
            }
        }

        impl Scalar for f64 {
            type Vectorized = m128d;
        }
    } else {
        // If all else fails, go for a scalar fallback
        impl<T: Add + Copy + Mul + Sized + Sub> Vector for T {
            const WIDTH: usize = 1;

            type Element = T;

            fn splat(x: Self::Element) -> Self {
                x
            }

            fn add(self, other: Self) -> Self {
                self + other
            }

            fn sub(self, other: Self) -> Self {
                self - other
            }

            fn mul(self, other: Self) -> Self {
                self * other
            }
        }

        impl Scalar for f32 {
            type Vectorized = f32;
        }

        impl Scalar for f64 {
            type Vectorized = f64;
        }
    }
}
