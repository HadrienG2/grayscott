//! SIMD abstraction layer

use cfg_if::cfg_if;

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
    #[inline]
    fn mul_add(self, mul: Self, add: Self) -> Self {
        self.mul(mul).add(add)
    }

    /// Multiply-subtract
    #[inline]
    fn mul_sub(self, mul: Self, sub: Self) -> Self {
        self.mul(mul).sub(sub)
    }

    /// Negated multiply-add
    #[inline]
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
// which is the support code used by Species. Though that comparison is a bit on
// the unfair side as I wrote that code a little more general than it needs to
// be, so that I can reuse it in other contexts later on.
//
cfg_if! {
    if #[cfg(target_feature = "avx")] {
        // Use AVX if available
        use safe_arch::{m256, m256d};

        impl Vector for m256 {
            const WIDTH: usize = 8;

            type Element = f32;

            #[inline]
            fn splat(x: Self::Element) -> Self {
                safe_arch::set_splat_m256(x)
            }

            #[inline]
            fn add(self, other: Self) -> Self {
                safe_arch::add_m256(self, other)
            }

            #[inline]
            fn sub(self, other: Self) -> Self {
                safe_arch::sub_m256(self, other)
            }

            #[inline]
            fn mul(self, other: Self) -> Self {
                safe_arch::mul_m256(self, other)
            }

            #[cfg(target_feature = "fma")]
            #[inline]
            fn mul_add(self, mul: Self, add: Self) -> Self {
                safe_arch::fused_mul_add_m256(self, mul, add)
            }

            #[cfg(target_feature = "fma")]
            #[inline]
            fn mul_sub(self, mul: Self, sub: Self) -> Self {
                safe_arch::fused_mul_sub_m256(self, mul, sub)
            }

            #[cfg(target_feature = "fma")]
            #[inline]
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

            #[inline]
            fn splat(x: Self::Element) -> Self {
                safe_arch::set_splat_m256d(x)
            }

            #[inline]
            fn add(self, other: Self) -> Self {
                safe_arch::add_m256d(self, other)
            }

            #[inline]
            fn sub(self, other: Self) -> Self {
                safe_arch::sub_m256d(self, other)
            }

            #[inline]
            fn mul(self, other: Self) -> Self {
                safe_arch::mul_m256d(self, other)
            }

            #[cfg(target_feature = "fma")]
            #[inline]
            fn mul_add(self, mul: Self, add: Self) -> Self {
                safe_arch::fused_mul_add_m256d(self, mul, add)
            }

            #[cfg(target_feature = "fma")]
            #[inline]
            fn mul_sub(self, mul: Self, sub: Self) -> Self {
                safe_arch::fused_mul_sub_m256d(self, mul, sub)
            }

            #[cfg(target_feature = "fma")]
            #[inline]
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

            #[inline]
            fn splat(x: Self::Element) -> Self {
                safe_arch::set_splat_m128(x)
            }

            #[inline]
            fn add(self, other: Self) -> Self {
                safe_arch::add_m128(self, other)
            }

            #[inline]
            fn sub(self, other: Self) -> Self {
                safe_arch::sub_m128(self, other)
            }

            #[inline]
            fn mul(self, other: Self) -> Self {
                safe_arch::mul_m128(self, other)
            }

            #[cfg(target_feature = "fma")]
            #[inline]
            fn mul_add(self, mul: Self, add: Self) -> Self {
                safe_arch::fused_mul_add_m128(self, mul, add)
            }

            #[cfg(target_feature = "fma")]
            #[inline]
            fn mul_sub(self, mul: Self, sub: Self) -> Self {
                safe_arch::fused_mul_sub_m128(self, mul, sub)
            }

            #[cfg(target_feature = "fma")]
            #[inline]
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

            #[inline]
            fn splat(x: Self::Element) -> Self {
                safe_arch::set_splat_m128d(x)
            }

            #[inline]
            fn add(self, other: Self) -> Self {
                safe_arch::add_m128d(self, other)
            }

            #[inline]
            fn sub(self, other: Self) -> Self {
                safe_arch::sub_m128d(self, other)
            }

            #[inline]
            fn mul(self, other: Self) -> Self {
                safe_arch::mul_m128d(self, other)
            }

            #[cfg(target_feature = "fma")]
            #[inline]
            fn mul_add(self, mul: Self, add: Self) -> Self {
                safe_arch::fused_mul_add_m128d(self, mul, add)
            }

            #[cfg(target_feature = "fma")]
            #[inline]
            fn mul_sub(self, mul: Self, sub: Self) -> Self {
                safe_arch::fused_mul_sub_m128d(self, mul, sub)
            }

            #[cfg(target_feature = "fma")]
            #[inline]
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

            #[inline]
            fn splat(x: Self::Element) -> Self {
                x
            }

            #[inline]
            fn add(self, other: Self) -> Self {
                self + other
            }

            #[inline]
            fn sub(self, other: Self) -> Self {
                self - other
            }

            #[inline]
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
