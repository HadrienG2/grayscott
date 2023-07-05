//! SIMD indexing for AVX

use crate::concentration::simd::SIMDIndices;
use cfg_if::cfg_if;
#[cfg(target_feature = "avx2")]
use safe_arch::m256i;
use safe_arch::{m256, m256d};

// Doubles are easy, they are implemented almost just like in
// SSE 4.1, it's only comparisons and shuffles that differ a little...
impl SIMDIndices<4> for m256d {
    type Mask = m256d;

    #[inline]
    fn from_idx_array(arr: [i32; 4]) -> Self {
        arr.map(f64::from).into()
    }

    #[inline]
    fn splat(x: i32) -> Self {
        safe_arch::set_splat_m256d(x as f64)
    }

    #[inline]
    fn increment(&mut self) {
        let ones = safe_arch::set_splat_m256d(1.0);
        *self = safe_arch::add_m256d(*self, ones);
    }

    #[inline]
    fn ge(self, other: Self) -> Self::Mask {
        safe_arch::cmp_op_mask_m256d::<{ safe_arch::cmp_op!(GreaterEqualOrdered) }>(self, other)
    }

    #[inline]
    fn lt(self, other: Self) -> Self::Mask {
        safe_arch::cmp_op_mask_m256d::<{ safe_arch::cmp_op!(LessThanOrdered) }>(self, other)
    }
}

// ...but when it comes to floats, we hit the fact that integer operations
// only landed in AVX2. Before that, we can't use m256i as an i32x8.
cfg_if! {
    // If we have AVX2, we handle things with m256i like in the SSE4.1 case
    if #[cfg(target_feature = "avx2")] {
        pub type Indices256 = m256i;
        //
        impl SIMDIndices<8> for Indices256 {
            type Mask = m256;

            #[inline]
            fn from_idx_array(arr: [i32; 8]) -> Self {
                arr.into()
            }

            #[inline]
            fn splat(x: i32) -> Self {
                safe_arch::set_splat_i32_m256i(x)
            }

            #[inline]
            fn increment(&mut self) {
                let ones = safe_arch::set_splat_i32_m256i(1);
                *self = safe_arch::add_i32_m256i(*self, ones);
            }

            #[inline]
            fn ge(self, other: Self) -> Self::Mask {
                safe_arch::cast_to_m256_from_m256i(
                    safe_arch::cmp_gt_mask_i32_m256i(self, other)
                        | safe_arch::cmp_eq_mask_i32_m256i(self, other),
                )
            }

            #[inline]
            fn lt(self, other: Self) -> Self::Mask {
                other.ge(self)
            }
        }
    } else {
        // If we don't have AVX2, we emulate m256i with [m256d; 2]
        pub type Indices256 = [m256d; 2];
        //
        impl SIMDIndices<8> for Indices256 {
            type Mask = m256;

            #[inline]
            fn from_idx_array(arr: [i32; 8]) -> Self {
                [
                    m256d::from_idx_array([arr[0], arr[1], arr[2], arr[3]]),
                    m256d::from_idx_array([arr[4], arr[5], arr[6], arr[7]]),
                ]
            }

            #[inline]
            fn splat(x: i32) -> Self {
                [
                    m256d::splat(x),
                    m256d::splat(x),
                ]
            }

            #[inline]
            fn increment(&mut self) {
                for inner in self {
                    inner.increment();
                }
            }

            #[inline]
            fn ge(self, other: Self) -> Self::Mask {
                let masks256 = array2(|i| self[i].ge(other[i]));
                let [lo128; hi128] = masks256.map(safe_arch::convert_to_m128_from_m256d);
                safe_arch::set_m128_m256(hi128, lo128)
            }

            #[inline]
            fn lt(self, other: Self) -> Self::Mask {
                let masks256 = array2(|i| self[i].lt(other[i]));
                let [lo128; hi128] = masks256.map(safe_arch::convert_to_m128_from_m256d);
                safe_arch::set_m128_m256(hi128, lo128)
            }
        }
    }
}
