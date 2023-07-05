//! SIMD indexing for SSE

use super::masks::{self, Mask128, Mask128d};
use crate::concentration::simd::SIMDIndices;
use safe_arch::{m128d, m128i};

// An m128i naturally acts as [i32; 4] in SSE2...
impl SIMDIndices<4> for m128i {
    type Mask = Mask128;

    #[inline]
    fn from_idx_array(arr: [i32; 4]) -> Self {
        arr.into()
    }

    #[inline]
    fn splat(x: i32) -> Self {
        safe_arch::set_splat_i32_m128i(x)
    }

    #[inline]
    fn increment(&mut self) {
        let ones = safe_arch::set_splat_i32_m128i(1);
        *self = safe_arch::add_i32_m128i(*self, ones);
    }

    #[inline]
    fn ge(self, other: Self) -> Self::Mask {
        other.lt(self)
    }

    #[inline]
    fn lt(self, other: Self) -> Self::Mask {
        masks::mask128_from_i32_m128i(safe_arch::cmp_lt_mask_i32_m128i(self, other))
    }
}

// ...however, we can't use an m128i as an [i64; 2] because some comparison
// operators we need do not exist. We hack around that by recalling that f64
// has more than enough mantissa digits to exactly hold an i32 value, which
// means an [f64; 2] can serve as an [i32; 2] without loss of precision.
impl SIMDIndices<2> for m128d {
    type Mask = Mask128d;

    #[inline]
    fn from_idx_array(arr: [i32; 2]) -> Self {
        arr.map(f64::from).into()
    }

    #[inline]
    fn splat(x: i32) -> Self {
        safe_arch::set_splat_m128d(x as f64)
    }

    #[inline]
    fn increment(&mut self) {
        let ones = safe_arch::set_splat_m128d(1.0);
        *self = safe_arch::add_m128d(*self, ones);
    }

    #[inline]
    fn ge(self, other: Self) -> Self::Mask {
        masks::mask128d_from_m128d(safe_arch::cmp_ge_mask_m128d(self, other))
    }

    #[inline]
    fn lt(self, other: Self) -> Self::Mask {
        masks::mask128d_from_m128d(safe_arch::cmp_lt_mask_m128d(self, other))
    }
}
