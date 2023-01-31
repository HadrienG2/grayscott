//! Implementation of SIMDXyz traits for safe_arch types
//!
//! In future Rust, I'll be able to drop this and use stdsimd instead.

use super::{SIMDIndices, SIMDMask, SIMDValues};

/// We ignore SSE before SSE2 since those CPUs are near extinct today...
#[cfg(target_feature = "sse2")]
mod sse2 {
    use super::*;
    use safe_arch::{m128, m128d, m128i};

    /// ...but SSE2 without SSE 4.1 is an important configuration since that's
    /// the default compiler setting on 64-bit CPUs. And SSE2 did not yet have
    /// blends yet, so we must emulate them in that configuration...
    #[cfg(not(target_feature = "sse4.1"))]
    mod pre_sse41 {
        use super::*;
        use std::{array, ops::BitAnd};

        #[derive(Copy, Clone, Debug, Eq, PartialEq)]
        pub struct Mask<const WIDTH: usize>([bool; WIDTH]);
        //
        impl<const WIDTH: usize> SIMDMask<WIDTH> for Mask<WIDTH> {
            #[inline]
            fn splat(b: bool) -> Self {
                Self([b; WIDTH])
            }
        }
        //
        impl<const WIDTH: usize> BitAnd for Mask<WIDTH> {
            type Output = Self;

            #[inline]
            fn bitand(self, other: Self) -> Self {
                Self(array::from_fn(|i| self.0[i] & other.0[i]))
            }
        }
        //
        pub type Mask128 = Mask<4>;
        pub type Mask128d = Mask<2>;

        #[inline]
        pub fn mask128_from_i32_m128i(mask: m128i) -> Mask128 {
            Mask(<[i32; 4]>::from(mask).map(|i| i < 0))
        }

        #[inline]
        pub fn mask128d_from_m128d(mask: m128d) -> Mask128d {
            Mask(mask.to_bits().map(|i| i != 0))
        }

        #[inline]
        pub fn blend_varying_m128(a: m128, b: m128, mask: Mask128) -> m128 {
            let a = <[f32; 4]>::from(a);
            let b = <[f32; 4]>::from(b);
            vector_from_fn(|i| if mask.0[i] { b[i] } else { a[i] })
        }

        #[inline]
        pub fn blend_varying_m128d(a: m128d, b: m128d, mask: Mask128d) -> m128d {
            let a = <[f64; 2]>::from(a);
            let b = <[f64; 2]>::from(b);
            vector_from_fn(|i| if mask.0[i] { b[i] } else { a[i] })
        }

        #[inline]
        fn vector_from_fn<Vector, T, const N: usize>(f: impl FnMut(usize) -> T) -> Vector
        where
            Vector: From<[T; N]>,
        {
            array::from_fn(f).into()
        }
    }
    #[cfg(not(target_feature = "sse4.1"))]
    use pre_sse41::*;

    /// Of course, if we do have SSE4.1 and its varying blends, we'll use them
    #[cfg(target_feature = "sse4.1")]
    mod sse41 {
        use super::*;

        pub type Mask128 = m128;
        pub type Mask128d = m128d;

        impl SIMDMask<4> for Mask128 {
            #[inline]
            fn splat(b: bool) -> Self {
                Self::from_bits([b as u32 * u32::MAX; 4])
            }
        }

        impl SIMDMask<2> for Mask128d {
            #[inline]
            fn splat(b: bool) -> Self {
                Self::from_bits([b as u64 * u64::MAX; 2])
            }
        }

        #[inline]
        pub fn mask128_from_i32_m128i(mask: m128i) -> Mask128 {
            safe_arch::cast_to_m128_from_m128i(mask)
        }

        #[inline]
        pub fn mask128d_from_m128d(mask: m128d) -> Mask128d {
            mask
        }

        pub use safe_arch::blend_varying_m128;
        pub use safe_arch::blend_varying_m128d;
    }
    #[cfg(target_feature = "sse4.1")]
    use sse41::*;

    // An m128i naturally acts as [i32; 4] in SSE2...
    impl SIMDIndices<4> for m128i {
        type Mask = Mask128;

        #[inline]
        fn from_array(arr: [i32; 4]) -> Self {
            Self::from(arr)
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
            mask128_from_i32_m128i(
                safe_arch::cmp_gt_mask_i32_m128i(self, other)
                    | safe_arch::cmp_eq_mask_i32_m128i(self, other),
            )
        }

        #[inline]
        fn lt(self, other: Self) -> Self::Mask {
            mask128_from_i32_m128i(safe_arch::cmp_lt_mask_i32_m128i(self, other))
        }
    }

    // ...however, we can't use an m128i as an [i64; 2] because some comparison
    // operators we need do not exist. We hack around that by recalling that f64
    // has more than enough mantissa digits to exactly hold an i32 value, which
    // means an [f64; 2] can serve as an [i32; 2] without loss of precision.
    impl SIMDIndices<2> for m128d {
        type Mask = Mask128d;

        #[inline]
        fn from_array(arr: [i32; 2]) -> Self {
            Self::from(arr.map(|i| i as f64))
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
            mask128d_from_m128d(safe_arch::cmp_ge_mask_m128d(self, other))
        }

        #[inline]
        fn lt(self, other: Self) -> Self::Mask {
            mask128d_from_m128d(safe_arch::cmp_lt_mask_m128d(self, other))
        }
    }

    impl SIMDValues<4, f32> for m128 {
        type Indices = m128i;

        type Mask = Mask128;

        #[inline]
        fn splat(x: f32) -> Self {
            safe_arch::set_splat_m128(x)
        }

        #[inline]
        fn blend(self, other: Self, mask: Self::Mask) -> Self {
            blend_varying_m128(self, other, mask)
        }

        #[inline(always)]
        fn shift_left(self, offset: usize) -> Self {
            safe_arch::cast_to_m128_from_m128i(lane_shl_i32_m128i(
                safe_arch::cast_to_m128i_from_m128(self),
                offset,
            ))
        }

        #[inline(always)]
        fn shift_right(self, offset: usize) -> Self {
            safe_arch::cast_to_m128_from_m128i(lane_shr_i32_m128i(
                safe_arch::cast_to_m128i_from_m128(self),
                offset,
            ))
        }

        #[inline]
        fn transpose(mut matrix: [Self; 4]) -> [Self; 4] {
            let [a, b, c, d] = &mut matrix;
            safe_arch::transpose_four_m128(a, b, c, d);
            matrix
        }

        #[inline(always)]
        fn store(self, target: &mut [f32]) {
            assert_eq!(target.len(), 4);
            let target = target as *mut [f32] as *mut [f32; 4];
            unsafe {
                safe_arch::store_unaligned_m128(&mut *target, self);
            }
        }

        #[inline]
        fn into_array(self) -> [f32; 4] {
            self.into()
        }
    }

    impl SIMDValues<2, f64> for m128d {
        type Indices = m128d;

        type Mask = Mask128d;

        #[inline]
        fn splat(x: f64) -> Self {
            safe_arch::set_splat_m128d(x)
        }

        #[inline]
        fn blend(self, other: Self, mask: Self::Mask) -> Self {
            blend_varying_m128d(self, other, mask)
        }

        #[inline(always)]
        fn shift_left(self, offset: usize) -> Self {
            safe_arch::cast_to_m128d_from_m128i(lane_shl_i32_m128i(
                safe_arch::cast_to_m128i_from_m128d(self),
                2 * offset,
            ))
        }

        #[inline(always)]
        fn shift_right(self, offset: usize) -> Self {
            safe_arch::cast_to_m128d_from_m128i(lane_shr_i32_m128i(
                safe_arch::cast_to_m128i_from_m128d(self),
                2 * offset,
            ))
        }

        #[inline]
        fn transpose([a, b]: [Self; 2]) -> [Self; 2] {
            [
                safe_arch::shuffle_abi_f64_all_m128d::<0b00>(a, b),
                safe_arch::shuffle_abi_f64_all_m128d::<0b11>(a, b),
            ]
        }

        #[inline(always)]
        fn store(self, target: &mut [f64]) {
            assert_eq!(target.len(), 2);
            let target = target as *mut [f64] as *mut [f64; 2];
            unsafe {
                safe_arch::store_unaligned_m128d(&mut *target, self);
            }
        }

        #[inline]
        fn into_array(self) -> [f64; 2] {
            self.into()
        }
    }

    #[inline(always)]
    fn lane_shl_i32_m128i(v: m128i, offset: usize) -> m128i {
        match offset {
            0 => v,
            1 => safe_arch::byte_shl_imm_u128_m128i::<4>(v),
            2 => safe_arch::byte_shl_imm_u128_m128i::<8>(v),
            3 => safe_arch::byte_shl_imm_u128_m128i::<12>(v),
            _ => safe_arch::set_splat_i32_m128i(0),
        }
    }

    #[inline(always)]
    fn lane_shr_i32_m128i(v: m128i, offset: usize) -> m128i {
        match offset {
            0 => v,
            1 => safe_arch::byte_shr_imm_u128_m128i::<4>(v),
            2 => safe_arch::byte_shr_imm_u128_m128i::<8>(v),
            3 => safe_arch::byte_shr_imm_u128_m128i::<12>(v),
            _ => safe_arch::set_splat_i32_m128i(0),
        }
    }
}

// FIXME: ...then do AVX2
