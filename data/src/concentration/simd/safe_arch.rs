//! Implementation of SIMDXyz traits for safe_arch types
//!
//! In future Rust, I'll be able to drop this and use stdsimd instead.

use super::{SIMDIndices, SIMDMask, SIMDValues};

// Implementation for 128-bit SSE vectors
// Only consider SSE2, as SSE-only is extinct according to the Steam Survey...
#[cfg(target_feature = "sse2")]
mod sse2 {
    use super::*;
    use safe_arch::{m128, m128d, m128i};

    // ...but we need blend for efficient `fill_slice()`, and that's SSE 4.1.
    // While SSE2 without SSE4.1 is also close to extinct in the Steam survey,
    // it remains important because that's the configuration that x86_64
    // compilers build for by default, and we don't want to drop all
    // vectorization just because this smaller code path can't be vectorized.
    cfg_if::cfg_if! {
        // All this to say, if SSE4.1 blend is available, use it...
        if #[cfg(target_feature = "sse4.1")] {
            pub type Mask128 = m128;
            pub type Mask128d = m128d;
            //
            impl SIMDMask<4> for Mask128 {
                #[inline]
                fn splat(b: bool) -> Self {
                    Self::from_bits([b as u32 * u32::MAX; 4])
                }
            }
            //
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
        } else {
            // ...otherwise, emulate it using SSE2 and scalar code
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
                Mask(<[i32; 4]>::from(mask).map(|i| i != 0))
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
    }

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
            other.lt(self)
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

    // Then we can finally use m128 as f32x4...
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

    // ...and m128d as f64x2...
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

// Implementation for 256-bit AVX vectors
#[cfg(target_feature = "avx")]
mod avx {
    use super::*;
    use cfg_if::cfg_if;
    #[cfg(target_feature = "avx2")]
    use safe_arch::m256i;
    use safe_arch::{m256, m256d};
    use std::{
        arch::x86_64::{__m256, __m256i, _mm256_permutevar8x32_ps},
        array,
    };

    // Doubles are easy, they are implemented almost just like in
    // SSE 4.1, it's only comparisons and shuffles that differ a little...
    impl SIMDMask<4> for m256d {
        #[inline]
        fn splat(b: bool) -> Self {
            Self::from_bits([b as u64 * u64::MAX; 4])
        }
    }
    //
    impl SIMDIndices<4> for m256d {
        type Mask = m256d;

        #[inline]
        fn from_array(arr: [i32; 4]) -> Self {
            Self::from(arr.map(|i| i as f64))
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
    //
    impl SIMDValues<4, f64> for m256d {
        type Indices = m256d;

        type Mask = m256d;

        #[inline]
        fn splat(x: f64) -> Self {
            safe_arch::set_splat_m256d(x)
        }

        #[inline]
        fn blend(self, other: Self, mask: Self::Mask) -> Self {
            safe_arch::blend_varying_m256d(self, other, mask)
        }

        #[inline(always)]
        fn shift_left(self, offset: usize) -> Self {
            let zero = safe_arch::set_splat_m256d(0.0);
            match offset {
                0 => self,
                1 => {
                    if cfg!(target_feature = "avx2") {
                        let s4s1_s2s3 = safe_arch::shuffle_ai_f64_all_m256d::<0b10_01_00_11>(self);
                        safe_arch::blend_m256d::<0b1110>(zero, s4s1_s2s3)
                    } else {
                        let s2s1_s4s3 = safe_arch::permute_m256d::<0b0101>(self);
                        let zero_s2s1 =
                            safe_arch::permute2z_m256d::<0b0000_1000>(s2s1_s4s3, s2s1_s4s3);
                        safe_arch::blend_m256d::<0b1010>(zero_s2s1, s2s1_s4s3)
                    }
                }
                2 => safe_arch::permute2z_m256d::<0b0000_1000>(self, self),
                3 => {
                    if cfg!(target_feature = "avx2") {
                        let s2s3_s4s1 = safe_arch::shuffle_ai_f64_all_m256d::<0b00_11_10_01>(self);
                        safe_arch::blend_m256d::<0b1000>(zero, s2s3_s4s1)
                    } else {
                        let s1s2 = safe_arch::cast_to_m128d_from_m256d(self);
                        let z0s1 = s1s2.shift_left(1);
                        safe_arch::insert_m128d_to_m256d::<1>(zero, z0s1)
                    }
                }
                _ => zero,
            }
        }

        #[inline(always)]
        fn shift_right(self, offset: usize) -> Self {
            let zero = safe_arch::set_splat_m256d(0.0);
            match offset {
                0 => self,
                1 => {
                    if cfg!(target_feature = "avx2") {
                        let s2s3_s4s1 = safe_arch::shuffle_ai_f64_all_m256d::<0b00_11_10_01>(self);
                        safe_arch::blend_m256d::<0b0111>(zero, s2s3_s4s1)
                    } else {
                        let s2s1_s4s3 = safe_arch::permute_m256d::<0b0101>(self);
                        let s4s3_zero =
                            safe_arch::permute2z_m256d::<0b1000_0001>(s2s1_s4s3, s2s1_s4s3);
                        safe_arch::blend_m256d::<0b1010>(s2s1_s4s3, s4s3_zero)
                    }
                }
                2 => safe_arch::permute2z_m256d::<0b1000_0001>(self, self),
                3 => {
                    if cfg!(target_feature = "avx2") {
                        let full_rotate =
                            safe_arch::shuffle_ai_f64_all_m256d::<0b00_00_00_11>(self);
                        safe_arch::blend_m256d::<0b0001>(zero, full_rotate)
                    } else {
                        let s3s4 = safe_arch::extract_m128d_from_m256d::<1>(self);
                        let s4z0 = s3s4.shift_right(1);
                        safe_arch::zero_extend_m128d(s4z0)
                    }
                }
                _ => zero,
            }
        }

        #[inline]
        fn transpose([a, b, c, d]: [Self; 4]) -> [Self; 4] {
            let a1b1_a3b3 = safe_arch::unpack_lo_m256d(a, b);
            let a2b2_a4b4 = safe_arch::unpack_hi_m256d(a, b);
            let c1d1_c3d3 = safe_arch::unpack_lo_m256d(c, d);
            let c2d2_c4d4 = safe_arch::unpack_hi_m256d(c, d);
            [
                safe_arch::permute2z_m256d::<0b0010_0000>(a1b1_a3b3, c1d1_c3d3),
                safe_arch::permute2z_m256d::<0b0010_0000>(a2b2_a4b4, c2d2_c4d4),
                safe_arch::permute2z_m256d::<0b0011_0001>(a1b1_a3b3, c1d1_c3d3),
                safe_arch::permute2z_m256d::<0b0011_0001>(a2b2_a4b4, c2d2_c4d4),
            ]
        }

        #[inline(always)]
        fn store(self, target: &mut [f64]) {
            assert_eq!(target.len(), 4);
            let target = target as *mut [f64] as *mut [f64; 4];
            unsafe {
                safe_arch::store_unaligned_m256d(&mut *target, self);
            }
        }

        #[inline]
        fn into_array(self) -> [f64; 4] {
            self.into()
        }
    }

    // ...but when it comes to floats, we hit the fact that integer operations
    // only landed in AVX2. Before that, we can't use m256i as an i32x8.
    cfg_if! {
        // If we have AVX2, we handle things with m256i like in the SSE4.1 case
        if #[cfg(target_feature = "avx2")] {
            impl SIMDIndices<8> for m256i {
                type Mask = m256;

                #[inline]
                fn from_array(arr: [i32; 8]) -> Self {
                    Self::from(arr)
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
            //
            pub type Indices = m256i;
        } else {
            // If we don't have AVX2, we emulate m256i with [m256d; 2]
            impl SIMDIndices<8> for [m256d; 2] {
                type Mask = m256;

                #[inline]
                fn from_array(arr: [i32; 8]) -> Self {
                    [
                        m256d::from_array([arr[0], arr[1], arr[2], arr[3]]),
                        m256d::from_array([arr[4], arr[5], arr[6], arr[7]]),
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
            //
            pub type Indices = [m256d; 2];
        }
    }
    //
    impl SIMDMask<8> for m256 {
        #[inline]
        fn splat(b: bool) -> Self {
            Self::from_bits([b as u32 * u32::MAX; 8])
        }
    }
    //
    impl SIMDValues<8, f32> for m256 {
        type Indices = Indices;

        type Mask = m256;

        #[inline]
        fn splat(x: f32) -> Self {
            safe_arch::set_splat_m256(x)
        }

        #[inline]
        fn blend(self, other: Self, mask: Self::Mask) -> Self {
            safe_arch::blend_varying_m256(self, other, mask)
        }

        #[inline(always)]
        fn shift_left(self, offset: usize) -> Self {
            // For even offsets, we can reuse the work done for m256d, as far as
            // I know no x86 implementation has f32 -> f64 bypass delays
            if offset % 2 == 0 {
                return safe_arch::cast_to_m256_from_m256d(
                    safe_arch::cast_to_m256d_from_m256(self).shift_left(offset / 2),
                );
            }

            // For odd offsets, we need new code
            let zero = safe_arch::set_splat_m256(0.0);
            if cfg!(target_feature = "avx2") {
                let rotate_left_indices =
                    m256i::from(array::from_fn(|i| (i.wrapping_sub(offset) % 8) as i32));
                let full_rotate = shuffle_av_f32_all_m256d(self, rotate_left_indices);
                match offset {
                    0 => unreachable!(),
                    1 => safe_arch::blend_m256::<0b11111110>(zero, full_rotate),
                    2 => unreachable!(),
                    3 => safe_arch::blend_m256::<0b11111000>(zero, full_rotate),
                    4 => unreachable!(),
                    5 => safe_arch::blend_m256::<0b11100000>(zero, full_rotate),
                    6 => unreachable!(),
                    7 => safe_arch::blend_m256::<0b10000000>(zero, full_rotate),
                    _ => zero,
                }
            } else {
                let carry = |v| safe_arch::permute2z_m256::<0b0000_1000>(v, v);
                match offset {
                    0 => unreachable!(),
                    1 => {
                        let s4s1s2s3_s8s5s6s7 = safe_arch::permute_m256::<0b10_01_00_11>(self);
                        let zero_s4s1s2s3 = carry(s4s1s2s3_s8s5s6s7);
                        safe_arch::blend_m256::<0b11101110>(zero_s4s1s2s3, s4s1s2s3_s8s5s6s7)
                    }
                    2 => unreachable!(),
                    3 => {
                        let s2s3s4s1_s6s7s8s5 = safe_arch::permute_m256::<0b00_11_10_01>(self);
                        let zero_s2s3s4s1 = carry(s2s3s4s1_s6s7s8s5);
                        safe_arch::blend_m256::<0b10001000>(zero_s2s3s4s1, s2s3s4s1_s6s7s8s5)
                    }
                    4 => unreachable!(),
                    5 | 7 => {
                        let s1s2s3s4 = safe_arch::cast_to_m128_from_m256(self);
                        let shifted = s1s2s3s4.shift_left(offset - 4);
                        safe_arch::insert_m128_to_m256::<1>(zero, shifted)
                    }
                    6 => unreachable!(),
                    _ => zero,
                }
            }
        }

        #[inline(always)]
        fn shift_right(self, offset: usize) -> Self {
            // For even offsets, we can reuse the work done for m256d, as far as
            // I know no x86 implementation has f32 -> f64 bypass delays
            if offset % 2 == 0 {
                return safe_arch::cast_to_m256_from_m256d(
                    safe_arch::cast_to_m256d_from_m256(self).shift_right(offset / 2),
                );
            }

            // For odd offsets, we need new code
            let zero = safe_arch::set_splat_m256(0.0);
            if cfg!(target_feature = "avx2") {
                let rotate_right_indices =
                    m256i::from(array::from_fn(|i| ((i + offset) % 8) as i32));
                let full_rotate = shuffle_av_f32_all_m256d(self, rotate_right_indices);
                match offset {
                    0 => unreachable!(),
                    1 => safe_arch::blend_m256::<0b01111111>(zero, full_rotate),
                    2 => unreachable!(),
                    3 => safe_arch::blend_m256::<0b00011111>(zero, full_rotate),
                    4 => unreachable!(),
                    5 => safe_arch::blend_m256::<0b00000111>(zero, full_rotate),
                    6 => unreachable!(),
                    7 => safe_arch::blend_m256::<0b00000001>(zero, full_rotate),
                    _ => zero,
                }
            } else {
                let carry = |v| safe_arch::permute2z_m256::<0b1000_0001>(v, v);
                match offset {
                    0 => unreachable!(),
                    1 => {
                        let s2s3s4s1_s6s7s8s5 = safe_arch::permute_m256::<0b00_11_10_01>(self);
                        let s6s7s8s5_zero = carry(s2s3s4s1_s6s7s8s5);
                        safe_arch::blend_m256::<0b01110111>(s6s7s8s5_zero, s2s3s4s1_s6s7s8s5)
                    }
                    2 => unreachable!(),
                    3 => {
                        let s4s1s2s3_s8s5s6s7 = safe_arch::permute_m256::<0b10_01_00_11>(self);
                        let s8s5s6s7_zero = carry(s4s1s2s3_s8s5s6s7);
                        safe_arch::blend_m256::<0b00010001>(s8s5s6s7_zero, s4s1s2s3_s8s5s6s7)
                    }
                    4 => unreachable!(),
                    5 | 7 => {
                        let s5s6s7s8 = safe_arch::extract_m128_from_m256::<1>(self);
                        let shifted = s5s6s7s8.shift_right(offset - 4);
                        safe_arch::zero_extend_m128(shifted)
                    }
                    6 => unreachable!(),
                    _ => zero,
                }
            }
        }

        #[inline]
        fn transpose([a, b, c, d, e, f, g, h]: [Self; 8]) -> [Self; 8] {
            use safe_arch::{permute2z_m256, shuffle_m256, unpack_hi_m256, unpack_lo_m256};

            // Fill 128-bit lanes with 1-2, 3-4, 5-6 and 7-8 pairs
            let a1b1a2b2_a5b5a6b6 = unpack_lo_m256(a, b);
            let a3b3a4b4_a7b7a8b8 = unpack_hi_m256(a, b);
            //
            let c1d1c2d2_c5d5c6d6 = unpack_lo_m256(c, d);
            let c3d3c4d4_c7d7c8d8 = unpack_hi_m256(c, d);
            //
            let e1f1e2f2_e5f5e6f6 = unpack_lo_m256(e, f);
            let e3f3e4f4_e7f7e8f8 = unpack_hi_m256(e, f);
            //
            let g1h1g2h2_g5h5g6h6 = unpack_lo_m256(g, h);
            let g3h3g4h4_g7h7g8h8 = unpack_hi_m256(g, h);

            // Bring the 1-2, 3-4, 5-6 and 7-8 pairs together in a SHUF-friendly layout
            let a1b1a2b2_e1f1e2f2 =
                permute2z_m256::<0b0010_0000>(a1b1a2b2_a5b5a6b6, e1f1e2f2_e5f5e6f6);
            let c1d1c2d2_g1h1g2h2 =
                permute2z_m256::<0b0010_0000>(c1d1c2d2_c5d5c6d6, g1h1g2h2_g5h5g6h6);
            //
            let a3b3a4b4_e3f3e4f4 =
                permute2z_m256::<0b0010_0000>(a3b3a4b4_a7b7a8b8, e3f3e4f4_e7f7e8f8);
            let c3d3c4d4_g3h3g4h4 =
                permute2z_m256::<0b0010_0000>(c3d3c4d4_c7d7c8d8, g3h3g4h4_g7h7g8h8);
            //
            let a5b5a6b6_e5f5e6f6 =
                permute2z_m256::<0b0011_0001>(a1b1a2b2_a5b5a6b6, e1f1e2f2_e5f5e6f6);
            let c5d5c6d6_g5h5g6h6 =
                permute2z_m256::<0b0011_0001>(c1d1c2d2_c5d5c6d6, g1h1g2h2_g5h5g6h6);
            //
            let a7b7a8b8_e7f7e8f8 =
                permute2z_m256::<0b0011_0001>(a3b3a4b4_a7b7a8b8, e3f3e4f4_e7f7e8f8);
            let c7d7c8d8_g7h7g8h8 =
                permute2z_m256::<0b0011_0001>(c3d3c4d4_c7d7c8d8, g3h3g4h4_g7h7g8h8);

            // Now we can bring the 1s, 2s, 3s, 4s, 5s, 6s, 7s and 8s together
            [
                shuffle_m256::<0b01_00_01_00>(a1b1a2b2_e1f1e2f2, c1d1c2d2_g1h1g2h2),
                shuffle_m256::<0b11_10_11_10>(a1b1a2b2_e1f1e2f2, c1d1c2d2_g1h1g2h2),
                shuffle_m256::<0b01_00_01_00>(a3b3a4b4_e3f3e4f4, c3d3c4d4_g3h3g4h4),
                shuffle_m256::<0b11_10_11_10>(a3b3a4b4_e3f3e4f4, c3d3c4d4_g3h3g4h4),
                shuffle_m256::<0b01_00_01_00>(a5b5a6b6_e5f5e6f6, c5d5c6d6_g5h5g6h6),
                shuffle_m256::<0b11_10_11_10>(a5b5a6b6_e5f5e6f6, c5d5c6d6_g5h5g6h6),
                shuffle_m256::<0b01_00_01_00>(a7b7a8b8_e7f7e8f8, c7d7c8d8_g7h7g8h8),
                shuffle_m256::<0b11_10_11_10>(a7b7a8b8_e7f7e8f8, c7d7c8d8_g7h7g8h8),
            ]
        }

        #[inline(always)]
        fn store(self, target: &mut [f32]) {
            assert_eq!(target.len(), 8);
            let target = target as *mut [f32] as *mut [f32; 8];
            unsafe {
                safe_arch::store_unaligned_m256(&mut *target, self);
            }
        }

        #[inline]
        fn into_array(self) -> [f32; 8] {
            self.into()
        }
    }

    // Intrinsic interface to vpermps that was forgotten by safe_arch
    #[inline(always)]
    #[cfg(target_feature = "avx2")]
    fn shuffle_av_f32_all_m256d(a: m256, idx: m256i) -> m256 {
        unsafe {
            let a = std::mem::transmute::<m256, __m256>(a);
            let idx = std::mem::transmute::<m256i, __m256i>(idx);
            let result = _mm256_permutevar8x32_ps(a, idx);
            std::mem::transmute::<__m256, m256>(result)
        }
    }
}