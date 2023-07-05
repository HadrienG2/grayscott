//! SIMD values for SSE

use super::masks::{self, Mask128, Mask128d};
use crate::concentration::simd::SIMDValues;
use safe_arch::{m128, m128d, m128i};

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
        masks::blend_varying_m128(self, other, mask)
    }

    #[inline]
    fn shift_left(self, offset: usize) -> Self {
        safe_arch::cast_to_m128_from_m128i(lane_shl_i32_m128i(
            safe_arch::cast_to_m128i_from_m128(self),
            offset,
        ))
    }

    #[inline]
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

    #[inline]
    fn store(self, target: &mut [f32]) {
        assert_eq!(target.len(), 4);
        // Safe due to assertion above
        unsafe {
            let target = target as *mut [f32] as *mut [f32; 4];
            safe_arch::store_unaligned_m128(&mut *target, self);
        }
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
        masks::blend_varying_m128d(self, other, mask)
    }

    #[inline]
    fn shift_left(self, offset: usize) -> Self {
        safe_arch::cast_to_m128d_from_m128i(lane_shl_i32_m128i(
            safe_arch::cast_to_m128i_from_m128d(self),
            2 * offset,
        ))
    }

    #[inline]
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

    #[inline]
    fn store(self, target: &mut [f64]) {
        assert_eq!(target.len(), 2);
        // Safe due to assertion above
        unsafe {
            let target = target as *mut [f64] as *mut [f64; 2];
            safe_arch::store_unaligned_m128d(&mut *target, self);
        }
    }
}

#[inline]
fn lane_shl_i32_m128i(v: m128i, offset: usize) -> m128i {
    match offset {
        0 => v,
        1 => safe_arch::byte_shl_imm_u128_m128i::<4>(v),
        2 => safe_arch::byte_shl_imm_u128_m128i::<8>(v),
        3 => safe_arch::byte_shl_imm_u128_m128i::<12>(v),
        _ => safe_arch::set_splat_i32_m128i(0),
    }
}

#[inline]
fn lane_shr_i32_m128i(v: m128i, offset: usize) -> m128i {
    match offset {
        0 => v,
        1 => safe_arch::byte_shr_imm_u128_m128i::<4>(v),
        2 => safe_arch::byte_shr_imm_u128_m128i::<8>(v),
        3 => safe_arch::byte_shr_imm_u128_m128i::<12>(v),
        _ => safe_arch::set_splat_i32_m128i(0),
    }
}
