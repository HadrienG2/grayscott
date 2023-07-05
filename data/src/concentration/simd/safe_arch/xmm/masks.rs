//! SIMD masking for SSE

use crate::concentration::simd::SIMDMask;
use safe_arch::{m128, m128d, m128i};

// We need blend for efficient `fill_slice()`, and that's SSE 4.1.
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
