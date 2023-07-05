//! SIMD masking for AVX

use crate::concentration::simd::SIMDMask;
use safe_arch::{m256, m256d};

impl SIMDMask<4> for m256d {
    #[inline]
    fn splat(b: bool) -> Self {
        Self::from_bits([b as u64 * u64::MAX; 4])
    }
}

impl SIMDMask<8> for m256 {
    #[inline]
    fn splat(b: bool) -> Self {
        Self::from_bits([b as u32 * u32::MAX; 8])
    }
}
