//! Implementation of SIMDXyz traits for safe_arch types
//!
//! In future Rust, I'll be able to drop this and use stdsimd instead.

#[cfg(target_feature = "sse2")]
mod xmm;
#[cfg(target_feature = "avx")]
mod ymm;
