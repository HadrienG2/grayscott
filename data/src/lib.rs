//! Data format used by the Gray-Scott reaction simulation

pub mod concentration;
pub mod hdf5;
pub mod parameters;

/// Computation precision
pub type Precision = f32;

/// Shorthand to create an array of 2 elements, knowing an index -> value mapping
#[inline(always)]
pub fn array2<T>(f: impl FnMut(usize) -> T) -> [T; 2] {
    std::array::from_fn::<T, 2, _>(f)
}
