//! Data format used by the Gray-Scott reaction simulation

pub mod concentration;
pub mod hdf5;
pub mod parameters;

use std::array;

/// Computation precision
pub type Precision = f32;

/// Shorthand to create an array of 2 elements, knowing an index -> value mapping
#[inline(always)]
pub fn array2<T>(f: impl FnMut(usize) -> T) -> [T; 2] {
    array::from_fn::<T, 2, _>(f)
}

/// Equivalent of unstable array.each_mut()
#[inline(always)]
fn array_each_mut<T, const N: usize>(a: &mut [T; N]) -> [&mut T; N] {
    let ptr = a.as_mut_ptr();
    array::from_fn(|i| unsafe { &mut *ptr.offset(i as isize) })
}
