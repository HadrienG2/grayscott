//! Data format used by the Gray-Scott reaction simulation

pub mod concentration;
#[cfg(feature = "hdf5")]
pub mod hdf5;
pub mod parameters;

use std::array;

/// Computation precision
pub type Precision = f32;

/// Shorthand to create an array of 2 elements, knowing an index -> value mapping
#[inline]
pub fn array2<T>(f: impl FnMut(usize) -> T) -> [T; 2] {
    array::from_fn::<T, 2, _>(f)
}

/// Equivalent of the unstable `<[T; N]>::each_mut()` function
#[allow(unused)]
#[inline]
fn array_each_mut<T, const N: usize>(a: &mut [T; N]) -> [&mut T; N] {
    let ptr = a.as_mut_ptr();
    // Safe because it's just an array map that works around a borrow checker limitation
    array::from_fn(|i| unsafe { &mut *ptr.add(i) })
}
