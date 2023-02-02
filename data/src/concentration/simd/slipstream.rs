//! Implementation of SIMDXyz traits for slipstream types

use super::{SIMDIndices, SIMDMask, SIMDValues};
use crate::Precision;
use slipstream::{
    vector::{align::Align, Masked},
    Mask, Vector,
};
use std::{array, ops::BitAnd};

impl<A: Align, M, const WIDTH: usize> SIMDMask<WIDTH> for Vector<A, M, WIDTH>
where
    M: Mask + BitAnd<Output = M>,
{
    #[inline]
    fn splat(b: bool) -> Self {
        Vector::splat(M::from_bool(b))
    }
}

impl<A: Align, const WIDTH: usize> SIMDIndices<WIDTH> for Vector<A, i32, WIDTH> {
    type Mask = <Self as Masked>::Mask;

    #[inline]
    fn from_array(arr: [i32; WIDTH]) -> Self {
        Vector::new(&arr[..])
    }

    #[inline]
    fn splat(x: i32) -> Self {
        Vector::splat(x)
    }

    #[inline]
    fn increment(&mut self) {
        let ones = Self::splat(1);
        *self += ones;
    }

    #[inline]
    fn ge(self, other: Self) -> Self::Mask {
        self.ge(other)
    }

    #[inline]
    fn lt(self, other: Self) -> Self::Mask {
        self.lt(other)
    }
}

impl<A: Align, const WIDTH: usize> SIMDValues<WIDTH> for Vector<A, Precision, WIDTH> {
    type Indices = Vector<A, i32, WIDTH>;

    type Mask = <Self as Masked>::Mask;

    #[inline]
    fn splat(x: Precision) -> Self {
        Vector::splat(x)
    }

    #[inline]
    fn blend(self, other: Self, mask: Self::Mask) -> Self {
        Vector::blend(self, other, mask)
    }

    #[inline]
    fn shift_left(self, offset: usize) -> Self {
        let arr = self.into_array();
        values_from_fn(|i| *arr.get(i.wrapping_sub(offset)).unwrap_or(&0.0))
    }

    #[inline]
    fn shift_right(self, offset: usize) -> Self {
        let arr = self.into_array();
        values_from_fn(|i| *arr.get(i + offset).unwrap_or(&0.0))
    }

    #[inline]
    fn transpose(matrix: [Self; WIDTH]) -> [Self; WIDTH] {
        let matrix = matrix.map(Self::into_array);
        array::from_fn(|target_vec| values_from_fn(|target_lane| matrix[target_lane][target_vec]))
    }

    #[inline]
    fn store(self, target: &mut [Precision]) {
        Vector::store(self, target)
    }

    #[inline]
    fn into_array(self) -> [Precision; WIDTH] {
        let arr: &[Precision; WIDTH] = self.as_ref();
        arr.clone()
    }
}

/// Build a slipstream vector from a SIMD array of elements
#[inline]
fn values_from_array<A: Align, const N: usize>(array: [Precision; N]) -> Vector<A, Precision, N> {
    Vector::new(&array[..])
}

/// Build a slipstream vector from a lane index -> element mapping
#[inline]
fn values_from_fn<A: Align, const N: usize>(
    f: impl FnMut(usize) -> Precision,
) -> Vector<A, Precision, N> {
    values_from_array(array::from_fn(f))
}
