//! SIMD-friendly concentration storage

#[cfg(feature = "safe_arch")]
mod safe_arch;
#[cfg(feature = "slipstream")]
mod slipstream;

use super::{Concentration, ScalarConcentration};
use crate::{
    array2, array_each_mut,
    parameters::{stencil_offset, STENCIL_SHAPE},
    Precision,
};
use ndarray::{s, Array2, ArrayView2, ArrayViewMut2, Axis};
use std::{
    array,
    convert::Infallible,
    ops::{BitAnd, Range},
};

/// SIMD-oriented Concentration implementation
///
/// As always in SIMD, this storage has a fairly non-obvious data layout. Let us
/// consider a computation which would be implemented in a scalar way using a 2D
/// array of floating-point numbers that has N rows and M columns.
///
/// We will impose that N is a multiple of the SIMD vector width W, denote
/// L = N/W their ratio, and show through the following ASCII art how this
/// storage would be laid out in terms of the equivalent scalar storage mIJ, in
/// the special case of a stencil of offset [1, 1] and SIMD vectors of width 2.
///
/// ```text
/// [0 0] [0       mL1    ] [0       mL2    ] ... [0       mLM    ] [0 0]
/// [0 0] [m11     m(L+1)1] [m12     m(L+1)2] ... [m1M     m(L+1)M] [0 0]
/// [0 0] [m21     m(L+2)1] [m22     m(L+2)2] ... [m2M     m(L+2)M] [0 0]
/// ...
/// [0 0] [mL1     m(2L)1 ] [mL2     m(2L)2 ] ... [mLM     m(2L)M ] [0 0]
/// [0 0] [m(L+1)1 0      ] [m(L+1)2 0      ] ... [m(L+1)M 0      ] [0 0]
/// ```
///
/// In a nutshell...
/// - The left and right edges of the array are padded with `stencil_offset[1]`
///   vectors of zeroes in order to regularize the computation and avoid 4K
///   aliasing in the common case where the central width is a power of 2.
/// - Every column of the matrix is sliced into W sub-columns of length N, which
///   are distributed across the lanes of the vectors in the center region (the
///   first vector contains the first element of the first sub-column, the
///   first element of the second sub-column, and so on).
/// - The top and the bottom of the matrix is padded with `stencil_offset[0]`
///   elements that respectively replicate the end of the previous sub-column
///   and the beginning of the next sub-column.
///
/// In this way, we can keep the same stencil access pattern that we used for
/// scalar computations, but in a SIMD data layout.
///
/// We can generalize to situations where N is not a multiple of W by imposing
/// that the last few elements of each column be set to zero, at the same time
/// where we would impose that the top/bottom padding matches the beginning/end
/// of central content vectors.
pub struct SIMDConcentration<const WIDTH: usize, Vector: SIMDValues<WIDTH>> {
    /// SIMD data storage
    simd: Array2<Vector>,

    /// Scalar data storage (lazily allocated on first use)
    scalar: ScalarConcentration,
}
//
impl<const WIDTH: usize, Vector: SIMDValues<WIDTH>> SIMDConcentration<WIDTH, Vector> {
    /// Read-only view of the SIMD data storage
    pub fn view(&self) -> ArrayView2<Vector> {
        self.simd.view()
    }

    /// Mutable view of the central region, without stencil edges
    pub fn simd_center_mut(&mut self) -> ArrayViewMut2<Vector> {
        let [center_rows, center_cols] = self.simd_center_range();
        self.simd.slice_mut(s![center_rows, center_cols])
    }

    /// Shape of the simd data store, given a scalar shape
    fn simd_shape([rows, cols]: [usize; 2]) -> [usize; 2] {
        // TODO: Remove this restriction if I later publish this as a general crate
        assert_eq!(
            rows % WIDTH,
            0,
            "For now, the number of rows must be a multiple of the SIMD width"
        );
        let simd_shape = [rows / WIDTH, cols];
        array2(|i| simd_shape[i] + STENCIL_SHAPE[i] - 1)
    }

    /// Fix up the output of the ndarray broadcasting constructor
    fn from_scalar_elem(shape: [usize; 2], elem: Precision) -> Self {
        // Build the SIMD array, initially full of broadcasted elements
        let elem = Vector::splat(elem);
        let shape = Self::simd_shape(shape);
        let mut simd = Array2::from_elem(shape, elem);

        // Zero out the left and right edges
        let zero = Vector::splat(0.0);
        let [_, left_offset] = stencil_offset();
        simd.slice_mut(s![.., ..left_offset]).fill(zero);
        let right_offset = simd.ncols() - left_offset;
        simd.slice_mut(s![.., right_offset..]).fill(zero);

        // Finish constructing the SIMDConcentration
        // (the outer stencil remains incorrect, it will be fixed by finalize())
        Self {
            simd,
            scalar: <Array2<Precision> as Default>::default(),
        }
    }

    /// Determine the range of indices that corresponds to the center of the
    /// array, without stencil edges.
    fn simd_center_range(&self) -> [Range<usize>; 2] {
        let [top_offset, left_offset] = stencil_offset();
        let right_offset = self.simd.ncols() - left_offset;
        let bottom_offset = self.simd.nrows() - top_offset;
        [top_offset..bottom_offset, left_offset..right_offset]
    }

    /// Make an array of size WIDTH from an index -> value mapping
    fn array<T>(f: impl FnMut(usize) -> T) -> [T; WIDTH] {
        array::from_fn(f)
    }

    /// Borrow-checker friendly implementation of write_scalar_view
    ///
    /// Assumes `validate_write()` has already been called by the caller, and
    /// assumes `simd_center` is the `simd_center_range()` of `self.simd`.
    fn write_scalar_view_impl(simd_center: ArrayView2<Vector>, target: ArrayViewMut2<Precision>) {
        // Check domain size
        let num_simd_cols = simd_center.ncols();

        // Split output matrix into one submatrix per SIMD vector lane
        let mut scalar_views = Self::split_scalar_matrix(target);
        debug_assert!(scalar_views
            .iter()
            .all(|view| view.shape() == simd_center.shape()));

        // Iterate over rows of the SIMD matrix and matching rows of the scalar matrices
        let mut scalar_rows_iter =
            array_each_mut(&mut scalar_views).map(|view| view.rows_mut().into_iter());
        for simd_row in simd_center.rows().into_iter() {
            let mut scalar_rows = array_each_mut(&mut scalar_rows_iter)
                .map(|rows| rows.next().expect("Unexpected scalar row iterator end"));

            // Iterate over sets of WIDTH consecutive columns of the SIMD row
            // These correspond to sets of WIDTH consecutive columns in the
            // scalar matrix, although interleaved across SIMD vectors.
            let mut scalar_chunks_iter =
                array_each_mut(&mut scalar_rows).map(|row| row.exact_chunks_mut(WIDTH).into_iter());
            for simd_chunk in simd_row.exact_chunks(WIDTH) {
                let mut scalar_chunks = array_each_mut(&mut scalar_chunks_iter)
                    .map(|iter| iter.next().expect("Unexpected scalar chunk iterator end"));

                // Use SIMD transpose to produce SIMD vectors that correspond to
                // WIDTH consecutive columns of a row of the scalar matrix
                let transposed_chunk_arr = Vector::transpose(Self::array(|i| simd_chunk[i]));

                // Write back this data into the scalar array
                for (transposed_chunk, scalar_chunk) in
                    (transposed_chunk_arr.into_iter()).zip(&mut scalar_chunks)
                {
                    transposed_chunk.store(
                        scalar_chunk
                            .as_slice_mut()
                            .expect("Unexpected non-contiguous scalar row chunk"),
                    );
                }
            }

            // Handle remainder by breaking down SIMD vectors into scalars
            let trailing_cols_offset = (num_simd_cols / WIDTH) * WIDTH;
            let simd_cols = simd_row.slice(s![trailing_cols_offset..]);
            let mut scalar_cols_iter = array_each_mut(&mut scalar_rows)
                .map(|row| row.slice_mut(s![trailing_cols_offset..]).into_iter());
            for simd_col in simd_cols {
                let scalar_cols = array_each_mut(&mut scalar_cols_iter)
                    .map(|iter| iter.next().expect("Unexpected scalar columns iterator end"));

                // Turn SIMD vector into scalar elements and write them down
                let simd_array: [Precision; WIDTH] = (*simd_col).into();
                for (simd_lane, target_col) in (simd_array.into_iter()).zip(scalar_cols) {
                    *target_col = simd_lane;
                }
            }
        }
    }

    /// Split the scalar matrix into a number of submatrices, each
    /// corresponding to one lane of a SIMD vector (check layout description
    /// in the SIMDConcentration documentation above if you are confused)
    fn split_scalar_matrix(scalars: ArrayViewMut2<Precision>) -> [ArrayViewMut2<Precision>; WIDTH] {
        let num_simd_rows = scalars.nrows() / WIDTH;
        let mut remainder_opt = Some(scalars);
        Self::array(move |i| {
            let remainder = remainder_opt
                .take()
                .expect("Shouldn't happen because the Option is reset below");
            if i < WIDTH - 1 {
                let (chunk, new_remainder) = remainder.split_at(Axis(0), num_simd_rows);
                remainder_opt = Some(new_remainder);
                chunk
            } else {
                remainder
            }
        })
    }
}
//
impl<const WIDTH: usize, Vector: SIMDValues<WIDTH>> Concentration
    for SIMDConcentration<WIDTH, Vector>
{
    type Context = ();

    type Error = Infallible;

    fn default(_context: &mut (), shape: [usize; 2]) -> Result<Self, Infallible> {
        Ok(Self::from_scalar_elem(shape, Precision::default()))
    }

    fn zeros(_context: &mut (), shape: [usize; 2]) -> Result<Self, Infallible> {
        Ok(Self::from_scalar_elem(shape, 0.0))
    }

    fn ones(_context: &mut (), shape: [usize; 2]) -> Result<Self, Infallible> {
        Ok(Self::from_scalar_elem(shape, 1.0))
    }

    fn shape(&self) -> [usize; 2] {
        let full_shape = self.raw_shape();
        let center_shape = array2(|i| full_shape[i] - STENCIL_SHAPE[i] + 1);
        [center_shape[0] * WIDTH, center_shape[1]]
    }

    /// Shape of the inner array of SIMD vectors
    fn raw_shape(&self) -> [usize; 2] {
        let &[rows, cols] = self.simd.shape() else { panic!("2D array should have 2D shape") };
        [rows, cols]
    }

    fn fill_slice(
        &mut self,
        _context: &mut (),
        [scalar_rows, cols]: [Range<usize>; 2],
        value: Precision,
    ) -> Result<(), Infallible> {
        // Ignore stencil edges which are not affected by this operation
        let mut simd_center = self.simd_center_mut();

        // Prepare to track which scalar row index we're looking at inside of
        // each SIMD vector lane
        let row = |i: usize| i32::try_from(i).expect("Too many rows for this impl");
        let start_row = Vector::Indices::splat(row(scalar_rows.start));
        let end_row = Vector::Indices::splat(row(scalar_rows.end));
        let num_simd_rows = simd_center.nrows();
        let mut current_row =
            Vector::Indices::from_idx_array(Self::array(|i| row(num_simd_rows * i)));

        // Update relevant rows and columns of the SIMD array
        let all_false = Vector::Mask::splat(false);
        let value = Vector::splat(value);
        for simd_row in 0..num_simd_rows {
            let mask = current_row.ge(start_row) & current_row.lt(end_row);
            if mask != all_false {
                for simd in simd_center.slice_mut(s![simd_row, cols.clone()]) {
                    *simd = simd.blend(value, mask);
                }
            }
            current_row.increment();
        }
        Ok(())
    }

    fn finalize(&mut self, _context: &mut ()) -> Result<(), Infallible> {
        // Ignore the left and right edge, these should stay at 0 forever
        let [center_rows, center_cols] = self.simd_center_range();
        debug_assert!((self.simd.slice(s![.., ..center_cols.start]).iter())
            .chain(self.simd.slice(s![.., center_cols.end..]).iter())
            .all(|&v| v == Vector::splat(0.0)));
        let center_cols = self.simd.slice_mut(s![.., center_cols]);

        // Separate the top and bottom rows that we need to fill from the
        // center of the stencil that was filled by previous operations.
        let (mut top, center_and_bottom) = center_cols.split_at(Axis(0), center_rows.start);
        let (center, mut bottom) =
            center_and_bottom.split_at(Axis(0), center_rows.end - center_rows.start);

        // Iterate over bottom rows to be filled, in blocks of constant SIMD
        // lane shift that keep getting further away from the vertical center
        let mut top_end = top.nrows();
        for (block_idx, mut bottom_block) in bottom
            .axis_chunks_iter_mut(Axis(0), center.nrows())
            .enumerate()
        {
            // Figure out a top block that matches bottom block's characteristics
            let mut top_block = top.slice_mut(s![top_end - bottom_block.nrows()..top_end, ..]);
            top_end -= bottom_block.nrows();

            // Are we still in a range where the stencil edges should contain data?
            let shift = block_idx + 1;
            if shift < WIDTH {
                // If so, fill that data at the bottom...
                for (dst, src) in bottom_block.iter_mut().zip(&center) {
                    *dst = src.shift_right(shift);
                }

                // ...and at the top
                let top_src = center.slice(s![center.nrows() - top_block.nrows().., ..]);
                for (dst, src) in (top_block.iter_mut()).zip(&top_src) {
                    *dst = src.shift_left(shift);
                }
            } else {
                // Otherwise, just fill everything with zeroes
                top_block.fill(Vector::splat(0.0));
                bottom_block.fill(Vector::splat(0.0));
            }
        }
        Ok(())
    }

    type ScalarView<'a> = ArrayView2<'a, Precision>;

    fn make_scalar_view(&mut self, _context: &mut ()) -> Result<Self::ScalarView<'_>, Infallible> {
        // Lazily allocate the internal scalar data storage
        if self.scalar.is_empty() {
            self.scalar = ScalarConcentration::default(self.shape());
        }

        // Extract the center of the SIMD domain
        let [center_rows, center_cols] = self.simd_center_range();
        let simd_center = self.simd.slice(s![center_rows, center_cols]);

        // Perform the write
        Self::write_scalar_view_impl(simd_center, self.scalar.view_mut());

        // Emit the scalar results
        Ok(self.scalar.view())
    }

    fn write_scalar_view(
        &mut self,
        _context: &mut (),
        target: ArrayViewMut2<Precision>,
    ) -> Result<(), Self::Error> {
        // Check that the target dimensions are correct
        Self::validate_write(&self, &target);

        // Extract the center of the SIMD domain
        let [center_rows, center_cols] = self.simd_center_range();
        let simd_center = self.simd.slice(s![center_rows, center_cols]);

        // Perform the write
        Self::write_scalar_view_impl(simd_center, target);
        Ok(())
    }
}

/// SIMD vector of floating-point values, as needed by SIMDConcentration
pub trait SIMDValues<const WIDTH: usize, Element = Precision>:
    Copy + PartialEq + Into<[Element; WIDTH]> + 'static
{
    /// Matching vector type for vector of indices
    type Indices: SIMDIndices<WIDTH, Mask = Self::Mask>;

    /// Matching mask type for blending
    type Mask: SIMDMask<WIDTH>;

    /// Broadcast a scalar value into all lanes of a vector
    fn splat(x: Element) -> Self;

    /// Imports enabled lanes from other, keeps disabled lanes from self
    fn blend(self, other: Self, mask: Self::Mask) -> Self;

    /// Shift vector lanes to the left, leaving zeros behind
    ///
    /// A left shift of 2 turns `|x1 x2 ... xN|` into `|x3 x4 ... xN 0 0|`.
    fn shift_left(self, offset: usize) -> Self;

    /// Shift vector lanes to the right, leaving zeros behind
    ///
    /// A right shift of 2 turns `|x1 x2 ... xN|` into `|0 0 x1 x2 ... xN-2|`.
    fn shift_right(self, offset: usize) -> Self;

    /// SIMD-sized matrix transpose
    ///
    /// Let there be a set of N vectors of width N:
    ///
    /// `|m11 m12 ... m1N|`, `|m21 m22 ... m2N|`, ..., `|mN1 mN2 ... mNN|`
    ///
    /// This operation turns it into the following set where the role of
    /// lanes and vectors is flipped:
    ///
    /// `|m11 m21 ... mN1|`, `|m12 m22 ... mN2|`, ..., `|m1N m2N ... mNN|`
    fn transpose(matrix: [Self; WIDTH]) -> [Self; WIDTH];

    /// Store vector into scalar storage, which must be suitably sized
    fn store(self, target: &mut [Element]);
}

/// Vector of indices
///
/// We use i32 for indices because x86 SIMD instruction sets have poor support
/// for unsigned integers and we don't want indices to be larger than data and
/// become the vectorization bottleneck.
pub trait SIMDIndices<const WIDTH: usize>: Copy {
    /// Matching mask type for blending
    type Mask: SIMDMask<WIDTH>;

    /// Turn an array of indices into this type
    fn from_idx_array(arr: [i32; WIDTH]) -> Self;

    /// Broadcast a scalar value into all lanes of a vector
    fn splat(x: i32) -> Self;

    /// Increment all lanes of the vector
    fn increment(&mut self);

    /// Compare with another vectors of indices and indicate which lanes are >=
    fn ge(self, other: Self) -> Self::Mask;

    /// Compare with another vectors of indices and indicate which lanes are <
    fn lt(self, other: Self) -> Self::Mask;
}

/// Vector of bool-like values
pub trait SIMDMask<const WIDTH: usize>: BitAnd<Output = Self> + Copy + PartialEq {
    /// Broadcast a scalar value into all lanes of a vector
    fn splat(b: bool) -> Self;
}
