//! Concentration of chemical species

#[cfg(feature = "gpu")]
pub mod gpu;
#[cfg(feature = "simd")]
pub mod simd;

use crate::array2;
use crate::Precision;
use ndarray::{s, Array2, ArrayView2, ArrayViewMut2};
use std::convert::Infallible;
use std::ops::Range;

/// Concentration of all species involved
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub struct Species<C: Concentration> {
    /// Context for manipulating concentrations
    context: C::Context,

    /// Concentration of species U
    u: Evolving<C>,

    /// Concentration of species V
    v: Evolving<C>,
}
//
impl<C: Concentration> Species<C> {
    /// Set up species concentration storage as the C++ version does
    ///
    /// - `context` provides some additional context needed to set up storage.
    ///   For simpler containers, this will be (), but more advanced use cases
    ///   like GPU need this possibility.
    /// - `shape` specifies the concentration matrix dimensions in
    ///   [rows, columns] format, e.g. [1080, 1920]
    pub fn new(mut context: C::Context, shape: [usize; 2]) -> Result<Self, C::Error> {
        // Start with U = 1.0 and V = 0.0 everywhere
        let mut u = Evolving::<C>::ones_out(&mut context, shape)?;
        let mut v = Evolving::<C>::zeros_out(&mut context, shape)?;
        let out_u = u.out();
        let out_v = v.out();

        // Add V and remove U at the center of images
        let num_range = [7, 8];
        let frac = 16;
        let row_shift = 4;
        let center_slice = array2(|i| {
            let shift = (i == 0) as usize * row_shift;
            let [start, end] = array2(|j| (shape[i] * num_range[j] / frac).saturating_sub(shift));
            start..end
        });
        out_u.fill_slice(&mut context, center_slice.clone(), 0.0)?;
        out_v.fill_slice(&mut context, center_slice, 1.0)?;

        // Make the newly generated concentrations become the input ones
        let mut result = Self { context, u, v };
        result.flip()?;
        Ok(result)
    }

    /// Check out the concentration context
    ///
    /// More advanced concentration types like ImageConcentration provide a way
    /// to cache quantities which are specific to this species concentration
    /// storage, but do not belong to individual concentration tables.
    pub fn context(&mut self) -> &mut C::Context {
        &mut self.context
    }

    /// Check out the shape of the concentration matrices
    pub fn shape(&self) -> [usize; 2] {
        self.u.shape()
    }

    /// Check out the raw shape of the concentration matrices
    pub fn raw_shape(&self) -> [usize; 2] {
        self.u.raw_shape()
    }

    /// Access the concentration matrices in (in_u, in_v, out_u, out_v) order
    pub fn in_out(&mut self) -> (&C, &C, &mut C, &mut C) {
        let (in_u, out_u) = self.u.in_out();
        let (in_v, out_v) = self.v.in_out();
        (in_u, in_v, out_u, out_v)
    }

    /// Make the output concentrations become the input ones and vice versa
    pub fn flip(&mut self) -> Result<(), C::Error> {
        self.u.flip(&mut self.context)?;
        self.v.flip(&mut self.context)?;
        Ok(())
    }

    /// Access V's current input concentration
    ///
    /// The concentration of the V species is the effective result of the
    /// simulation (what gets stored to HDF5, etc).
    pub fn access_result<'s, R: 's>(
        &'s mut self,
        f: impl FnOnce(&'s mut C, &mut C::Context) -> R,
    ) -> R {
        f(&mut self.v.0[0], &mut self.context)
    }

    /// Access a scalar view of V's current input concentration
    ///
    /// The concentration of the V species is the effective result of the
    /// simulation (what gets stored to HDF5, etc). For some compute backends
    /// (SIMD, GPU...), some expensive preprocessing steps may be needed in
    /// order to get to a normal 2D view of it. This is how you get this done.
    ///
    /// Use `make_scalar_view` if you can directly use the borrowed data in a
    /// zero-copy manner. If you would need to clone it to get an owned
    /// version, prefer allocating your own storage and writing to it using
    /// `write_scalar_view`, ideally with allocation reuse.
    pub fn make_result_view(&mut self) -> Result<C::ScalarView<'_>, C::Error> {
        self.access_result(|v, ctx| v.make_scalar_view(ctx))
    }

    /// Write down the result into an externally allocated 2D array of scalars
    ///
    /// The concentration of the V species is the effective result of the
    /// simulation (what gets stored to HDF5, etc). For some compute backends
    /// (SIMD, GPU...), some expensive preprocessing steps may be needed in
    /// order to get to a normal 2D view of it. This is how you get this done.
    ///
    /// Use `write_scalar_view` if you need owned data (e.g. to send it to
    /// another thread). If a borrow is enough, prefer `make_scalar_view`.
    ///
    /// # Panics
    ///
    /// Panics if the target does not have the same shape as this table.
    pub fn write_result_view(&mut self, target: ArrayViewMut2<Precision>) -> Result<(), C::Error> {
        self.access_result(|v, ctx| v.write_scalar_view(ctx, target))
    }
}

/// Pair of Concentrations where one acts as an input and the other as an output
#[derive(Clone, Debug, PartialEq)]
pub struct Evolving<C: Concentration>([C; 2]);
//
impl<C: Concentration> Evolving<C> {
    /// Access the input and output concentration
    pub fn in_out(&mut self) -> (&C, &mut C) {
        let [input, output] = &mut self.0;
        (input, output)
    }

    /// Set up concentration storage with all-zeros output concentration
    fn zeros_out(context: &mut C::Context, shape: [usize; 2]) -> Result<Self, C::Error> {
        Ok(Self([
            C::default(context, shape)?,
            C::zeros(context, shape)?,
        ]))
    }

    /// Set up concentration storage with all-ones output concentration
    fn ones_out(context: &mut C::Context, shape: [usize; 2]) -> Result<Self, C::Error> {
        Ok(Self([
            C::default(context, shape)?,
            C::ones(context, shape)?,
        ]))
    }

    /// Check the shape of concentration matrices
    fn shape(&self) -> [usize; 2] {
        self.0[0].shape()
    }

    /// Check the raw shape of concentration matrices
    fn raw_shape(&self) -> [usize; 2] {
        self.0[0].raw_shape()
    }

    /// Access the output concentration
    fn out(&mut self) -> &mut C {
        &mut self.0[1]
    }

    /// Make the output concentration become the input one
    fn flip(&mut self, context: &mut C::Context) -> Result<(), C::Error> {
        let [input, output] = &mut self.0;
        output.finalize(context)?;
        std::mem::swap(input, output);
        Ok(())
    }
}

/// Tabulated concentration of a chemical species
///
/// In the following, shape refers to a [rows, cols] array indicating the
/// dimensions of the useful region of the storage, in scalar numbers.
///
/// SIMD-friendly implementations may choose to allocate more storage, but
/// should not expose any of it through methods of this trait. For all intents
/// and purpose, they should behave as if an ndarray of scalars of the right
/// dimensions had been allocated.
pub trait Concentration: Sized {
    /// Parameters needed to initialize and operate on the concentration storage
    type Context: Sized;

    /// Errors returned from methods
    type Error: std::error::Error + Send + Sync + Sized + 'static;

    /// Create an array of a certain shape, whose contents are meant to be overwritten
    fn default(context: &mut Self::Context, shape: [usize; 2]) -> Result<Self, Self::Error>;

    /// Create an array of a certain shape, filled with zeros
    ///
    /// You must call finalize() once you're done modifying the array (if you're
    /// using Evolving::flip, it takes care of this)
    fn zeros(context: &mut Self::Context, shape: [usize; 2]) -> Result<Self, Self::Error>;

    /// Create an array of a certain shape, filled with ones
    ///
    /// You must call finalize() once you're done modifying the array (if you're
    /// using Evolving::flip, it takes care of this)
    fn ones(context: &mut Self::Context, shape: [usize; 2]) -> Result<Self, Self::Error>;

    /// Retrieve the shape that was passed in to the constructor
    fn shape(&self) -> [usize; 2];

    /// Retrieve the shape of the internal data store
    ///
    /// The interpretation of this value is implementation-dependent.
    fn raw_shape(&self) -> [usize; 2] {
        self.shape()
    }

    /// Fill a certain slice of the matrix with a certain value
    ///
    /// Like the constructor's shape, this slice is expressed in units of scalar
    /// numbers, as if the storage were a 2D ndarray of scalars of the size
    /// specified at construction time.
    ///
    /// You must call finalize() once you're done modifying the array (if you're
    /// using Evolving::flip, it takes care of this)
    fn fill_slice(
        &mut self,
        context: &mut Self::Context,
        slice: [Range<usize>; 2],
        value: Precision,
    ) -> Result<(), Self::Error>;

    /// Finalize the output table before making it the input
    ///
    /// Implement this when the data layout calls for some expensive operation
    /// (e.g. duplication, zeroing of elements, submission of GPU commands...),
    /// which we don't want to perform on every operation but only when the
    /// final output is going to be read.
    fn finalize(&mut self, _context: &mut Self::Context) -> Result<(), Self::Error> {
        Ok(())
    }

    /// Reinterpretation of the matrix as a 2D array of scalars
    type ScalarView<'a>: AsScalars
    where
        Self: 'a;

    /// Transform the table into a 2D array of scalars, or expose it as such
    ///
    /// This operation may require reshuffling data, do not use it frequently.
    ///
    /// Use `make_scalar_view` if you can directly use the borrowed data in a
    /// zero-copy manner. If you would need to clone it to get an owned
    /// version, prefer allocating your own storage and writing to it using
    /// `write_scalar_view`, ideally with allocation reuse.
    fn make_scalar_view(
        &mut self,
        context: &mut Self::Context,
    ) -> Result<Self::ScalarView<'_>, Self::Error>;

    /// Write down the data into an externally allocated 2D array of scalars
    ///
    /// This operation may require reshuffling data and will incur expensive
    /// memory traffic, do not use it frequently.
    ///
    /// Use `write_scalar_view` if you need owned data (e.g. to send it to
    /// another thread). If a borrow is enough, prefer `make_scalar_view`.
    ///
    /// # Panics
    ///
    /// Panics if the target does not have the same shape as this table.
    fn write_scalar_view(
        &mut self,
        context: &mut Self::Context,
        target: ArrayViewMut2<Precision>,
    ) -> Result<(), Self::Error>;

    /// Target validation for `write_scalar_view()` implementations
    fn validate_write(&self, target: &ArrayViewMut2<Precision>) {
        let [rows, cols] = self.shape();
        assert_eq!(target.nrows(), rows);
        assert_eq!(target.ncols(), cols);
    }
}
//
/// Data that can be reinterpreted as a 2D ndarray of scalar data
pub trait AsScalars {
    fn as_scalars(&self) -> ArrayView2<Precision>;
}

/// Straightforward Concentration implementation based on an ndarray of floats
pub type ScalarConcentration = Array2<Precision>;
//
impl Concentration for ScalarConcentration {
    type Context = ();

    type Error = Infallible;

    fn default(_context: &mut (), shape: [usize; 2]) -> Result<Self, Infallible> {
        Ok(ScalarConcentration::default(shape))
    }

    fn zeros(_context: &mut (), shape: [usize; 2]) -> Result<Self, Infallible> {
        Ok(ScalarConcentration::zeros(shape))
    }

    fn ones(_context: &mut (), shape: [usize; 2]) -> Result<Self, Infallible> {
        Ok(ScalarConcentration::ones(shape))
    }

    fn shape(&self) -> [usize; 2] {
        let [rows, cols] = ScalarConcentration::shape(self) else { panic!("Expected 2D shape") };
        [*rows, *cols]
    }

    fn fill_slice(
        &mut self,
        _context: &mut (),
        slice: [Range<usize>; 2],
        value: Precision,
    ) -> Result<(), Infallible> {
        let slice = s![slice[0].clone(), slice[1].clone()];
        self.slice_mut(slice).fill(value);
        Ok(())
    }

    type ScalarView<'a> = ArrayView2<'a, Precision>;

    fn make_scalar_view(&mut self, _context: &mut ()) -> Result<Self::ScalarView<'_>, Infallible> {
        Ok(self.view())
    }

    fn write_scalar_view(
        &mut self,
        _context: &mut (),
        mut target: ArrayViewMut2<Precision>,
    ) -> Result<(), Self::Error> {
        Self::validate_write(&self, &target);
        target.assign(&self);
        Ok(())
    }
}
//
impl AsScalars for ArrayView2<'_, Precision> {
    fn as_scalars(&self) -> ArrayView2<Precision> {
        self.reborrow()
    }
}
