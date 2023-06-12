//! Concentration of chemical species

#[cfg(feature = "gpu")]
pub mod gpu;
pub mod simd;

use crate::array2;
use crate::Precision;
use ndarray::{s, Array2, ArrayView2};
use std::borrow::Borrow;
use std::convert::Infallible;
use std::ops::Range;

/// Concentration of all species involved
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub struct Species<C: Concentration> {
    /// Context for manipulating concentrations
    context: C::Context,

    /// Concentration of species U
    pub u: Evolving<C>,

    /// Concentration of species V
    pub v: Evolving<C>,
}
//
impl<C: Concentration> Species<C> {
    /// Set up species concentration storage as the C++ version does
    ///
    /// `shape` specifies the concentration matrix dimensions, e.g. [1080, 1920]
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

    /// Check out the shape of the concentration matrices
    pub fn shape(&self) -> [usize; 2] {
        self.u.shape()
    }

    /// Check out the raw shape of the concentration matrices
    pub fn raw_shape(&self) -> [usize; 2] {
        self.u.raw_shape()
    }

    /// Make the output concentrations become the input ones
    pub fn flip(&mut self) -> Result<(), C::Error> {
        self.u.flip(&mut self.context)?;
        self.v.flip(&mut self.context)?;
        Ok(())
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

    /// View the input concentration as a 2D array of scalars
    pub fn make_scalar_input_view(&mut self) -> Result<C::ScalarView<'_>, C::Error> {
        self.0[0].make_scalar_view()
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
    type Error: Sized;

    /// Create an array of a certain shape, filled with defaulted elements
    fn default(context: &mut Self::Context, shape: [usize; 2]) -> Result<Self, Self::Error>;

    /// Create an array of a certain shape, filled with zeros
    fn zeros(context: &mut Self::Context, shape: [usize; 2]) -> Result<Self, Self::Error>;

    /// Create an array of a certain shape, filled with ones
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
    fn fill_slice(
        &mut self,
        context: &mut Self::Context,
        slice: [Range<usize>; 2],
        value: Precision,
    ) -> Result<(), Self::Error>;

    /// Finalize the output table before making it the input
    ///
    /// This is useful when the data layout calls for some expensive operation
    /// (e.g. duplication, zeroing of elements, submission of GPU commands...),
    /// which we don't want to perform on every operation but only when the
    /// final output is going to be read.
    fn finalize(&mut self, context: &mut Self::Context) -> Result<(), Self::Error> {
        Ok(())
    }

    /// Reinterpretation of the matrix as a 2D ndarray
    type ScalarView<'a>: Borrow<ArrayView2<'a, Precision>>
    where
        Self: 'a;

    /// View the matrix as a 2D ndarray
    ///
    /// This operation may require reshuffling data, do not use it frequently.
    fn make_scalar_view(
        &mut self,
        context: &mut Self::Context,
    ) -> Result<Self::ScalarView<'_>, Self::Error>;
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
}
