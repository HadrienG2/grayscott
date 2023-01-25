//! Concentration of chemical species

use crate::Precision;
use ndarray::{s, Array2, Ix2, ShapeBuilder};

/// Concentration of all species involved
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub struct Species {
    /// Concentration of species U
    pub u: Evolving,

    /// Concentration of species V
    pub v: Evolving,
}
//
impl Species {
    /// Set up species concentration storage as the C++ version does
    ///
    /// `shape` specifies the concentration matrix dimensions, e.g. [1080, 1920]
    ///
    pub fn new(shape: impl Clone + ShapeBuilder<Dim = Ix2>) -> Self {
        // Start with U = 1.0 and V = 0.0 everywhere
        let mut u = Evolving::ones_out(shape.clone());
        let mut v = Evolving::zeros_out(shape.clone());
        let out_u = u.out();
        let out_v = v.out();

        // Add V and remove U at the center of images
        let num_begin = 7;
        let num_end = 8;
        let frac = 16;
        let row_shift = 4;
        let start_row = out_u.nrows() * num_begin / frac - row_shift;
        let end_row = out_u.nrows() * num_end / frac - row_shift;
        let start_col = out_u.ncols() * num_begin / frac;
        let end_col = out_u.ncols() * num_end / frac;
        out_u
            .slice_mut(s![start_row..end_row, start_col..end_col])
            .fill(0.0);
        out_v
            .slice_mut(s![start_row..end_row, start_col..end_col])
            .fill(1.0);

        // Make the newly generated concentrations become the input ones
        let mut result = Self { u, v };
        result.flip();
        result
    }

    /// Check out the shape of the concentration matrices
    pub fn shape(&self) -> [usize; 2] {
        self.u.shape()
    }

    /// Make the output concentrations become the input ones
    pub fn flip(&mut self) {
        self.u.flip();
        self.v.flip();
    }
}

/// Pair of Concentration where one acts as an input and the other as an output
#[derive(Clone, Debug, PartialEq)]
pub struct Evolving([Concentration; 2]);
//
impl Evolving {
    /// Access the input concentration
    pub fn input(&self) -> &Concentration {
        &self.0[0]
    }

    /// Access the input and output concentration
    pub fn inout(&mut self) -> (&Concentration, &mut Concentration) {
        let [input, output] = &mut self.0;
        (input, output)
    }

    /// Set up concentration storage with all-zeros output concentration
    fn zeros_out(shape: impl Clone + ShapeBuilder<Dim = Ix2>) -> Self {
        Self([
            Concentration::default(shape.clone()),
            Concentration::zeros(shape),
        ])
    }

    /// Set up concentration storage with all-ones output concentration
    fn ones_out(shape: impl Clone + ShapeBuilder<Dim = Ix2>) -> Self {
        Self([
            Concentration::default(shape.clone()),
            Concentration::ones(shape),
        ])
    }

    /// Check the shape of concentration matrices
    fn shape(&self) -> [usize; 2] {
        let [rows, cols] = self.0[0].shape() else { panic!("Expected 2D shape") };
        [*rows, *cols]
    }

    /// Access the output concentration
    fn out(&mut self) -> &mut Concentration {
        &mut self.0[1]
    }

    /// Make the output concentration become the input one
    fn flip(&mut self) {
        let [input, output] = &mut self.0;
        std::mem::swap(input, output);
    }
}

/// Concentration of a chemical species
pub type Concentration = Array2<Precision>;
