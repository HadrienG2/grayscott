//! Moving concentration data to and from HDF5 files

use crate::{
    concentration::{AsScalars, Concentration, ScalarConcentration, Species},
    Precision,
};
use hdf5::{Dataset, File};
use std::path::Path;

pub use hdf5::Result;

/// Common configuration for reading and writing to HDF5 files
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Config<'dsname, FileName: AsRef<Path>> {
    /// Name of the HDF5 file to be accessed
    pub file_name: FileName,

    /// Name of the dataset within the file
    pub dataset_name: Option<&'dsname str>,
}
//
impl<'dsname, FileName: AsRef<Path>> Config<'dsname, FileName> {
    fn dataset_name(&self) -> &'dsname str {
        self.dataset_name.unwrap_or("matrix")
    }
}

/// Mechanism to write data into an HDF5 file
pub struct Writer(State);
//
impl Writer {
    /// Create or truncate a file
    ///
    /// The file will be dimensioned to store a certain amount of V species
    /// concentration arrays.
    pub fn create(
        config: Config<'_, impl AsRef<Path>>,
        species: &Species<impl Concentration>,
        num_images: usize,
    ) -> Result<Self> {
        let dataset_name = config.dataset_name();
        let file = File::create(config.file_name)?;
        let [rows, cols] = species.shape();
        let dataset = file
            .new_dataset::<Precision>()
            .chunk([1, rows, cols])
            .shape([num_images, rows, cols])
            .lzf()
            .create(dataset_name)?;
        Ok(Self(State {
            file,
            dataset,
            position: 0,
        }))
    }

    /// Write a new V species concentration to the file
    pub fn write(&mut self, result: impl AsScalars) -> Result<()> {
        self.0
            .dataset
            .write_slice(result.as_scalars(), (self.0.position, .., ..))?;
        self.0.position += 1;
        Ok(())
    }

    /// Flush the file to the underlying storage medium and close it
    ///
    /// This should automatically happen on Drop, but doing it manually allows
    /// you to catch and handle errors, instead of letting them lead to panics.
    pub fn close(self) -> Result<()> {
        self.0.file.close()
    }
}

/// Mechanism to read data back from an HDF5 file
pub struct Reader {
    /// Common HDF5 I/O state
    state: State,

    /// Number of concentration arrays to be read
    num_images: usize,
}
//
impl Reader {
    /// Open an existing file
    pub fn open(config: Config<'_, impl AsRef<Path>>) -> Result<Self> {
        let dataset_name = config.dataset_name();
        let file = File::open(config.file_name)?;
        let dataset = file.dataset(dataset_name)?;
        let num_images = dataset.num_chunks().expect("Dataset should be chunked");
        Ok(Self {
            state: State {
                file,
                dataset,
                position: 0,
            },
            num_images,
        })
    }

    /// Shape of images that will be read out
    pub fn image_shape(&self) -> [usize; 2] {
        let dataset_shape = self.state.dataset.shape();
        assert_eq!(
            dataset_shape.len(),
            3,
            "Dataset should be a stack of 2D images"
        );
        [dataset_shape[1], dataset_shape[2]]
    }

    /// Number of images to be read out
    pub fn num_images(&self) -> usize {
        self.num_images
    }

    /// Read the next V species concentration array, if any
    ///
    /// You can equivalently treat this reader as an iterator of arrays.
    pub fn read(&mut self) -> Option<Result<ScalarConcentration>> {
        (self.state.position < self.num_images).then(|| {
            let result = self
                .state
                .dataset
                .read_slice_2d((self.state.position, .., ..))?;
            self.state.position += 1;
            Ok(result)
        })
    }
}
//
impl Iterator for Reader {
    type Item = Result<ScalarConcentration>;

    fn next(&mut self) -> Option<Self::Item> {
        self.read()
    }
}

/// HDF5 file and dataset handle
struct State {
    /// File handle
    file: File,

    /// Dataset
    dataset: Dataset,

    /// Number of images that were read or written so far
    position: usize,
}
