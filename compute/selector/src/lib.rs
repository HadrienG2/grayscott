//! Pick the best compute backend allowed by enabled crate features, expose it
//! as a Simulation typedef.

cfg_if::cfg_if! {
    // TODO: Add more advanced and preferrable implementations above
    if #[cfg(feature = "compute_gpu_specialized")] {
        pub type Simulation = compute_gpu_specialized::Simulation;
    } else if #[cfg(feature = "compute_gpu_naive")] {
        pub type Simulation = compute_gpu_naive::Simulation;
    } else if #[cfg(feature = "compute_parallel")] {
        pub type Simulation = compute_parallel::Simulation;
    } else if #[cfg(feature = "compute_block")] {
        pub type Simulation = compute_block::Simulation;
    } else if #[cfg(feature = "compute_autovec")] {
        pub type Simulation = compute_autovec::Simulation;
    } else if #[cfg(feature = "compute_manualvec")] {
        pub type Simulation = compute_manualvec::Simulation;
    } else if #[cfg(feature = "compute_regular")] {
        pub type Simulation = compute_regular::Simulation;
    } else if #[cfg(any(feature = "compute_naive", test))] {
        pub type Simulation = compute_naive::Simulation;
    } else {
        // If no backend was specified, use a backend skeleton that throws a
        // minimal number of compiler errors.
        use compute::{NoArgs, Simulate, SimulateBase, SimulateCreate};
        use data::{concentration::{Species, ScalarConcentration}, parameters::Parameters};
        use std::convert::Infallible;
        //
        pub struct Simulation;
        //
        impl SimulateBase for Simulation {
            type CliArgs = NoArgs;

            type Concentration = ScalarConcentration;

            type Error = Infallible;

            fn make_species(&self, shape: [usize; 2]) -> Result<Species<ScalarConcentration>, Infallible> {
                Species::new((), shape)
            }
        }
        //
        impl SimulateCreate for Simulation {
            fn new(_params: Parameters, _args: NoArgs) -> Result<Self, Infallible> {
                std::compile_error!("Please enable at least one compute backend via crate features")
            }
        }
        //
        impl Simulate for Simulation {
            fn perform_steps(
                &self,
                _species: &mut Species<ScalarConcentration>,
                _steps: usize
            ) -> Result<(), Infallible> {
                Ok(())
            }
        }
    }
}
