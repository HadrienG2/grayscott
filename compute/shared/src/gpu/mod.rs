//! Common facilities shared by all GPU compute backends

#![allow(clippy::result_large_err)]

pub mod context;

use self::context::{config::VulkanConfig, VulkanContext};
use crate::{SimulateBase, SimulateCreate};
use data::{concentration::Species, parameters::Parameters};
#[allow(unused_imports)]
use log::{debug, error, info, log, trace, warn};
use vulkano::sync::{future::NowFuture, FlushError, GpuFuture};

/// Lower-level, asynchronous interface to a GPU compute backend
///
/// GPU programming is, by nature, asynchronous. After the CPU has submitted
/// work to the GPU, it can move on to other things, and only wait for the GPU
/// when it needs actual results from it. This interface lets you leverage this
/// property for performance by exposing the asynchronism in the API.
///
/// Furthermore, creating multiple Vulkan contexts is expensive, and there is no
/// easy way to communicate between them, so for visualization purposes, we will
/// want to use a single context for both compute and visualization. This
/// requires a little more API surface, which is exposed by this interface.
///
/// If you implement this, then SimulateCreate will be implemented for free and
/// the provided `perform_steps_impl()` method can be used to implement
/// `Simulate`. We can't provide a blanket `Simulate` impl for both
/// `SimulateStep` and `SimulateGpu`, and since `SimulateStep` is more newbie
/// focused it took priority for usage simplicity.
pub trait SimulateGpu: SimulateBase
where
    <Self as SimulateBase>::Error: From<FlushError>,
{
    /// Variant of SimulateCreate::new() that also accepts a preliminary Vulkan
    /// context configuration
    ///
    /// Used by clients who intend to reuse the simulation's Vulkan context for
    /// other purposes, in order to specify their requirements on the Vulkan
    /// context.
    ///
    /// Implementors of SimulateGpu should ensure that their final Vulkan
    /// configuration accepts a subset of the devices accepted by `config`.
    fn with_config(
        params: Parameters,
        args: Self::CliArgs,
        config: VulkanConfig,
    ) -> Result<Self, Self::Error>;

    /// Access the Vulkan context used by the simulation
    fn context(&self) -> &VulkanContext;

    /// Quick access to `vulkano::now()` on our device
    fn now(&self) -> NowFuture {
        vulkano::sync::now(self.context().device.clone())
    }

    /// GpuFuture returned by `prepare_steps`
    type PrepareStepsFuture<After: GpuFuture + 'static>: GpuFuture + 'static;

    /// Prepare to perform `steps` simulation steps
    ///
    /// This is an asynchronous version of `Simulate::perform_steps`: it
    /// schedules for some simulation steps to occur after the work designated
    /// by `after`, but does not submit the work to the GPU.
    ///
    /// It is then up to the caller to schedule any extra GPU work they need,
    /// then synchronize as needed.
    fn prepare_steps<After: GpuFuture>(
        &self,
        after: After,
        species: &mut Species<Self::Concentration>,
        steps: usize,
    ) -> Result<Self::PrepareStepsFuture<After>, Self::Error>;

    /// Use this to implement `Simulate::perform_steps`
    fn perform_steps_impl(
        &self,
        species: &mut Species<Self::Concentration>,
        steps: usize,
    ) -> Result<(), Self::Error> {
        self.prepare_steps(
            vulkano::sync::now(self.context().device.clone()),
            species,
            steps,
        )?
        .then_signal_fence_and_flush()?
        .wait(None)?;
        Ok(())
    }
}
//
impl<T: SimulateGpu> SimulateCreate for T
where
    <T as SimulateBase>::Error: From<FlushError>,
{
    fn new(params: Parameters, args: Self::CliArgs) -> Result<Self, Self::Error> {
        Self::with_config(params, args, VulkanConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Once;

    fn init_logger() {
        static INIT_LOGGER: Once = Once::new();
        INIT_LOGGER.call_once(|| env_logger::init());
    }

    #[test]
    fn setup_vulkan() -> context::ContextBuildResult<()> {
        init_logger();
        VulkanConfig {
            enumerate_portability: true,
            ..VulkanConfig::default()
        }
        .build()?;
        Ok(())
    }
}
