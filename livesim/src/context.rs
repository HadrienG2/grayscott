//! Basic Vulkan context and simulation engine initialization

use crate::{surface, Result};
use compute::gpu::context::{config::VulkanConfig, VulkanContext};
#[cfg(feature = "gpu")]
use compute::gpu::SimulateGpu;
#[cfg(not(feature = "gpu"))]
use compute::SimulateCreate;
use compute_selector::Simulation;
use data::parameters::Parameters;
use log::info;
use std::sync::Arc;
use ui::SharedArgs;
use vulkano::{device::Queue, swapchain::Surface};
use winit::window::Window;

/// Simulation context = Simulation + custom VulkanContext if needed, lets you
/// borrow either the simulation or the VulkanContext
pub struct SimulationContext {
    /// Gray-Scott reaction simulation, which may have its own VulkanContext
    simulation: Simulation,

    /// Custom VulkanContext if `simulation` doesn't have one
    #[cfg(not(feature = "gpu"))]
    context: VulkanContext,
}
//
impl SimulationContext {
    /// Set up the simulation and Vulkan context
    pub fn new(args: &SharedArgs<Simulation>, window: &Arc<Window>) -> Result<Self> {
        // Configure simulation
        let [kill_rate, feed_rate, time_step] = args.kill_feed_deltat();
        let parameters = Parameters {
            kill_rate,
            feed_rate,
            time_step,
            ..Default::default()
        };

        // Create simulation, forwarding our context config if it's Vulkan-based
        let simulation = {
            #[cfg(feature = "gpu")]
            {
                info!("Rendering will share the simulation's Vulkan context");
                Simulation::with_config(
                    parameters,
                    args.backend,
                    Self::vulkan_config(window.clone()),
                )
            }
            #[cfg(not(feature = "gpu"))]
            {
                Simulation::new(parameters, args.backend)
            }
        }?;

        // Create a dedicated context if the simulation is not Vulkan-based
        {
            #[cfg(feature = "gpu")]
            {
                Ok(Self { simulation })
            }
            #[cfg(not(feature = "gpu"))]
            {
                info!("Rendering will use its own Vulkan context");
                let context = Self::vulkan_config(window.clone()).build()?;
                Ok(Self {
                    simulation,
                    context,
                })
            }
        }
    }

    /// Gray-Scott simulation
    pub fn simulation(&self) -> &Simulation {
        &self.simulation
    }

    /// Vulkan context
    pub fn vulkan(&self) -> &VulkanContext {
        #[cfg(feature = "gpu")]
        {
            self.simulation.context()
        }
        #[cfg(not(feature = "gpu"))]
        {
            &self.context
        }
    }

    /// Rendering surface
    pub fn surface(&self) -> &Arc<Surface> {
        self.vulkan()
            .surface
            .as_ref()
            .expect("There should be one (window specified in VulkanConfig)")
    }

    /// Rendering queue (only queue currently used)
    pub fn queue(&self) -> &Arc<Queue> {
        &self.vulkan().queues[0]
    }

    /// Vulkan context configuration that we need
    fn vulkan_config(window: Arc<Window>) -> VulkanConfig {
        let default_config = VulkanConfig::default();
        VulkanConfig {
            window_and_reqs: Some((window, Box::new(surface::requirements))),
            // TODO: Flesh out as code is added
            ..default_config
        }
    }
}
