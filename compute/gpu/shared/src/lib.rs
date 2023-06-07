use anyhow::anyhow;
#[allow(unused_imports)]
use log::{debug, error, info, log, trace, warn};
use std::{cmp::Ordering, ops::Deref, sync::Arc};
use vulkano::{
    device::physical::PhysicalDevice,
    instance::{
        debug::{
            DebugUtilsMessageSeverity, DebugUtilsMessageType, DebugUtilsMessenger,
            DebugUtilsMessengerCreateInfo, Message,
        },
        Instance, InstanceCreateInfo, InstanceExtensions, Version,
    },
    ExtensionProperties, VulkanLibrary,
};

/// Load the Vulkan library
fn load_library() -> anyhow::Result<Arc<VulkanLibrary>> {
    let library = VulkanLibrary::new()?;
    info!("Loaded Vulkan library:");
    debug!("- Supports Vulkan v{}", library.api_version());
    debug!(
        "- Supports instance extensions {}",
        format_extension_properties(library.extension_properties())
    );
    debug!(
        "- Supports layers {:#?}",
        library
            .layer_properties()?
            .map(|layer| {
                format!(
                    "{} v{} for Vulkan >= {}",
                    layer.name(),
                    layer.implementation_version(),
                    layer.vulkan_version()
                )
            })
            .collect::<Vec<_>>()
    );
    Ok(library)
}

/// Set up a Vulkan instance
fn setup_instance(
    library: Arc<VulkanLibrary>,
    enabled_layers: Vec<String>,
    enabled_extensions: InstanceExtensions,
) -> anyhow::Result<DebuggedInstance> {
    let unsupported_extensions = *library.supported_extensions()
        - library.supported_extensions_with_layers(enabled_layers.iter().map(String::as_ref))?;
    if unsupported_extensions != InstanceExtensions::empty() {
        debug!(
            "Selected layer(s) {enabled_layers:?} do NOT support extensions {unsupported_extensions:#?}"
        );
    }

    let create_info = InstanceCreateInfo {
        enabled_extensions,
        enabled_layers,
        ..InstanceCreateInfo::application_from_cargo_toml()
    };

    type DUMS = DebugUtilsMessageSeverity;
    type DUMT = DebugUtilsMessageType;
    let mut debug_messenger_info = DebugUtilsMessengerCreateInfo {
        message_severity: DUMS::ERROR | DUMS::WARNING,
        message_type: DUMT::GENERAL,
        ..DebugUtilsMessengerCreateInfo::user_callback(Arc::new(|message: &Message| {
            let level = match message.severity {
                DUMS::ERROR => log::Level::Error,
                DUMS::WARNING => log::Level::Warn,
                DUMS::INFO => log::Level::Debug,
                DUMS::VERBOSE => log::Level::Trace,
                _ => log::Level::Info,
            };
            let target = message
                .layer_prefix
                .map(|layer| format!("Vulkan {:?} {layer}", message.ty))
                .unwrap_or(format!("Vulkan {:?}", message.ty));
            log!(target: &target, level, "{}", message.description);
        }))
    };
    if cfg!(debug_assertions) {
        debug_messenger_info.message_severity |= DUMS::INFO | DUMS::VERBOSE;
        debug_messenger_info.message_type |= DUMT::VALIDATION | DUMT::PERFORMANCE;
    };

    info!("Will now create a Vulkan instance with configuration {create_info:#?}");
    let instance = unsafe {
        Instance::with_debug_utils_messengers(
            library,
            create_info,
            std::iter::once(debug_messenger_info.clone()),
        )?
    };
    let _messenger = unsafe { DebugUtilsMessenger::new(instance.clone(), debug_messenger_info)? };
    debug!(
        "Vulkan instance supports Vulkan v{}",
        instance.api_version()
    );

    Ok(DebuggedInstance {
        instance,
        _messenger,
    })
}

/// Vulkan instance with debug logging
struct DebuggedInstance {
    /// Vulkan instance
    instance: Arc<Instance>,

    /// Messenger that logs instance debug messages, should be kept alive as
    /// long as the instance is kept alive
    _messenger: DebugUtilsMessenger,
}
//
impl Deref for DebuggedInstance {
    type Target = Arc<Instance>;

    fn deref(&self) -> &Self::Target {
        &self.instance
    }
}

/// Suggested set of layers to enable
fn suggested_layers() -> Vec<String> {
    if cfg!(debug_assertions) {
        vec!["VK_LAYER_KHRONOS_validation".to_owned()]
    } else {
        vec![]
    }
}

/// Suggested set of instance extensions to enable
fn suggested_instance_extensions() -> InstanceExtensions {
    InstanceExtensions {
        ext_debug_utils: true,
        ..Default::default()
    }
}

/// Pick a physical device to run the simulation on
///
/// As currently written, the simulation code does not support running on
/// multiple devices. This adds quite a bit of complexity due to the need for
/// efficient data transfers and synchronization between devices, and it is
/// often possible to just run independent simulations on each device instead.
///
fn select_device(
    instance: &DebuggedInstance,
    mut can_use: impl FnMut(&Arc<PhysicalDevice>) -> bool,
    preference: impl FnMut(&Arc<PhysicalDevice>, &Arc<PhysicalDevice>) -> Ordering,
) -> anyhow::Result<Arc<PhysicalDevice>> {
    let selected_device = instance
        .enumerate_physical_devices()?
        .inspect(|device| {
            info!("Found physical device {}:", device.properties().device_name);
            debug!("- With {:#?}", device.properties());
            debug!(
                "- With device extensions {}",
                format_extension_properties(device.extension_properties())
            );
            debug!("- With features {:#?}", device.supported_features());
            debug!("- With {:#?}", device.memory_properties());
            debug!(
                "- With queue families {:#?}",
                device.queue_family_properties()
            );
            if device.api_version() >= Version::V1_3
                || device.supported_extensions().ext_tooling_info
            {
                debug!("- With tools {:#?}", device.tool_properties().unwrap());
            }
        })
        .filter(|device| {
            let can_use = can_use(device);
            if can_use {
                info!("=> Device meets requirements");
            } else {
                info!("=> Device does NOT meet requirements");
            }
            can_use
        })
        .max_by(preference);
    if let Some(device) = selected_device {
        info!("Selected device {}", device.properties().device_name);
        Ok(device)
    } else {
        Err(anyhow!("Did not find any device matching requirements"))
    }
}

/// Format Vulkan extension properties for display
fn format_extension_properties(extension_properties: &[ExtensionProperties]) -> String {
    format!(
        "{:#?}",
        extension_properties
            .iter()
            .map(|ext| format!("{} v{}", ext.extension_name, ext.spec_version))
            .collect::<Vec<_>>()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() -> anyhow::Result<()> {
        env_logger::init();

        let library = load_library()?;
        let instance =
            setup_instance(library, suggested_layers(), suggested_instance_extensions())?;

        // TODO: Refine device selection criteria once more code is written
        let device = select_device(&instance, |_| true, |_, _| Ordering::Equal);

        Ok(())
    }
}
