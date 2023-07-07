//! Vulkan instance

use super::ContextBuildResult;
#[allow(unused_imports)]
use log::{debug, error, info, log, trace, warn};
use std::{ops::Deref, sync::Arc};
use vulkano::{
    instance::{
        debug::{
            DebugUtilsMessageSeverity, DebugUtilsMessageType, DebugUtilsMessenger,
            DebugUtilsMessengerCreateInfo, Message,
        },
        Instance, InstanceCreateInfo, InstanceExtensions,
    },
    VulkanLibrary,
};

/// Select Vulkan instance extensions
#[allow(unused_variables)]
pub fn select_extensions(
    library: &VulkanLibrary,
    mut extensions: InstanceExtensions,
    will_render: bool,
) -> InstanceExtensions {
    if cfg!(feature = "gpu-debug-utils") {
        extensions.ext_debug_utils = true;
    }
    #[cfg(feature = "livesim")]
    {
        if will_render {
            extensions = extensions.union(&vulkano_win::required_extensions(library));
        }
    }
    extensions
}

/// Vulkan instance with debug logging
///
/// Logging will stop once this struct is dropped, even if there are
/// other Arc<Instance> remaining in flight
pub struct DebuggedInstance {
    /// Vulkan instance
    instance: Arc<Instance>,

    /// Messenger that logs instance debug messages
    pub(super) _messenger: Option<DebugUtilsMessenger>,
}
//
impl DebuggedInstance {
    /// Set up a Vulkan instance
    ///
    /// This is the point where layers and instance extensions are enabled. See
    /// [`suggested_layers()`] and [`suggested_instance_extensions()`] for suggested
    /// layers and instance extensions.
    ///
    /// If you set `enumerate_portability` to `true` here, you will be able to use
    /// devices that do not fully conform to the Vulkan specification, like MoltenVK
    /// on macOS and iOS, but in that case your device requirements and preferences
    /// must account for the associated [non-conforming
    /// behavior](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_portability_subset.html).
    pub fn new(
        library: Arc<VulkanLibrary>,
        enabled_layers: Vec<String>,
        enabled_extensions: InstanceExtensions,
        enumerate_portability: bool,
    ) -> ContextBuildResult<DebuggedInstance> {
        // Warn if selected layers don't support some extensions
        let unsupported_extensions = *library.supported_extensions()
            - library
                .supported_extensions_with_layers(enabled_layers.iter().map(String::as_ref))?;
        if unsupported_extensions != InstanceExtensions::empty() {
            debug!(
                "Selected layer(s) {enabled_layers:?} do NOT support extensions {unsupported_extensions:#?}"
            );
        }

        // Configure instance
        let create_info = InstanceCreateInfo {
            enabled_extensions,
            enabled_layers,
            enumerate_portability,
            ..InstanceCreateInfo::application_from_cargo_toml()
        };
        info!("Will now create a Vulkan instance with {create_info:#?}");

        // Create instance, with debug logging if supported
        let result = if enabled_extensions.ext_debug_utils {
            // Configure debug utils messenger
            let debug_messenger_cfg = Self::configure_debug_messenger();
            info!("Setting up debug utils with {debug_messenger_cfg:#?}");

            // Safe because our logger does not call into Vulkan APIs
            unsafe {
                let instance = Instance::with_debug_utils_messengers(
                    library,
                    create_info,
                    Some(debug_messenger_cfg.clone()),
                )?;
                let messenger = DebugUtilsMessenger::new(instance.clone(), debug_messenger_cfg)?;
                Self {
                    instance,
                    _messenger: Some(messenger),
                }
            }
        } else {
            Self {
                instance: Instance::new(library, create_info)?,
                _messenger: None,
            }
        };
        trace!("Vulkan instance supports Vulkan v{}", result.api_version());
        Ok(result)
    }

    /// Configure Vulkan logging via the debug utils messenger
    fn configure_debug_messenger() -> DebugUtilsMessengerCreateInfo {
        type DUMSeverity = DebugUtilsMessageSeverity;
        type DUMType = DebugUtilsMessageType;
        let mut debug_messenger_info = DebugUtilsMessengerCreateInfo {
            message_severity: DUMSeverity::empty(),
            message_type: DUMType::GENERAL,
            ..DebugUtilsMessengerCreateInfo::user_callback(Arc::new(|message: &Message| {
                // SAFETY: This callback must not call into Vulkan APIs
                let level = match message.severity {
                    DUMSeverity::ERROR => log::Level::Error,
                    DUMSeverity::WARNING => log::Level::Warn,
                    DUMSeverity::INFO => log::Level::Debug,
                    DUMSeverity::VERBOSE => log::Level::Trace,
                    _ => log::Level::Info,
                };
                let target = message
                    .layer_prefix
                    .map(|layer| format!("Vulkan {:?} {layer}", message.ty))
                    .unwrap_or(format!("Vulkan {:?}", message.ty));
                log!(target: &target, level, "{}", message.description);
            }))
        };
        if log::STATIC_MAX_LEVEL >= log::Level::Error {
            debug_messenger_info.message_severity |= DUMSeverity::ERROR;
        }
        if log::STATIC_MAX_LEVEL >= log::Level::Warn {
            debug_messenger_info.message_severity |= DUMSeverity::WARNING;
        }
        if log::STATIC_MAX_LEVEL >= log::Level::Debug {
            debug_messenger_info.message_severity |= DUMSeverity::INFO;
        }
        if log::STATIC_MAX_LEVEL >= log::Level::Trace {
            debug_messenger_info.message_severity |= DUMSeverity::VERBOSE;
        }
        if cfg!(debug_assertions) {
            debug_messenger_info.message_type |= DUMType::VALIDATION | DUMType::PERFORMANCE;
        };
        debug_messenger_info
    }
}
//
impl Deref for DebuggedInstance {
    type Target = Arc<Instance>;

    fn deref(&self) -> &Self::Target {
        &self.instance
    }
}
