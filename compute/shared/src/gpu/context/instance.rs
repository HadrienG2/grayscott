//! Vulkan instance

use super::ContextBuildResult;
#[allow(unused_imports)]
use log::{debug, error, info, log, trace, warn};
use std::{fmt::Write, ops::Deref, sync::Arc};
use vulkano::{
    instance::{
        debug::{
            DebugUtilsMessageSeverity, DebugUtilsMessageType, DebugUtilsMessenger,
            DebugUtilsMessengerCallback, DebugUtilsMessengerCallbackLabel,
            DebugUtilsMessengerCreateInfo,
        },
        Instance, InstanceCreateFlags, InstanceCreateInfo, InstanceExtensions,
    },
    VulkanLibrary,
};
#[cfg(feature = "livesim")]
use winit::window::Window;

/// Select Vulkan instance extensions
#[allow(unused_variables)]
pub fn select_extensions(
    library: &VulkanLibrary,
    mut extensions: InstanceExtensions,
    #[cfg(feature = "livesim")] window: Option<&Window>,
) -> InstanceExtensions {
    if cfg!(feature = "gpu-debug-utils") {
        extensions.ext_debug_utils = true;
    }
    #[cfg(feature = "livesim")]
    {
        if let Some(window) = window {
            extensions |= vulkano::swapchain::Surface::required_extensions(window);
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

        // Set up debug logging, if enabled
        let debug_messenger_cfg = enabled_extensions
            .ext_debug_utils
            .then(Self::configure_debug_messenger);

        // Configure instance
        let mut flags = InstanceCreateFlags::default();
        if enumerate_portability {
            flags |= InstanceCreateFlags::ENUMERATE_PORTABILITY;
        }
        let create_info = InstanceCreateInfo {
            flags,
            enabled_extensions,
            enabled_layers,
            debug_utils_messengers: debug_messenger_cfg.clone().into_iter().collect(),
            ..InstanceCreateInfo::application_from_cargo_toml()
        };
        info!("Will now create a Vulkan instance with {create_info:#?}");
        let instance = Instance::new(library, create_info)?;
        trace!(
            "Vulkan instance supports Vulkan v{}",
            instance.api_version()
        );

        // Set up runtime debug logging
        let _messenger = debug_messenger_cfg
            .map(|cfg| DebugUtilsMessenger::new(instance.clone(), cfg))
            .transpose()?;
        Ok(Self {
            instance,
            _messenger,
        })
    }

    /// Configure Vulkan logging via the debug utils messenger
    fn configure_debug_messenger() -> DebugUtilsMessengerCreateInfo {
        type DUMSeverity = DebugUtilsMessageSeverity;
        type DUMType = DebugUtilsMessageType;
        let mut debug_messenger_info = DebugUtilsMessengerCreateInfo {
            message_severity: DUMSeverity::empty(),
            message_type: DUMType::GENERAL,
            // SAFETY: This callback does not call into Vulkan APIs
            ..DebugUtilsMessengerCreateInfo::user_callback(unsafe {
                DebugUtilsMessengerCallback::new(|severity, ty, data| {
                    let level = match severity {
                        DUMSeverity::ERROR => log::Level::Error,
                        DUMSeverity::WARNING => log::Level::Warn,
                        DUMSeverity::INFO => log::Level::Debug,
                        DUMSeverity::VERBOSE => log::Level::Trace,
                        _ => log::Level::Info,
                    };
                    if level > log::max_level() {
                        return;
                    }
                    let target = data
                        .message_id_name
                        .map(|id_name| format!("Vulkan {ty:?} {id_name}"))
                        .unwrap_or(format!("Vulkan {ty:?}"));
                    fn labels<'iter>(
                        iter: impl Iterator<Item = DebugUtilsMessengerCallbackLabel<'iter>>,
                    ) -> Vec<&'iter str> {
                        iter.map(|label| label.label_name).collect::<Vec<_>>()
                    }
                    let queue_labels = labels(data.queue_labels);
                    let cmd_buf_labels = labels(data.cmd_buf_labels);
                    let objects = data
                        .objects
                        .map(|obj| {
                            let mut desc = format!("{:?} #{}", obj.object_type, obj.object_handle);
                            if let Some(name) = obj.object_name {
                                write!(desc, " named \"{name}\"").expect("can't fail");
                            }
                            desc
                        })
                        .collect::<Vec<_>>();
                    log!(
                        target: &target,
                        level,
                        "{} (id: {}, queue_labels: {:?}, cmd_buf_labels: {:?}, objects: {:?})",
                        data.message,
                        data.message_id_number,
                        queue_labels,
                        cmd_buf_labels,
                        objects,
                    );
                })
            })
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
        info!("Setting up debug utils with {debug_messenger_info:#?}");
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
