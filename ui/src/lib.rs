//! This crate collects elements that are shared between the three CLI
//! programs data-to-pics, livesim and simulate.

#[cfg(feature = "simulation")]
use compute::SimulateBase;
#[cfg(any(feature = "simulation", feature = "visualization"))]
use data::Precision;
use std::{
    borrow::Cow,
    path::{Path, PathBuf},
};

/// CLI arguments shared by simulation programs
#[cfg(feature = "simulation")]
#[derive(clap::Args)]
pub struct SharedArgs<Simulation: SimulateBase> {
    /// Rate of the process which converts V into P
    #[arg(short, long)]
    pub killrate: Option<Precision>,

    /// Rate of the process which feeds U and drains U, V and P
    #[arg(short, long)]
    pub feedrate: Option<Precision>,

    /// Number of simulation steps to perform between images
    #[arg(short = 'e', long)]
    pub nbextrastep: Option<usize>,

    /// Number of rows of the images to be created
    #[arg(short = 'r', long, default_value_t = 1080)]
    pub nbrow: u32,

    /// Number of columns of the images to be created
    #[arg(short = 'c', long, default_value_t = 1920)]
    pub nbcol: u32,

    /// Simulated time interval on each simulation step
    #[arg(short = 't', long)]
    pub deltat: Option<Precision>,

    /// Backend-specific CLI arguments
    #[command(flatten)]
    pub backend: Simulation::CliArgs,
}
//
#[cfg(feature = "simulation")]
impl<Simulation: SimulateBase> SharedArgs<Simulation> {
    /// Argument defaults for killrate, feedrate and deltat that clap can't handle
    pub fn kill_feed_deltat(&self) -> [Precision; 3] {
        let default_params = data::parameters::Parameters::default();
        let kill_rate = self.killrate.unwrap_or(default_params.kill_rate);
        let feed_rate = self.feedrate.unwrap_or(default_params.feed_rate);
        let time_step = self.deltat.unwrap_or(default_params.time_step);
        [kill_rate, feed_rate, time_step]
    }

    /// Domain shape
    pub fn domain_shape(&self) -> [usize; 2] {
        [self.nbrow as usize, self.nbcol as usize]
    }
}

/// Default simulation output file name
pub fn simulation_output_path(specified_path: Option<PathBuf>) -> Cow<'static, Path> {
    let default_path: &Path = "output.h5".as_ref();
    specified_path.map_or(default_path.into(), Cow::from)
}

/// Set up logging using syslog
#[cfg(feature = "tui")]
pub fn init_syslog() {
    syslog::init(
        syslog::Facility::default(),
        if cfg!(debug_assertions) {
            log::LevelFilter::Debug
        } else {
            log::LevelFilter::Info
        },
        None,
    )
    .expect("Failed to initialize syslog");
    eprintln!("Since stderr is not usable inside of a TUI, logs will be emitted on syslog...");
}

/// Set up indicatif-based progress reporting
#[cfg(feature = "tui")]
pub fn init_progress_reporting(
    message: impl Into<std::borrow::Cow<'static, str>>,
    num_steps: usize,
) -> indicatif::ProgressBar {
    use indicatif::{ProgressBar, ProgressFinish, ProgressStyle};
    use std::time::Duration;

    let progress = ProgressBar::new(num_steps as u64)
        .with_message(message)
        .with_style(
            ProgressStyle::with_template("{msg} {pos}/{len} {wide_bar} {elapsed}/~{duration}")
                .expect("Failed to parse style"),
        )
        .with_finish(ProgressFinish::AndClear);
    progress.enable_steady_tick(Duration::from_millis(100));
    progress
}

/// Color gradient
#[cfg(feature = "visualization")]
pub const GRADIENT: colorous::Gradient = colorous::INFERNO;

/// Amplitude scale
#[cfg(feature = "visualization")]
pub const MAX_AMPLITUDE: Precision = 0.5;

/// Amplitude rescaling factor so MAX_AMPLITUDE matches the end of GRADIENT
#[cfg(feature = "visualization")]
pub const AMPLITUDE_SCALE: Precision = 1.0 / MAX_AMPLITUDE;
