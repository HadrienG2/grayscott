//! This crate collects elements that are shared between the three CLI
//! programs data-to-pics, livesim and simulate.

use clap::Args;
use colorous::Gradient;
use compute::SimulateBase;
use data::{parameters::Parameters, Precision};

/// CLI arguments shared by the "livesim" and "simulate" executables
#[derive(Args)]
pub struct SharedArgs<Simulation: SimulateBase> {
    /// Rate of the process which converts V into P
    #[arg(short, long)]
    pub killrate: Option<Precision>,

    /// Rate of the process which feeds U and drains U, V and P
    #[arg(short, long)]
    pub feedrate: Option<Precision>,

    /// Number of simulation steps to perform between images
    #[arg(short = 'e', long, default_value_t = 34)]
    pub nbextrastep: usize,

    /// Number of rows of the images to be created
    #[arg(short = 'r', long, default_value_t = 1080)]
    pub nbrow: usize,

    /// Number of columns of the images to be created
    #[arg(short = 'c', long, default_value_t = 1920)]
    pub nbcol: usize,

    /// Simulated time interval on each simulation step
    #[arg(short = 't', long)]
    pub deltat: Option<Precision>,

    /// Backend-specific CLI arguments
    #[command(flatten)]
    pub backend: Simulation::CliArgs,
}

/// Argument defaults for killrate, feedrate and deltat that clap can't handle
pub fn kill_feed_deltat(args: &SharedArgs<impl SimulateBase>) -> [Precision; 3] {
    let default_params = Parameters::default();
    let kill_rate = args.killrate.unwrap_or(default_params.kill_rate);
    let feed_rate = args.feedrate.unwrap_or(default_params.feed_rate);
    let time_step = args.deltat.unwrap_or(default_params.time_step);
    [kill_rate, feed_rate, time_step]
}

/// Color gradient shared by the "simulate" and "livesim" visualizations
pub const GRADIENT: Gradient = colorous::INFERNO;

/// Amplitude scale shared by the "simulate" and "livesim" visualizations
pub const MAX_AMPLITUDE: Precision = 0.6;

/// Amplitude scale factor associated with MAX_AMPLITUDE
pub const AMPLITUDE_SCALE: Precision = 1.0 / MAX_AMPLITUDE;
