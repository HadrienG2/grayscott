use clap::Parser;
use compute::{Simulate, SimulateBase, SimulateCreate};
use compute_selector::Simulation;
use data::{concentration::AsScalars, parameters::Parameters};
use ui::SharedArgs;

/// Perform Gray-Scott simulation
#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// CLI arguments shared with the "livesim" executable
    #[command(flatten)]
    shared: SharedArgs<Simulation>,
}

fn main() {
    // Enable logging to stderr
    env_logger::init();

    // Parse CLI arguments and handle clap-incompatible defaults
    let args = Args::parse();
    let [kill_rate, feed_rate, time_step] = ui::kill_feed_deltat(&args.shared);

    // Set up the simulation
    // TODO: Once ready to share the GPU context, send in our requirements
    let simulation = Simulation::new(
        Parameters {
            kill_rate,
            feed_rate,
            time_step,
            ..Default::default()
        },
        args.shared.backend,
    )
    .expect("Failed to create simulation");

    // Set up chemical species concentration storage
    let mut species = simulation
        .make_species([args.shared.nbrow, args.shared.nbcol])
        .expect("Failed to set up simulation grid");

    // TODO: Set up event loop
    while false {
        // TODO: Add fast path for GPU backends
        simulation
            .perform_steps(&mut species, args.shared.nbextrastep)
            .expect("Failed to compute simulation steps");
        species
            .make_result_view()
            .expect("Failed to extract result")
            .as_scalars();

        // TODO: Add rendering
    }
}
