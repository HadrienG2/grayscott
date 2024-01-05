//! Common facilities shared by all compute backends

#[cfg(feature = "criterion")]
#[doc(hidden)]
pub mod benchmark;
#[cfg(feature = "cpu")]
pub mod cpu;
#[cfg(feature = "gpu")]
pub mod gpu;

use clap::Args;
use data::{
    concentration::{Concentration, Species},
    parameters::Parameters,
};
use std::{error::Error, fmt::Debug};

/// Commonalities between all ways to implement a simulation
pub trait SimulateBase: Sized {
    /// Supplementary CLI arguments allowing fine-tuning of this backend
    ///
    /// To honor the principle of least surprise and make criterion
    /// microbenchmarks work smoothly, any argument you add must have a default
    /// value and should also be configurable through environment variables.
    type CliArgs: Args + Debug;

    /// Concentration type
    type Concentration: Concentration;

    /// Error type used by simulation operations
    type Error: Error + From<<Self::Concentration as Concentration>::Error> + Send + Sync;

    /// Set up a species concentration grid
    fn make_species(&self, shape: [usize; 2]) -> Result<Species<Self::Concentration>, Self::Error>;
}

/// No CLI parameters
#[derive(Args, Copy, Clone, Debug, Default, Eq, Hash, PartialEq)]
pub struct NoArgs;

/// Commonalities between all ways to set up a simulation
pub trait SimulateCreate: SimulateBase {
    /// Set up the simulation
    fn new(params: Parameters, args: Self::CliArgs) -> Result<Self, Self::Error>;
}

/// Simulation compute backend interface expected by the binaries
pub trait Simulate: SimulateBase + SimulateCreate {
    /// Perform `steps` simulation time steps on the specified grid
    ///
    /// At the end of the simulation, the input concentrations of `species` will
    /// contain the final simulation results.
    fn perform_steps(
        &self,
        species: &mut Species<Self::Concentration>,
        steps: usize,
    ) -> Result<(), Self::Error>;
}

/// Macro that generates a complete criterion benchmark harness for you
/// Consider using the higher-level cpu_benchmark and gpu_benchmark macros below
#[doc(hidden)]
#[macro_export]
#[cfg(feature = "criterion")]
macro_rules! criterion_benchmark {
    ($backend:ident, $($workload:ident => $workload_name:expr),*) => {
        $(
            fn $workload(c: &mut $crate::benchmark::criterion::Criterion) {
                $crate::benchmark::criterion_benchmark::<$backend::Simulation>(
                    c,
                    stringify!($backend),
                    $crate::benchmark::$workload,
                    $workload_name,
                )
            }
        )*
        $crate::benchmark::criterion::criterion_group!(
            benches,
            $($workload),*
        );
        $crate::benchmark::criterion::criterion_main!(benches);
    };
}

/// Macro that generates a CPU criterion benchmark harness for you
#[macro_export]
#[cfg(feature = "criterion")]
macro_rules! cpu_benchmark {
    ($backend:ident) => {
        $crate::criterion_benchmark!(
            $backend,
            compute_workload => "compute",
            full_sync_workload => "full"
        );
    };
}

/// Macro that generates a GPU criterion benchmark harness for you
#[macro_export]
#[cfg(feature = "criterion")]
macro_rules! gpu_benchmark {
    ($backend:ident) => {
        $crate::criterion_benchmark!(
            $backend,
            compute_workload => "compute",
            full_sync_workload => "full_sync",
            full_gpu_future_workload => "full_future"
        );
    };
}

/// As long as this RAII guard is alive, denormals are flushed to zero in the
/// current thread (not in other threads, sadly)
///
/// Denormals are an obscure feature of IEEE-754 which slightly improves
/// precision of computations involving on very small numbers. They are rarely
/// useful, but spontaneously appear when manipulating exponentially decaying
/// numbers and cause a tremendous execution slowdown on Intel processors.
/// Therefore, flushing them to zero is often the right thing to do...
///
/// This guard operates on a best-effort basis : if no facility for flushing
/// denormals is available/known on the target CPU, denormals will be left alone.
pub struct DenormalsFlusher {
    /// Truth that the FTZ flag was previously unset, and should thus be reset
    /// once the code that uses DenormalsFlusher finishes executing.
    #[cfg(target_feature = "sse")]
    should_reset: bool,
}
//
impl DenormalsFlusher {
    /// Start flushing denormals
    pub fn new() -> Self {
        // SAFETY: Actually UB, alas there's no UB-free way to do this in Rust
        //         at the time of writing because the Rust MXCSR intrinsics are
        //         deprecated and Rust inline assembly currently has no way to
        //         tell the compiler backend that it's clobbering MXCSR.
        //
        //         The following consequences of this UB have been considered:
        //
        //         - The compiler backend may reset MXCSR to the state that it
        //           expects at any point of program execution between the
        //           construction and DenormalsFlusher, which will result in the
        //           FTZ flag being cleared earlier than desired. While
        //           annoying, this is unlikely and does fit into our
        //           "best-effort flushing" API contract.
        //         - Since the FP contract of all known compilers specifies an
        //           expected FP environment where FTZ is off, there is no known
        //           circumstance where it would be okay for the compiler to
        //           expect this flag to be set, so it's fine if we clear it too
        //           eagerly in the destructor.
        //         - Most worryingly, however, the compiler backend may reorder
        //           floating-point operations across our MXCSR state change,
        //           resulting in denormals being incorrectly flushed or not
        //           flushed close to the construction and destruction of the
        //           DenormalsFlusher. We attempt to guard against this using
        //           compiler fences, under the assumption that realistic
        //           floating-point computations take inputs from memory or
        //           write output to memory and are thus affected by fences. The
        //           cost of fences should not be a problem since users are not
        //           expected to create/destroy DenormalFlushers in a loop.
        #[cfg(target_feature = "sse")]
        unsafe {
            use std::sync::atomic::{compiler_fence, Ordering};
            compiler_fence(Ordering::Release);
            let mut mxcsr_buf: u32 = 0;
            let initial_mxcsr: u32;
            std::arch::asm!(
                "stmxcsr [{mxcsr_buf}]",
                "mov {initial_mxcsr:e}, [{mxcsr_buf}]",
                // 0x8000 is the FTZ flag of SSE's MXCSR register
                "or dword ptr [{mxcsr_buf}], 0x8000",
                "ldmxcsr [{mxcsr_buf}]",
                mxcsr_buf = in(reg) &mut mxcsr_buf,
                initial_mxcsr = out(reg) initial_mxcsr,
                options(nostack)
            );
            let should_reset = initial_mxcsr & 0x8000 == 0;
            compiler_fence(Ordering::Acquire);
            Self { should_reset }
        }
        #[cfg(not(target_feature = "sse"))]
        Self {}
    }
}
//
impl Drop for DenormalsFlusher {
    fn drop(&mut self) {
        // SAFETY: See comment in new()
        #[cfg(target_feature = "sse")]
        unsafe {
            if self.should_reset {
                use std::sync::atomic::{compiler_fence, Ordering};
                compiler_fence(Ordering::Release);
                let mut mxcsr_buf: u32 = 0;
                std::arch::asm!(
                    "stmxcsr [{mxcsr_buf}]",
                    // 0xFFFF7FFF == !0x8000 as u32
                    "and dword ptr [{mxcsr_buf}], 0xFFFF7FFF",
                    "ldmxcsr [{mxcsr_buf}]",
                    mxcsr_buf = in(reg) &mut mxcsr_buf,
                    options(nostack)
                );
                compiler_fence(Ordering::Acquire);
            }
        }
    }
}
//
impl Default for DenormalsFlusher {
    fn default() -> Self {
        Self::new()
    }
}
