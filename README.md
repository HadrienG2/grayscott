# Performance with stencil (Rust version)

This is a Rust version of the examples from the ["Performance with Stencil"
course](https://lappweb.in2p3.fr/~paubert/PERFORMANCE_WITH_STENCIL/index.html),
with a few new tricks of mine.

## Prerequisites

In addition to [a recent Rust toolchain](https://www.rust-lang.org/learn/get-started),
you will need to install development packages for the following C/++ libraries:

- [hdf5](https://github.com/aldanor/hdf5-rust#compatibility)
- [hwloc](https://github.com/Ichbinjoe/hwloc2-rs#prerequisites)

Additinally, GPU examples use the Vulkan API through the
[vulkano](https://docs.rs/vulkano) library, which comes with [extra build
requirements](https://github.com/vulkano-rs/vulkano#setup-and-troubleshooting).

In addition to the Vulkano build requirements, actually running the GPU examples
requires at least one working Vulkan implementation (e.g. any reasonably modern
Linux GPU driver). Debug builds additionally enable Vulkan validation layers
for richer debug logs, so these must be installed too.

The microbenchmarks are implemented using `criterion`, and we use the newer
`cargo-criterion` runner mechanism, which requires a separate binary that you
can install using this command:

```
$ cargo install cargo-criterion
```

## Structure

In the same spirit as the C++ version, the code is sliced into several crates:

- `data` defines the general data model, parameters and HDF5 file I/O.
- `compute/xyz` crates implement the various compute backends, based on a small
  abstraction layer defined in `compute/shared`. Here are the compute backends
  in suggested learning order:
    * The `naive` backend follows the original naive algorithm, but makes
      idiomatic use of the NumPy-like `ndarray` multidimensional array library
      for the sake of readability.
    * The `regular` backend leverages the fact that the computation is simpler
      at the center of the domain than it is at the edges in order to get about
      2x more performance on the center pixels, at the cost of some code
      duplication between the center and edge computations.
    * The `autovec` backend shapes the computation and data in such a way that
      the compiler can automatically vectorize most of the code. The code is
      simpler and more portable than if it were written directly against harware
      intrinsics, but this implementation strategy also puts us at the mercy of
      compiler autovectorizer whims. Data layout is also improved, pretty much
      like what was done in the `_intrinsics` C++ version.
    * The `manualvec` backend does the vectorization manually instead, like the
      `_intrinsics` C++ version does under the hood. It is significantly more
      complex and less portable than `autovec` while having comparable runtime
      performance, which shows that for this particular problem
      autovectorization can actually be a better tradeoff.
        * Due to Rust's [orphan rules](https://github.com/Ixrec/rust-orphan-rules),
          a significant share of the SIMD abstraction layer that is needed by
          the shared `Species` concentration storage code is implemented in the
          `data` crate instead, see `data/src/concentration/simd/safe_arch.rs`.
        * Since this backend shows that manual vectorization is not worthwhile
          for this problem, the following backends in this list go back to
          autovectorization for simplicity.
    * The `block` backend demonstrates how to use a blocked iteration technique
      to improve CPU cache locality, as the `_link_block` C++ version does.
    * The `parallel` backend implements multi-threaded iteration using
      [rayon](https://docs.rs/rayon), via a fork/join recursive splitting
      technique.
    * The `gpu_xyz` backends implement GPU-based computations using the Vulkan
      API.
        * The `naive` backend starts simple with image-based concentrations
          and a straightforward algorithm.
        * The `specialized` backend uses specialization constants in order to...
          1. Reduce dangerous information duplication between GPU and CPU code
          2. Make the GPU work-group size tunable via CLI or environment
          3. Let the shader compiler know about simulation parameters at
             compile time (this allows for more optimized shader code, though
             here the simulation is so memory-bound it doesn't matter).
        * TODO: Add more backends here as they are implemented.
- The `compute/selector` crate provides a way for compute binaries to
  selectively enable compute backends and pick the most powerful backend
  amongst those that are currently enabled.
- The `ui` crate lets the various binaries listed below share code and
  command-line options where appropriate.
- `simulate` is a binary that runs the simulation. It uses the same CLI argument
  syntax as the `xyz_gray_scott` binaries from the C++ version, but the
  choice of compute backend is made through Cargo features. For each
  `compute/xyz` backend, there is a matching `compute_xyz` feature.
- `livesim` is a variation of `simulate` that displays each image to a live
  window instead of writing images to files, and runs indefinitely. It is
  designed to compute as many simulation steps per second as possible while
  keeping the animation smooth, and should thus provide a nice visual overview
  of how fast backends are.
- `data-to-pics` is a binary that converts HDF5 output datafiles from `simulate`
  into PNG images, much like the `gray_scott2pic` binary from the C++ version
  except it uses a different color palette.

## Usage

To run the simulation, build and run the `simulate` program as follows...

```
$ cargo run --release --bin simulate --features <backend> -- <CLI args>
```

...where `<backend>` is the name of a compute backend, such as "compute_block",
and `<CLI args>` accepts the same arguments as the C++ version. You can put
a `--help` in there for self-documentation.

Then, to convert the HDF5 output into PNG images for visualization purposes, you
can use the `data-to-pics` program, using something like the following...

```
$ mkdir -p pics
$ cargo run --release --bin data-to-pics -- -i <input> -o pics
```

...where `<input>` is the name of the input HDF5 file produced by `simulate`
(`output.h5` by default).

Alternatively, you can run a live version of the simulation which produces a
visual render similar to the aforementioned PNG images in real time, using the
following command:

```
$ cargo run --release --bin livesim --features <backend> -- <CLI args>
```

---

To run all the microbenchmarks, you can use this command:

```
$ cargo criterion
```

Alternatively, you can run microbenchmarks for a specific compute backend `xyz`,
which can speed up compilation by avoiding compilation of unused backends:

```
$ cargo criterion --package xyz
```

You can also selectively run benchmarks based on a regular expression, like so:

```
$ cargo criterion -- '(parallel|gpu).*2048x.*32'
```

The microbenchmark runner exports a more detailed HTML report in
`target/criterion/reports/index.html` that you may want to have a look at.

---

By default, the Rust compiler produces binaries that are compatible with all
CPUs implementing the target architecture. For `x86_64`, which you are most
likely using, that means using no vector instruction set newer than SSE2, making
you miss out on vector processing opportunities.

To unleash your CPU's full number-crunching power, you may want to tell the
compiler to instead produce nonportable binaries that are only guaranteed to
work on your CPU. You can do this by setting the following environment variable
before running the above cargo commands.

```
$ export RUSTFLAGS='-C target-cpu=native'
```
