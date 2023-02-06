# Performance with stencil (Rust version)

This is a Rust version of the examples from the ["Performance with Stencil"
course](https://lappweb.in2p3.fr/~paubert/PERFORMANCE_WITH_STENCIL/index.html),
with a few new tricks of mine.

## Prerequisites

In addition to [a recent Rust toolchain](https://www.rust-lang.org/learn/get-started),
you will need to install development packages for the following C/++ libraries:

- [hdf5](https://github.com/aldanor/hdf5-rust#compatibility)
- hwloc

The microbenchmarks are implemented using `criterion`, and we use the newer
`cargo-criterion` runner mechanism, which requires a separate binary that you
can install using this command:

```
$ cargo install cargo-criterion
```

## Structure

In the same spirit as the C++ version, the code is sliced into several crates:

- `data` defines the general data model, parameters and HDF5 file I/O.
- `compute/xyz` crates implement the various compute backends, except for
  `compute/bench` which provides shared benchmarking utilities. Here are the
  backends in suggested learning order:
    * The `naive` backend follows the original naive algorithm but makes
      idiomatic use of the `ndarray` multidimensional array library.
    * The `regular` backend leverages the fact that the computation is simpler
      at the center of the domain than it is at the edges in order to get about
      2x more performance on the center pixels.
    * The `autovec` backend shapes the computation and data in such a way that
      the compiler can automatically vectorize most of the code. The code is
      simpler and more portable than if it were written directly against harware
      intrinsics, but this implementation strategy also puts us at the mercy of
      compiler autovectorizer whims. Data layout is also improved as in the
      `_intrinsics` C++ version.
    * The `manualvec` backend does the vectorization manually instead, like the
      `_intrinsics` C++ version. It is significantly more complex and less
      portable while having comparable runtime performance, which shows that for
      this particular problem autovectorization provides a better tradeoff.
        * Due to Rust's [orphan rules](https://github.com/Ixrec/rust-orphan-rules),
          a significant share of the SIMD abstraction layer that is needed by
          the shared `Species` concentration storage code is implemented in the
          `data` crate instead, see `data/src/concentration/simd/safe_arch.rs`.
        * Since this backend shows that manual vectorization is not worthwhile
          for this problem, the following backends go back to autovectorization.
    * The `block` backend uses a blocked iteration technique to improve the CPU
      cache hit rate, as the `_link_block` C++ version does.
    * TODO: Add more backends here as they are implemented.
- `reaction` is a binary that runs the simulation. It uses the same CLI argument
  syntax as the `xyz_gray_scott` binaries from the C++ version, but the
  choice of compute backend is made through Cargo features. For each
  `compute/xyz` backend, there is a matching `xyz` feature.
- `data-to-pics` is a binary that converts HDF5 output datafiles from `reaction`
  into PNG images, much like the `gray_scott2pic` binary from the C++ version
  except it uses a different color palette.

## Usage

To run the simulation, build and run the `reaction` program as follows...

```
$ cargo run --release --bin reaction --features <backend> -- <CLI args>
```

...where `<backend>` is the name of a compute backend and `<CLI args>` accepts
the same arguments as the C++ version. You can put a `--help` in there for
self-documentation.

Then, to convert the HDF5 output into PNG images for visualization purposes, you
can use the `data-to-pics` program, using something like the following...

```
$ mkdir pics
$ cargo run --release --bin data-to-pics -- -i <input> -o pics
```

...where `<input>` is the name of the input HDF5 file produced by `<reaction>`
(`output.h5` by default).

---

To run all the microbenchmarks, you can use this command:

```
$ cargo criterion --workspace
```

Alternatively, you can run microbenchmarks for a specific compute backend `xyz`:

```
$ cargo criterion --package xyz
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
