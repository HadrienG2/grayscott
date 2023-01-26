# Performance with stencil (Rust version)

This is a Rust version of the examples from the ["Performance with Stencil"
course](https://lappweb.in2p3.fr/~paubert/PERFORMANCE_WITH_STENCIL/index.html),
with a few new tricks of mine.

## Prerequisites

In addition to [a recent Rust toolchain](https://www.rust-lang.org/learn/get-started),
you will need to have installed the HDF5 library and development packages.
More information [is available here](https://github.com/aldanor/hdf5-rust#compatibility).

The microbenchmarks are implemented using `criterion`, and we use the newer
`cargo-criterion` runner mechanism, which requires a separate binary that you
can install using this command:

```
$ cargo install cargo-criterion
```

## Structure

In the same spirit as the C++ version, the code is sliced into several crates:

- `data` defines the general data model, parameters and HDF5 file I/O.
- `compute-xyz` crates implement the various compute backends, except for
  `compute-bench` which provides shared benchmarking utilities.
    * `compute-naive` replicates the original naive algorithm while making
      idiomatic use of the `ndarray` multidimensional library.
    * TODO: Add more backends here as they are implemented.
- `reaction` is a binary that runs the simulation. It uses the same CLI argument
  syntax as the `xyz_gray_scott` binaries from the C++ version, but the
  choice of compute backend is made through Cargo features. For each
  `compute-xyz` backend, there is a matching `xyz` feature.
- `data-to-pics` is a binary that converts HDF5 output datafiles from `reaction`
  into PNG images, much like the `gray_scott2pic` binary from the C++ version
  except it uses a different color palette.

## Usage

To run the simulation, run this command...

```
$ cargo run --release --bin reaction --features <backend> -- <CLI args>
```

...where `<backend>` is the name of a compute backend and `<CLI args>` accepts
the same arguments as the C++ version. You can put a `--help` in there for
self-documentation.

Then, to convert the HDF5 output into PNG images for visualization purposes, you
can run something like...

```
$ mkdir pics
$ cargo run --release --bin data-to-pics -- -i <input> -o pics
```

...where `<input>` is the name of the input HDF5 file (`output.h5` by default).

---

To run all the microbenchmarks, you can do...

```
$ cargo criterion --workspace
```

Alternatively, you can run microbenchmarks for a specific compute backend `xyz`:

```
$ cargo criterion --package compute-xyz
```

The microbenchmark runner exports a more detailed HTML report in
`target/criterion/reports/index.html` that you may want to have a look at.
