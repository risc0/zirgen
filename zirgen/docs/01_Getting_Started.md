# Getting Started

We don't currently release Zirgen in any packaged form, so it's only available
through this repository. This guide assumes you're using Bazel as your build
system, and want to use Zirgen "out of tree" from a separate project.


# Project structure

## Pulling in the Zirgen compiler

From a fresh project directory, we're going to need to create three build
configuration files. First, create a `.bazelrc` file, and put the following two
lines in it. These are necessary to ensure MLIR, one of Zirgen's dependencies,
compiles correctly.
```
build --cxxopt=-std=c++17
build --host_cxxopt=-std=c++17
```

Second, we need to pin our Bazel version. Zirgen is currently built with Bazel
6.0, so we recommend creating a `.bazelversion` file with the following content.
If you want to use a different version of Bazel for any reason, your mileage may
vary.
```
6.0.0
```

Third, create a `WORKSPACE` file. This is a Bazel configuration file that deals
with "global" configurations like project dependencies, and this is where we're
going to define how to pull in the Zirgen compiler:
```
workspace(name = "zirgen-oot")

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

git_repository(
    name = "zirgen",
    branch = "main", # feel free to pin a particular commit instead for stability!
    remote = "https://github.com/risc0/zirgen.git",
)

load("@zirgen//bazel/rules/zirgen:deps.bzl", "zirgen_dependencies")
zirgen_dependencies()

# configure transitive dependencies
load("@bazel_skylib//:workspace.bzl", "bazel_skylib_workspace")
bazel_skylib_workspace()

load("@llvm-raw//utils/bazel:configure.bzl", "llvm_configure")
llvm_configure(name = "llvm-project")
```

At this point, it should be possible to build the Zirgen compiler, which might
take a few minutes but only needs to be done once:
```
bazel build @zirgen//zirgen/dsl:zirgen
```

## Setting up your new circuit

Now that the compiler is set up, we just need to start the new circuit. Create a
new directory called `circuit`; the name is not important, but consistency is
key! In this directory, create a new file called `circuit.zir`, and include the
following code:
```
test Hello {
    Log("Hello world!");
}
```

Next, add a `BUILD.bazel` file in the same directory, and add the following to
it to configure a new build rule that generates Rust code for the circuit.
```
load("@zirgen//bazel/rules/zirgen:dsl-defs.bzl", "zirgen_genfiles")

filegroup(
    name = "imports",
    srcs = glob(["*.zir"]),
)

zirgen_genfiles(
    name = "CircuitIncs",
    zir_file = ":circuit.zir",
    data = [":imports"],
    zirgen_outs = [
        (
            ["--emit=rust"],
            "circuit.rs.inc",
        ),
    ],
)
```

## Hello world!

Now with everything set up, we're ready to run the new circuit. The output of
the Zirgen compiler is generated code that can then be called by 0STARK to
generate real proofs. Our circuit is so trivial that there's not really much to
generate, but it can be done with the following Bazel command, which will place
the code in the Bazel build directory:
```
bazel build //circuit:GenerateCircuitIncs
```

Of more immediate interest, it's also possible to test and debug circuits using
a built-in interpreter. This doesn't generate real proofs, but it does make it
easy to try things out as you go along. To run our circuit in the interpreter:
```
bazel run @zirgen//zirgen/dsl:zirgen -- $(pwd)/circuit/circuit.zir --test
```
> ```
> Running Hello
> [0] Log: Hello world!
> Lookups resolved
> Verifying constraints for Hello
> Verifying zll constraints for Hello
> ```

This command invokes the Zirgen compiler through Bazel. Everything after `--` is
passed directly to the `zirgen` executable as a command line option: the first
argument indicates the "main" file of the circuit. The `--test` indicates that
all the tests in the source code should be run in the interpreter. In this case,
our code has one test named `Hello`, and it logs the string "Hello world!" on
cycle zero. And presto! We've set up, written, and run a brand-new circuit!

[Prev](README.md)
[Next](02_Conceptual_Overview.md)
