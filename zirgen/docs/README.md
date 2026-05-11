# The Zirgen Circuit Language

## Introduction

The Zirgen circuit language is a domain-specific language for writing arithmetic
circuits for the RISC Zero proof system. What kinds of things can you build
with Zirgen?
* zk accelerators (hashing, bigint operations, cryptographic primitives)
* zkVMs
* recursion circuits
* arbitrary zkApps (though we recommend doing this in Rust instead!)

## User Guide Table of Contents

1. [Getting Started](01_Getting_Started.md)
2. [Basic Concepts](02_Conceptual_Overview.md)
3. [Building a Fibonacci Circuit](03_Building_a_Fibonacci_Circuit.md)
4. [Components](04_Components.md)
5. [Muxes](05_Muxes.md)
6. [Arrays and Loops](99_Arrays_and_Loops.md)
7. [Backs](99_Backs.md)
8. [Externs](99_Externs.md)
9. [Built-in Components](A1_Builtin_Components.md)

## Developer Documentation

For compiler development and contributing to Zirgen, see:

* [Architecture](../../ARCHITECTURE.md) - Compiler structure, MLIR dialects, and codegen pipeline
* [Bazel Build Guide](../../docs/BAZEL_BUILD_GUIDE.md) - Build system and common patterns
* [Scripts Reference](../../docs/SCRIPTS_REFERENCE.md) - Development scripts documentation
* [Code Navigation](../../docs/CODE_NAVIGATION.md) - Finding your way around the codebase
* [Contributing](../../CONTRIBUTING.md) - Contribution guidelines and commit format

[Next](01_Getting_Started.md)
