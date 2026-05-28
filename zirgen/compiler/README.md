# zirgen/compiler

This directory contains the backend compiler infrastructure: code generation, layout computation, constraint systems, and circuit-specific passes.

## Subdirectories

### codegen/

The code generator that emits Rust, C++, and CUDA source from the typed MLIR module. The main entry points are `emitCode` and `emitCodeZirgenPoly` in `codegen.cpp`. `RustStreamEmitter`, `GpuStreamEmitter`, and `CppStreamEmitter` are abstract interfaces that language-specific implementations (`gen_rust.cpp`, `gen_gpu.cpp`, `gen_cpp.cpp`) satisfy. GPU codegen uses a liveness-based slot-reuse pool to bound CUDA register pressure for the `eval_check` kernel. The `CodegenOptions`/`EmitCodeOptions` structs control per-stage output files and extra passes.

### layout/

Computes and visualizes the witness column layout: which buffers hold which component fields and at which offsets. `viz.cpp` provides a textual layout dump used for debugging. The layout pass assigns physical column indices to `ZStruct` layout types and is a prerequisite for both the step-function and validity-polynomial emission.

### r1cs/

R1CS (Rank-1 Constraint System) export for Circom integration. `zirgen-r1cs.cpp` is a standalone tool that converts the typed MLIR module into an R1CS representation, enabling ZIR-defined circuits to interoperate with Circom-based verifiers.

### passes/

Compiler-level MLIR passes that operate across dialects, including optimizations that do not belong to a single dialect's transform library (e.g., cross-dialect DCE, inlining helpers).

### edsl/

Embedded DSL (C++ API) for constructing circuits programmatically rather than from `.zir` source, used by some test circuits and accelerator definitions.

### stats/

Pass that collects and prints statistics about the compiled circuit (operation counts, constraint counts, etc.) for profiling and debugging.

### zkp/

ZKP-layer helpers — primarily utilities related to the RISC Zero proof protocol that the compiler needs to reference (e.g., protocol-info constants consumed by `codegen.h`).

### picus/

Integration with the Picus tool for determinism checking of circuits.

### tools/

Standalone binaries built on top of the compiler library, including `zirgen-r1cs` for Circom export.
