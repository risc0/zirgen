# Zirgen — Developer Guide

Zirgen is a compiler for an embedded DSL (the "eDSL") that produces arithmetic
circuits for the RISC Zero ZK proof system. It lowers C++ eDSL code through a
custom MLIR dialect stack and emits Rust, C++, or CUDA source files suitable
for integration with the prover.

## Build system

The project uses **Bazel** as its primary build system.

```bash
# Build everything
bazel build //...

# Build a specific target
bazel build //zirgen/compiler/edsl:edsl

# Run a single test target
bazel test //zirgen/circuit/fib:fib_test

# Run all tests
bazel test //...
```

Cargo is available for Rust crates under `zirgen/` but Bazel is the canonical
build for the C++/MLIR components. The `bootstrap` Cargo profile is used for
the initial Rust build of recursion verifiers; see `Cargo.toml`.

## Testing

```bash
# Run lit (LLVM integrated tests) for a directory
bazel test //zirgen/compiler/edsl:lit

# Run the full test suite (may take several minutes)
bazel test //...
```

LLVM's `lit` framework drives most compiler tests. Test files live alongside
their sources and typically end in `.zir`, `.mlir`, or `.cpp`.

## Directory layout

```
zirgen/
  compiler/
    edsl/           C++ eDSL layer — Val, Buffer, Module, component.h
    codegen/        Source emitters — Rust, C++, CUDA stream emitters
    layout/         Layout allocation and buffer management
    passes/         MLIR transformation passes (optimization, lowering)
    tools/          Standalone compiler binaries (zirgen, zirgen-r1cs, …)
    zkp/            ZK proof helpers (FRI, DEEP-ALI)
  Dialect/
    Zll/            Core ZK dialect: field elements, buffers, digests, IOPs
    ZStruct/        Structured types: structs, arrays, maps, layouts
    ZHL/            High-level zirgen language dialect
    ZHLT/           High-level type dialect (after type inference)
    IOP/            Interactive Oracle Proof dialect
    BigInt/         Big-integer ops
    R1CS/           R1CS / Circom interop
  circuit/
    rv32im/         RISC-V zkVM circuit (V1–V3)
    recursion/      Recursion / proof-composition circuit
    bigint/         Big-integer accelerator circuits
    keccak/         Keccak accelerator
    fib/            Fibonacci example circuit
    hello_v3/       Minimal hello-world example
```

## Architecture overview

### eDSL layer (`compiler/edsl/`)

User-facing C++ API. Circuits are written as ordinary C++ functions that call
eDSL primitives (`Val`, `Buffer`, `IF`, `NONDET`, `BACK`, …). These primitives
build an MLIR function body via `Module::addFunc`. The eDSL uses a
thread-local `Module` singleton and a `CompContext` singleton for component
construction.

Key types:
- `Val` — a field-element value (wraps `mlir::Value` of `Zll::ValType`)
- `Buffer` — a typed slice of the witness/constraint buffer
- `DigestVal` — a hash digest value
- `ReadIopVal` — an IOP stream handle for Fiat-Shamir transcripts
- `Module` — owns the MLIR context and the function being built
- `Comp<T>` / `CompImpl<T>` — component smart-pointer and CRTP base

### MLIR dialect stack

```
eDSL C++ API
    ↓  (Module::addFunc)
Zll + ZStruct + ZHL MLIR
    ↓  (optimization passes: inlining, CSE, canonicalization)
Lowered Zll MLIR
    ↓  (codegen passes)
Rust / C++ / CUDA source
```

**Zll dialect** (`Dialect/Zll/`): core primitive ops — arithmetic over field
elements and extension fields, buffer load/store, digest hashing, IOP reads,
`NondetOp`, `IfOp`, `BackOp`.

**ZStruct dialect** (`Dialect/ZStruct/`): structured-data ops — `ConstructOp`,
`LookupOp`, `SubscriptOp`, `MapOp`, `ReduceOp`, plus layout IR that maps
named component fields to buffer offsets.

**ZHL / ZHLT dialects** (`Dialect/ZHL/`, `Dialect/ZHLT/`): represent the
high-level zirgen source language before lowering to Zll.

### Codegen layer (`compiler/codegen/`)

After optimization the module is lowered to source code by the
`RustStreamEmitter`, `GpuStreamEmitter`, or `CppStreamEmitter` interfaces.
Language-specific syntax is implemented by `RustLanguageSyntax`,
`CppLanguageSyntax`, and `CudaLanguageSyntax` (all subclass `LanguageSyntax`).

Entry points:
- `emitCode` — emit all stages to the output directory
- `emitCodeZirgenPoly` — emit zirgen polynomial source
- `emitRecursion` — encode and emit a recursion witness

### Multi-stage circuits

Some circuits (e.g. rv32im) are split into execution *stages*. Each stage has
its own optimized MLIR module. `Module::optimize(stageCount)` triggers the
split. `Module::runStage` interprets individual stages during test execution.

## Key dependencies

- LLVM / MLIR (built from source via Bazel)
- RISC Zero risc0 crate (via `risc0/` subdirectory)
- Clang for host compilation of generated circuits
