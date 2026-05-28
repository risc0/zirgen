# Zirgen Developer Guide

## Building

```bash
# Build the compiler binary
bazel build //zirgen/Main:gen_zirgen

# Run RV32IM v2 codegen (writes outputs to bazel-bin/zirgen/circuit/rv32im/v2/dsl/)
bazel build //zirgen/circuit/rv32im/v2/dsl:codegen

# Full CI: build zirgen codegen, copy outputs to risc0, run risc0 tests
./scripts/copy_codegen_to_risc0.sh

# Copy outputs only, skip risc0 tests
./scripts/copy_codegen_to_risc0.sh --copy-only

# Build with CUDA support
./scripts/copy_codegen_to_risc0.sh --cuda

# Low-RAM machines (serializes nvcc jobs)
./scripts/copy_codegen_to_risc0.sh --low-memory
```

## Running Tests

```bash
# Run zirgen DSL tests
bazel test //zirgen/dsl/...

# Run circuit tests
bazel test //zirgen/circuit/...

# Risc0 tests after copying codegen outputs
cd $RISC0_ROOT && cargo xtask bootstrap
cargo test -F prove -p risc0-zkvm --lib -- --skip end_to_end
```

## Pipeline Overview

The compiler entry point is `zirgen/Main/gen_zirgen.cpp`. Given a `.zir` source file, it runs:

```
.zir source
    │
    ▼
1. Parse        zirgen::dsl::Parser        → AST
    │
    ▼
2. Lower        zirgen::dsl::lower()       → ZHL MLIR module
    │
    ▼
3. Typecheck    zirgen::Typing::typeCheck()→ ZHLT MLIR module
    │
    ▼
4. Passes       PassManager                → typed module with CheckFuncOp + StepFuncOps
    │            (accum/globals, typing, generate-check, inline/hoist, CSE)
    ▼
5. emitPoly     Utils.cpp                  → validity outputs
    │            (MakePolynomial, taps → emitCodeZirgenPoly)
    │            Writes: validity.ir, poly_ext.rs, taps.rs, eval_check_*.cu, rust_poly_fp_*.cpp
    ▼
6. More passes  PassManager                → optimized module
    │            (elide structs, expand layout, DCE, CSE, degree check)
    ▼
7. makeStepFuncs                           → step functions module
    │            (LowerStepFuncs, BuffersToArgs, canonicalize, symbol DCE)
    ▼
8. emitTarget   Utils.cpp × 3 targets     → Rust + C++ + CUDA outputs
                 (defs, types, layout, steps for each language)
```

The **validity polynomial** (eval_check) is emitted in step 5; **witness generation** (step functions) and **layout/type/defs** are emitted in step 8.

## Dialect Layer Map

| Dialect | Location | Role |
|---------|----------|------|
| ZHL | `zirgen/Dialect/ZHL/` | User-facing high-level IR; output of `dsl::lower()` |
| ZHLT | `zirgen/Dialect/ZHLT/` | Typed high-level IR; holds `CheckFuncOp` and `StepFuncOp` |
| ZStruct | `zirgen/Dialect/ZStruct/` | Layout and buffer operations |
| Zll | `zirgen/Dialect/Zll/` | Low-level polynomial ops (`GetOp`, `SetOp`, arithmetic) |
| BigInt | `zirgen/Dialect/BigInt/` | Big-integer arithmetic for accelerators |
| IOP | `zirgen/Dialect/IOP/` | Interactive oracle proof operations |

## Key Entry Points for New Contributors

| Goal | File |
|------|------|
| Understand full pipeline | `zirgen/Main/gen_zirgen.cpp` |
| Change pass ordering | `zirgen/Main/Main.cpp` (`addAccumAndGlobalPasses`, `addTypingPasses`) |
| Change validity / poly GPU codegen | `zirgen/compiler/codegen/gen_gpu.cpp` |
| Change step/layout emission | `zirgen/Main/Utils.cpp` (`emitTarget`) |
| Add or rename codegen outputs | `zirgen/compiler/codegen/codegen.cpp` (FileEmitter) |
| Change DSL grammar | `zirgen/dsl/parser.cpp`, `zirgen/dsl/lexer.cpp` |
| Change typing / component instantiation | `zirgen/Conversions/Typing/ComponentManager.h` |
| Tune CUDA register pressure | `zirgen/compiler/codegen/gen_gpu.cpp` (slot pool constants) |
| Modify eval_check kernel shape | `zirgen/compiler/codegen/gpu/eval_check.tmpl.cu` |
| Copy outputs to risc0 | `scripts/copy_codegen_to_risc0.sh` |

## Further Reading

- `AGENTS.md` — Deep reference on the pipeline, GPU codegen, risc0 integration, and performance constraints
- `zirgen/docs/01_Getting_Started.md` — DSL language introduction
- `zirgen/docs/02_Conceptual_Overview.md` — Conceptual DSL overview
- `docs/TESTING_IN_RISC0.md` — Copy flow, bootstrap, ZKR zip limitations
- `docs/GENERATING_CUDA_KERNELS.md` — Running codegen and po2 notes
