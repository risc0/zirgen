# Zirgen Architecture

This document provides an architectural overview of the Zirgen circuit compiler, covering the compiler structure, MLIR dialects, codegen pipeline, and development workflow.

## Table of Contents

1. [Overview](#overview)
2. [Directory Structure](#directory-structure)
3. [MLIR Dialects](#mlir-dialects)
4. [Compiler Pipeline](#compiler-pipeline)
5. [Code Generation](#code-generation)
6. [Key Technologies](#key-technologies)

## Overview

Zirgen is a circuit compiler for the RISC Zero proof system. It compiles domain-specific language (`.zir` files) into arithmetic circuits, producing:
- **Witness layout** - Memory layout for execution trace
- **Constraint polynomials** (validity) - Arithmetic constraints
- **Witness-generation code** (steps) - Code to execute circuits

The compiler generates code in multiple target languages:
- **Rust** - Circuit metadata and polynomial constraints
- **C++** - Step functions and layout definitions
- **CUDA** - GPU-accelerated validity checking kernels
- **Metal** - GPU kernels for Apple Silicon (optional)

**Important**: RISC Zero does not run Zirgen directly. Instead, Zirgen-generated files are vendored into the RISC Zero repository. The workflow is:
1. Change circuit or codegen in Zirgen
2. Run codegen to generate output files
3. Copy outputs into RISC Zero repository
4. Build and test RISC Zero with new circuit code

## Directory Structure

```
zirgen/
├── zirgen/                      # Main compiler source
│   ├── Dialect/                 # MLIR dialect definitions
│   │   ├── Zll/                 # Low-level IR
│   │   ├── ZStruct/             # Layout and structs
│   │   ├── ZHLT/                # High-level typed IR
│   │   ├── ZHL/                 # High-level operations
│   │   ├── BigInt/              # Arbitrary precision integers
│   │   ├── R1CS/                # Rank-1 constraint system
│   │   └── IOP/                 # Interactive oracle proofs
│   ├── Conversions/             # Dialect lowering passes
│   ├── Main/                    # Entry points and pipeline
│   ├── dsl/                     # DSL parser and lowering
│   ├── compiler/                # Code generation
│   │   ├── codegen/             # Target code generators
│   │   │   ├── gen_rust.cpp     # Rust codegen
│   │   │   ├── gen_cpp.cpp      # C++ codegen
│   │   │   ├── gen_gpu.cpp      # CUDA/Metal codegen
│   │   │   └── gpu/             # GPU kernel templates
│   │   ├── zkp/                 # ZKP primitives (hashing, fields)
│   │   ├── r1cs/                # Circom R1CS integration
│   │   ├── layout/              # Memory layout computation
│   │   └── stats/               # Circuit statistics
│   ├── circuit/                 # Circuit implementations
│   │   ├── rv32im/              # RISC-V zkVM circuit (main)
│   │   │   ├── v1/              # Version 1 (EDSL-based)
│   │   │   └── v2/              # Version 2 (DSL-based, current)
│   │   ├── recursion/           # Recursion/composition circuit
│   │   ├── keccak/              # Keccak/SHA3 circuit
│   │   ├── bigint/              # BigInt operations
│   │   ├── fib/                 # Fibonacci example
│   │   └── verify/              # Verification circuits
│   ├── components/              # Reusable ZIR components
│   └── docs/                    # User-facing documentation
├── bazel/                       # Build system configuration
│   ├── rules/                   # Custom Bazel rules
│   │   ├── zirgen/              # Circuit compilation rules
│   │   ├── lit/                 # LLVM Integrated Tester
│   │   └── clang_format/        # Code formatting
│   ├── platform/                # Platform detection
│   └── toolchain/               # Cross-compilation toolchains
├── scripts/                     # Development scripts
├── hooks/                       # Git hooks
├── docs/                        # Technical documentation
└── risc0/                       # Core libraries (fp, core)
```

## MLIR Dialects

Zirgen implements seven custom MLIR dialects for circuit compilation:

### 1. Zll (Zirgen Low-Level)
**Location**: `zirgen/Dialect/Zll/`

Low-level intermediate representation with operations like:
- `GetOp`, `SetOp` - Read/write buffers
- Polynomial operations
- Field arithmetic operations

**Default field**: BabyBear prime field

### 2. ZStruct
**Location**: `zirgen/Dialect/ZStruct/`

Manages layout, buffers, and struct representations. Depends on:
- Affine dialect
- Func dialect
- Zll dialect

### 3. ZHLT (Zirgen High-Level Typed)
**Location**: `zirgen/Dialect/ZHLT/`

High-level typed IR with two main operation types:
- `CheckFuncOp` - Validity constraint checking
- `StepFuncOp` - Witness generation

Depends on Func and Zll dialects.

### 4. ZHL (Zirgen High-Level)
**Location**: `zirgen/Dialect/ZHL/`

High-level operations for circuit description. Depends on Func and Zll dialects.

### 5. BigInt
**Location**: `zirgen/Dialect/BigInt/`

Arbitrary precision integer operations with R1CS constraint support. Used for cryptographic operations like RSA.

### 6. R1CS
**Location**: `zirgen/Dialect/R1CS/`

Rank-1 Constraint System dialect for Circom integration. Enables using Circom circuits within Zirgen.

### 7. IOP (Interactive Oracle Proof)
**Location**: `zirgen/Dialect/IOP/`

Interactive oracle proof primitives. Depends on Zll dialect.

Each dialect has:
- `.td` (TableGen) definitions for operations and types
- IR implementation
- Transform and analysis passes
- Lowering passes to other dialects

## Compiler Pipeline

**Entry Point**: `zirgen/Main/gen_zirgen.cpp`

The compilation pipeline consists of these stages:

### 1. Parse
**Source**: `zirgen/dsl/Parser`

Reads `.zir` source files (with `-I` include directories) and constructs an Abstract Syntax Tree (AST).

### 2. Lower to MLIR
**Source**: `zirgen/dsl/lower.cpp`

Lowers the AST to MLIR dialects:
- Zll (low-level operations)
- ZStruct (layout and buffers)
- ZHLT (high-level typed operations)

### 3. Typecheck
**Source**: `zirgen/Typing::typeCheck()`

Performs type checking and inference on the MLIR module.

### 4. Transformation Passes

Multiple passes transform and optimize the IR:

**Accumulation and Globals**: Generate accumulator logic and global state
**Typing Passes**: back, layout, exec, buffers, steps
**Generate Check**: Create `CheckFuncOp` operations for validity constraints
**Inline/Hoist**: Optimization passes

Result: Typed MLIR module with `CheckFuncOp` (validity) and `StepFuncOp` (witness generation).

### 5. Emit Polynomials
**Source**: `zirgen/compiler/codegen/codegen.cpp` - `emitCodeZirgenPoly`

Builds polynomial representations from check functions:
- Clones `CheckFuncOp` to `func::FuncOp`
- Runs `MakePolynomial` pass
- Computes taps (trace access patterns)
- Generates polynomial code

**Outputs**:
- `validity.ir` - Polynomial intermediate representation
- `poly_ext.rs` - Rust polynomial extensions
- `taps.rs`, `taps_provider_impl.rs.inc` - Trace access patterns
- `info.rs` - Circuit metadata
- `rust_poly_fp_*.cpp` - C++ polynomial evaluation (split by `--validity-split-count`)
- `eval_check_*.cu`, `eval_check.cuh` - CUDA validity checking kernels (split)
- `validity.rs.inc` - Rust validity implementation

**Validity Splitting**: Large circuits split validity checking into multiple files for compilation performance and GPU register pressure management. RV32IM v2 uses `--validity-split-count=4`.

### 6. Additional Optimization Passes

- Elide structs - Remove unnecessary struct indirection
- Expand layout - Flatten layout representations
- Dead code elimination (DCE)
- Common subexpression elimination (CSE)
- Degree checking - Verify polynomial constraints don't exceed maximum degree

### 7. Lower Step Functions
**Source**: `zirgen/Main/gen_zirgen.cpp` - `makeStepFuncs`

Lowers step functions for witness generation:
- Convert buffers to function arguments
- Canonicalize operations
- Privatize symbols
- Dead code elimination

Produces step functions module for code generation.

### 8. Emit Target Code
**Source**: `zirgen/Main/Utils.cpp` - `emitTarget`

Generates final code for each target (Rust, C++, CUDA):

**Common outputs for all targets**:
- `defs.*` - Constant definitions
- `types.*` - Type definitions
- `layout.*` - Memory layout (declaration and implementation)
- `steps.*` - Step functions (declaration and implementation)

**Note**: eval_check and polynomial code are generated in step 5, not here.

**Step Splitting**: Step functions can optionally be split with `--step-split-count` for large circuits. Keccak uses `--step-split-count=16`.

## Code Generation

### Target Emitters

**Rust**: `zirgen/compiler/codegen/gen_rust.cpp`
- Circuit metadata
- Polynomial extensions
- Type definitions
- Trait implementations

**C++**: `zirgen/compiler/codegen/gen_cpp.cpp`
- Step function implementations
- Layout definitions
- Header files with declarations

**CUDA**: `zirgen/compiler/codegen/gen_gpu.cpp`
- GPU kernel code
- Optimized for register pressure
- 64-bit buffer indexing for large circuits (po2 23/24)
- Template-based code generation

### GPU Kernel Optimization

GPU code generation includes critical optimizations:

**Register Pressure Management**:
- CUDA has a 255-register-per-thread limit
- `eval_check` kernels are 50%+ of proof time
- Exceeding the limit causes register spilling to local memory (major performance impact)

**Slot Pool System** (`gen_gpu.cpp`):
```
kFpSlotCount = 32          // Fp slots (s0..s31)
kFpExtSlotCount = 16       // FpExt slots (e0..e15, 4x Fp each)
kFpOverflowCount = 8       // Overflow slots (su0..su7)
kFpExtOverflowCount = 8    // FpExt overflow (eu0..eu7)
kBufSlotCount = 8          // Buffer slots (buf0..buf7)
```

**Liveness-Based Slot Reuse**: `computeLastUse` performs liveness analysis and reuses slots when values are no longer needed. Prefer-operand reuse allocates results into operand slots when possible (e.g., `s3 = s3 + s4`).

**64-bit Buffer Indexing**: For large circuits (po2 23/24), buffer accesses use 64-bit indices to prevent overflow:
```cuda
buf[((uint64_t)reg) * size + offset]
```

**Domain Size**: Templates use `1u << po2` (unsigned) to avoid signed overflow at po2 31.

**Templates**: `zirgen/compiler/codegen/gpu/`
- `eval_check.tmpl.cu` - Standard eval_check kernel
- `recursion/eval_check.tmpl.cu` - Recursion circuit variant
- `eval_check.tmpl.metal.h` - Metal (Apple Silicon)

Templates use Mustache syntax and are filled with generated polynomial bodies.

### INV_RATE and VRAM

**INV_RATE** is hardcoded to **4** in eval_check templates. It determines:
- Domain size: `domain = (1 << po2) × INV_RATE`
- Buffer allocation: Sized by domain × columns
- VRAM usage: INV_RATE=4 means 4× minimum domain size

This is a protocol-level constant. Changing it would require:
- Template updates
- RISC Zero prover changes
- Security/performance review

## Key Technologies

### MLIR/LLVM
**Purpose**: Multi-level intermediate representation infrastructure
**Usage**: Dialect definitions, transformation passes, optimization

### Bazel
**Purpose**: Build system
**Usage**:
- Hermetic builds with reproducible results
- Custom rules for circuit compilation (`build_circuit`, `zirgen_genfiles`)
- LLVM integration at specific commit
- Cross-compilation support (RISC-V, x86_64, arm64)

**Key Files**:
- `WORKSPACE` - Dependencies and toolchains
- `bazel/rules/zirgen/dsl-defs.bzl` - DSL compilation rules
- `bazel/rules/zirgen/edsl-defs.bzl` - Embedded DSL rules

### C++
**Purpose**: Primary implementation language
**Standards**: Modern C++ (C++17+)
**Usage**: Compiler, codegen, dialect implementations

### Rust
**Purpose**: Target language for generated code
**Usage**: Circuit implementation in RISC Zero

### CUDA
**Purpose**: GPU acceleration
**Usage**: eval_check kernels, polynomial evaluation
**Constraint**: 255 registers per thread limit

### Circom Integration
**Purpose**: R1CS circuit compatibility
**Components**:
- R1CS file format parser (`r1cs/r1csfile.cpp`)
- Witness file parser (`r1cs/wtnsfile.cpp`)
- R1CS → BigInt dialect lowering (`r1cs/lower.cpp`)
- Constraint validation (`r1cs/validate.cpp`)

### Field Arithmetic
**Default**: BabyBear prime field
**Purpose**: Finite field arithmetic for constraints
**Location**: `zirgen/compiler/zkp/`

## Related Documentation

- [Getting Started](zirgen/docs/01_Getting_Started.md) - User guide for writing circuits
- [Language Overview](zirgen/docs/02_Conceptual_Overview.md) - ZIR language concepts
- [Bazel Build Guide](docs/BAZEL_BUILD_GUIDE.md) - Building and testing
- [Scripts Reference](docs/SCRIPTS_REFERENCE.md) - Development scripts
- [Code Navigation](docs/CODE_NAVIGATION.md) - Finding your way around
- [AGENTS.md](AGENTS.md) - Detailed pipeline and codegen reference
- [Testing in RISC Zero](docs/TESTING_IN_RISC0.md) - Integration workflow
- [Generating CUDA Kernels](docs/GENERATING_CUDA_KERNELS.md) - GPU codegen
