# Zirgen Compiler Architecture

This document provides an overview of the Zirgen compiler architecture, including the multi-stage compilation pipeline, dialect hierarchy, transformation passes, and code generation.

## Overview

The Zirgen compiler is built on **MLIR (Multi-Level Intermediate Representation)** and transforms high-level circuit descriptions written in the Zirgen DSL into efficient implementations across multiple backends (Rust, C++, GPU). The compilation process involves multiple intermediate representations (dialects), each serving a specific purpose in the transformation pipeline.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                      Input: .zir DSL File                        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
                    ┌────────────────┐
                    │  DSL Parser    │
                    └────────┬───────┘
                             │
                             ▼
                    ┌────────────────┐
                    │      AST       │
                    └────────┬───────┘
                             │
                             ▼
         ┌───────────────────────────────────────────────┐
         │  Lower to ZHL Dialect (Untyped High-Level)    │
         │  - All values are generic Expr types          │
         │  - Direct mapping from DSL constructs          │
         └───────────────────┬───────────────────────────┘
                             │
                             ▼
         ┌───────────────────────────────────────────────┐
         │  Type Checking → ZHLT Dialect (Typed)         │
         │  - Resolve component types                    │
         │  - Add field types (!zll.val)                 │
         │  - Introduce layout types                     │
         └───────────────────┬───────────────────────────┘
                             │
                             ▼
         ┌───────────────────────────────────────────────┐
         │  ZHLT Transformation Passes                   │
         │  - ElideRedundantMembers                      │
         │  - HoistAllocs, HoistCommonMuxCode            │
         │  - GenerateSteps, LowerStepFuncs              │
         │  - OptimizeParWitgen, OutlineIfs              │
         └───────────────────┬───────────────────────────┘
                             │
                             ▼
         ┌───────────────────────────────────────────────┐
         │  Lower to ZStruct Dialect (Structured Types)  │
         │  - Explicit struct/array/union types          │
         │  - Layout management                          │
         │  - Memory access operations                   │
         └───────────────────┬───────────────────────────┘
                             │
                             ▼
         ┌───────────────────────────────────────────────┐
         │  ZStruct Transformation Passes                │
         │  - OptimizeLayout, ExpandLayout               │
         │  - InlineLayout, Unroll                       │
         │  - BuffersToArgs                              │
         └───────────────────┬───────────────────────────┘
                             │
                             ▼
         ┌───────────────────────────────────────────────┐
         │  Lower to Zll Dialect (Low-Level IR)          │
         │  - Field arithmetic operations                │
         │  - Buffer operations with taps                │
         │  - Constraint operations                      │
         │  - Control flow (if, nondet, barrier)         │
         └───────────────────┬───────────────────────────┘
                             │
                             ▼
         ┌───────────────────────────────────────────────┐
         │  Zll Optimization Passes                      │
         │  - SplitStage (per execution stage)           │
         │  - MakePolynomial (constraint polynomial)     │
         │  - ComputeTaps (register access analysis)     │
         │  - BalancedSplit, InlineFpExt                 │
         │  - IfToMultiply / MultiplyToIf                │
         └───────────────────┬───────────────────────────┘
                             │
                     ┌───────┴────────┐
                     │                │
                     ▼                ▼
           ┌──────────────┐  ┌──────────────┐
           │ Code Gen:    │  │ Code Gen:    │
           │ Rust         │  │ C++          │
           └──────┬───────┘  └──────┬───────┘
                  │                 │
                  ▼                 ▼
         ┌─────────────┐   ┌─────────────┐
         │ .rs files   │   │ .cpp/.h     │
         └─────────────┘   └─────────────┘

                              ▼
                     ┌──────────────┐
                     │ Code Gen:    │
                     │ GPU (CUDA/   │
                     │     Metal)   │
                     └──────┬───────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │ .cu / .metal    │
                   └─────────────────┘

Alternative Entry Point:
┌─────────────────┐
│ R1CS Format     │
│ (External)      │
└────────┬────────┘
         │
         ▼
┌──────────────────┐
│ R1CS Dialect     │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ BigInt Dialect   │
└────────┬─────────┘
         │
         └───────→ (joins Zll pipeline)
```

## Dialect Hierarchy

The Zirgen compiler uses multiple MLIR dialects, each representing a different abstraction level:

### 1. ZHL (Zirgen High-Level) - Untyped
**Location**: `zirgen/Dialect/ZHL/`

- **Purpose**: First IR after parsing, untyped representation
- **Key Feature**: Single `Expr` type for all values
- **Operations**: Components, arrays, lookups, constraints, control flow
- **Next Stage**: Type checking → ZHLT

[Full Documentation](../zirgen/Dialect/ZHL/README.md)

### 2. ZHLT (Zirgen High-Level Typed) - Typed
**Location**: `zirgen/Dialect/ZHLT/`

- **Purpose**: Typed high-level IR after type checking
- **Key Feature**: Full type information (components, fields, layouts)
- **Operations**: Typed component operations, layout management, step functions
- **Passes**: ElideRedundantMembers, HoistAllocs, GenerateSteps, OptimizeParWitgen
- **Next Stage**: Lowering → ZStruct

[Full Documentation](../zirgen/Dialect/ZHLT/README.md)

### 3. ZStruct (Zirgen Structured Types)
**Location**: `zirgen/Dialect/ZStruct/`

- **Purpose**: Structured data representation with explicit layouts
- **Key Feature**: Struct, union, array types with layout information
- **Operations**: Lookup, subscript, load, store, pack
- **Passes**: OptimizeLayout, InlineLayout, Unroll, BuffersToArgs
- **Next Stage**: LowerComposites → Zll

[Full Documentation](../zirgen/Dialect/ZStruct/README.md)

### 4. Zll (Zirgen Low-Level) - Core Dialect
**Location**: `zirgen/Dialect/Zll/`

- **Purpose**: Low-level circuit IR for constraint generation and code generation
- **Key Feature**: Field arithmetic, buffer operations, constraints, taps
- **Operations**: Arithmetic (add, mul, inv), memory (get, set), constraints (eqz), control flow (if, nondet, barrier), crypto (hash, digest)
- **Passes**: SplitStage, MakePolynomial, ComputeTaps, BalancedSplit, IfToMultiply
- **Next Stage**: Code generation → Rust/C++/GPU

[Full Documentation](../zirgen/Dialect/Zll/README.md)

### 5. IOP (Interactive Oracle Proof)
**Location**: `zirgen/Dialect/IOP/`

- **Purpose**: Verifier transcript operations
- **Key Feature**: Proof reading, commitment, challenge generation
- **Operations**: read, commit, rng_bits, rng_val
- **Usage**: Embedded in Zll verification functions

[Full Documentation](../zirgen/Dialect/IOP/README.md)

### 6. R1CS (Rank-1 Constraint System) - Import Only
**Location**: `zirgen/Dialect/R1CS/`

- **Purpose**: Import external R1CS constraint systems
- **Key Feature**: Standard R1CS format support
- **Operations**: def (wire), mul (factor), sum, constrain
- **Flow**: R1CS → BigInt → Zll

[Full Documentation](../zirgen/Dialect/R1CS/README.md)

## Compilation Pipeline Stages

### Stage 1: Parsing and Lowering
**Input**: `.zir` DSL file
**Output**: ZHL module

1. **Parse** (`zirgen/dsl/parser.cpp`): Tokenize and parse DSL syntax → AST
2. **Lower** (`zirgen/dsl/lower.cpp`): Transform AST → ZHL operations
3. All values are generic `!zhl.expr` type

### Stage 2: Type Checking
**Input**: ZHL module
**Output**: ZHLT module

1. **Type Check** (`zirgen/Conversions/Typing/ComponentManager.h`):
   - Resolve component types
   - Infer field types
   - Add layout information
   - Validate constraints

### Stage 3: High-Level Optimization
**Input**: ZHLT module
**Output**: Optimized ZHLT module

Passes (from `gen_zirgen.cpp:199-227`):
- Accumulation and global variable handling
- `ElideRedundantMembers` - Remove duplicate struct fields
- Field DCE (Dead Code Elimination)
- Field CSE (Common Subexpression Elimination)
- Type inference and checking
- `GenerateSteps` - Create step functions
- `InlinePure` - Inline pure operations
- `HoistInvariants` - Move loop-invariant code
- `LowerStepFuncs` - Convert to step function format

### Stage 4: Structural Lowering
**Input**: ZHLT module
**Output**: ZStruct module

1. Lower typed components to structured types
2. Add explicit layout operations
3. Introduce reference types

### Stage 5: Structural Optimization
**Input**: ZStruct module
**Output**: Optimized ZStruct module

Passes:
- `OptimizeLayout` - Reorder fields for efficiency
- `ExpandLayout` - Expand global layouts
- `Unroll` - Unroll map/reduce operations
- `InlineLayout` - Inline layout offsets
- `BuffersToArgs` - Convert buffers to function arguments

### Stage 6: Low-Level Lowering
**Input**: ZStruct module
**Output**: Zll module

**LowerComposites Pass** (`zirgen/Dialect/Zll/Conversion/ZStructToZll/`):
- `zstruct.load` → `zll.get` / `zll.get_global`
- `zstruct.store` → `zll.set` / `zll.set_global`
- Struct/array types → buffer types
- Layout abstractions → concrete buffer offsets

### Stage 7: Low-Level Optimization
**Input**: Zll module
**Output**: Optimized Zll module

**Per-Stage Processing** (for each execution stage):
1. `SplitStagePass` - Extract stage (exec, verify_mem, verify_bytes, compute_accum, verify_accum)
2. Stage-specific optimizations
3. Canonicalization
4. CSE

**Polynomial Generation**:
1. `MakePolynomialPass` - Convert constraints to polynomial form
2. `ComputeTapsPass` - Assign tap indices to register accesses
3. Additional optimizations

**Other Passes**:
- `BalancedSplit` - Split large blocks (default 1000 ops)
- `IfToMultiply` / `MultiplyToIf` - Conditional transformations
- `InlineFpExt` - Inline field extension operations
- `AddReductions` - Add reduction operations
- `DropConstraints` - Remove constraints for execution
- `SortForReproducibility` - Deterministic ordering

### Stage 8: Code Generation
**Input**: Optimized Zll module
**Output**: Target code (Rust/C++/GPU)

See [Code Generation Documentation](CODEGEN.md) for details.

## Key Compiler Components

### Tools
Located in `zirgen/compiler/tools/`:

1. **zirgen-opt**: Main optimizer and pass driver
   - Registers all dialect passes
   - Supports MLIR pass CLI options
   - Primary tool for IR transformations

2. **zirgen-translate**: Code generation frontend
   - Rust code generation (`zirgen-to-rust-*`)
   - C++ code generation (`cpp-codegen`)
   - Polynomial operations
   - Tap definitions

3. **zirgen-r1cs**: R1CS format importer
   - Imports external R1CS files
   - Converts to Zll via BigInt

### Main Compilation Entry Point
**File**: `zirgen/Main/gen_zirgen.cpp`

Orchestrates the full pipeline:
1. Parse DSL (line 182)
2. Type check (line 187)
3. Apply ZHLT passes (lines 199-227)
4. Lower to Zll
5. Apply Zll passes
6. Code generation

### Pass Infrastructure
Located in `zirgen/compiler/passes/`:

- Pass registration
- Pass utilities
- Pass management helpers

### Code Generation
Located in `zirgen/compiler/codegen/`:

- `codegen.cpp` - Main pipeline orchestration
- `gen_rust.cpp` - Rust emission
- `gen_cpp.cpp` - C++ emission
- `gen_gpu.cpp` - GPU emission (CUDA/Metal)
- `gpu/` - GPU template files

## Analysis Infrastructure

### Zll Analysis
Located in `zirgen/Dialect/Zll/Analysis/`:

- **DegreeAnalysis**: Tracks polynomial degree of constraints
- **TapsAnalysis**: Analyzes register accesses and cycle offsets
- **MixPowerAnalysis**: Analyzes constraint polynomial mixing
- **BigInt Support**: Range tracking for overflow detection

### Buffer Analysis
- Identifies buffer usage patterns
- Optimizes buffer allocation
- Converts buffers to function arguments

## Execution Stages

Circuits are decomposed into multiple execution stages separated by `zll.barrier` operations:

1. **exec**: Main execution logic
2. **verify_mem**: Memory verification
3. **verify_bytes**: Byte verification
4. **compute_accum**: Accumulator computation
5. **verify_accum**: Accumulator verification

Each stage is extracted and compiled separately for modular circuit execution.

## Supported Fields

- **BabyBear**: Prime p = 2^27 * 15 + 1, extension degree 4
- **Goldilocks**: Prime p = 2^64 - 2^32 + 1, extension degree 2

Both base fields and extension fields are fully supported throughout the pipeline.

## Type System

### Scalar Types (from Zll)
- `!zll.val<Field>` - Field element (base or extension)
- `!zll.digest` - Cryptographic digest
- `!zll.constraint` - Constraint polynomial

### Structured Types (from ZStruct)
- `!zstruct.struct` - Component struct
- `!zstruct.union` - Mux/union type
- `!zstruct.array` - Fixed-size array
- `!zstruct.layout` - Memory layout
- `!zstruct.ref` - Reference type

### Memory Types
- `!zll.buffer` - Memory buffer (Constant/Mutable/Global/Temporary)

### Special Types
- `!iop.iop` - IOP transcript handle
- `!r1cs.wire`, `!r1cs.factor` - R1CS constraint types

## Key Architectural Patterns

### 1. Multi-Level IR
Progressive lowering through multiple abstraction levels enables:
- Targeted optimizations at each level
- Clean separation of concerns
- Maintainable compiler architecture

### 2. Dialect-Specific Operations
Each dialect defines operations tailored to its abstraction level, with well-defined lowering between dialects.

### 3. Barrier-Based Stage Separation
`zll.barrier` operations delimit execution stages, enabling efficient multi-stage decomposition and parallel compilation.

### 4. Tap-Based Memory Abstraction
Register access uses "tap" indices, enabling flexible memory layouts and optimization.

### 5. Template-Driven Codegen
Code generation uses Mustache templates for flexible multi-target emission.

### 6. Constraint as First-Class Operations
Constraints are explicit operations (`zll.eqz`, `zll.and_eqz`), enabling constraint-aware optimization.

## Performance Optimizations

### Compiler-Level
- Aggressive inlining of pure functions
- Common subexpression elimination
- Dead code elimination
- Loop unrolling
- Field-aware constant folding

### Circuit-Level
- Layout optimization for memory access
- Balanced splitting of large functions
- Conditional transformation (if ↔ multiply)
- Parallel witness generation optimizations

### Code Generation
- Target-specific optimizations (SIMD for GPU)
- Template-based emission for efficiency
- Inline hints for critical paths

## File Organization

```
zirgen/
├── Dialect/          # MLIR dialect definitions
│   ├── ZHL/         # Untyped high-level
│   ├── ZHLT/        # Typed high-level
│   ├── ZStruct/     # Structured types
│   ├── Zll/         # Low-level (core)
│   ├── IOP/         # Interactive oracle proof
│   ├── R1CS/        # R1CS import
│   └── BigInt/      # Big integer operations
├── compiler/
│   ├── codegen/     # Code generation backends
│   ├── tools/       # CLI tools (opt, translate, r1cs)
│   ├── passes/      # Pass infrastructure
│   ├── layout/      # Layout management
│   └── zkp/         # Cryptographic utilities
├── dsl/             # DSL parser and lowering
├── Conversions/     # Dialect conversions
│   └── Typing/      # ZHL → ZHLT type checking
└── Main/            # Main compilation entry points
```

## See Also

- [Code Generation](CODEGEN.md) - Multi-target code emission
- [Zll Dialect](../zirgen/Dialect/Zll/README.md) - Core low-level IR
- [ZHLT Dialect](../zirgen/Dialect/ZHLT/README.md) - Typed high-level IR
- [ZStruct Dialect](../zirgen/Dialect/ZStruct/README.md) - Structured types
- [ZHL Dialect](../zirgen/Dialect/ZHL/README.md) - Untyped high-level IR
- [IOP Dialect](../zirgen/Dialect/IOP/README.md) - Verifier operations
- [R1CS Dialect](../zirgen/Dialect/R1CS/README.md) - R1CS import
