# Code Navigation Guide

This guide helps developers navigate the Zirgen codebase and find specific functionality.

## Table of Contents

1. [Quick Reference](#quick-reference)
2. [Finding Functionality](#finding-functionality)
3. [Key Files by Task](#key-files-by-task)
4. [Directory Deep Dive](#directory-deep-dive)

## Quick Reference

### Common Development Tasks

| Task | Location |
|------|----------|
| Add new DSL syntax | `zirgen/dsl/parser.cpp` (parser), `zirgen/dsl/lexer.cpp` (lexer), `zirgen/dsl/ast.cpp` (AST) |
| Modify AST → MLIR lowering | `zirgen/dsl/lower.cpp` |
| Add new dialect operation | `zirgen/Dialect/<dialect>/<dialect>.td`, then implement in `.cpp` |
| Change codegen pipeline | `zirgen/Main/gen_zirgen.cpp` (main pipeline), `zirgen/Main/Main.cpp` (passes) |
| Modify GPU codegen | `zirgen/compiler/codegen/gen_gpu.cpp` |
| Change GPU kernel template | `zirgen/compiler/codegen/gpu/eval_check.tmpl.cu` (standard), `gpu/recursion/eval_check.tmpl.cu` (recursion) |
| Add new Rust codegen | `zirgen/compiler/codegen/gen_rust.cpp` |
| Add new C++ codegen | `zirgen/compiler/codegen/gen_cpp.cpp` |
| Modify register allocation | `zirgen/compiler/codegen/gen_gpu.cpp:GpuPolyContext`, `computeLastUse` |
| Add typing pass | `zirgen/Conversions/Typing/` |
| Add optimization pass | `zirgen/Dialect/<dialect>/Transforms/` |
| Add new circuit | `zirgen/circuit/<circuit_name>/` |
| Modify RV32IM circuit | `zirgen/circuit/rv32im/v2/dsl/` (ZIR files) |
| Modify recursion circuit | `zirgen/circuit/recursion/` (C++ EDSL) |
| Add reusable component | `zirgen/components/` (ZIR library) |
| Modify verification | `zirgen/circuit/verify/` |
| Change build rules | `bazel/rules/zirgen/` |

### Where to Look By Symptom

| Symptom | Check These Files |
|---------|------------------|
| Compilation error in `.zir` file | `zirgen/dsl/parser.cpp` (parser), check component includes |
| Type error in circuit | `zirgen/Conversions/Typing/` (typing passes) |
| Wrong codegen output | `zirgen/compiler/codegen/gen_*.cpp` (target emitters) |
| CUDA kernel compile error | `zirgen/compiler/codegen/gpu/eval_check.tmpl.cu` (template) |
| Register spilling in GPU | `zirgen/compiler/codegen/gen_gpu.cpp` (slot allocation, adjust `kFpSlotCount`) |
| Missing generated file | Check BUILD.bazel `outs` list, verify split count |
| Wrong circuit behavior | Circuit-specific files in `zirgen/circuit/<name>/` |
| Test failure | Corresponding `*_test.cpp` file |
| Build failure | `WORKSPACE`, `BUILD.bazel`, `bazel/rules/` |
| Integration with RISC Zero | `scripts/copy_codegen_to_risc0.sh`, `docs/TESTING_IN_RISC0.md` |

## Finding Functionality

### Entry Points

**Main compiler binary**: `zirgen/Main/gen_zirgen.cpp`
- Parses command-line options
- Runs full compilation pipeline
- Entry point for codegen

**DSL compiler**: `zirgen/dsl/driver.cpp`
- Standalone DSL compilation
- Testing entry point

**Tools**:
- `zirgen/compiler/tools/zirgen-opt.cpp` - MLIR optimization tool
- `zirgen/compiler/tools/zirgen-translate.cpp` - MLIR translation tool
- `zirgen/compiler/tools/zirgen-r1cs.cpp` - R1CS/Circom integration

### Pipeline Flow

Follow code execution through the compiler:

1. **Parse**: `zirgen/dsl/Parser::parse()` → AST
2. **Lower**: `zirgen/dsl::lower()` → MLIR
3. **Typecheck**: `zirgen::Typing::typeCheck()`
4. **Passes**: See `zirgen/Main/Main.cpp:addTypingPasses()`
5. **Emit Poly**: `zirgen/compiler/codegen/codegen.cpp:emitCodeZirgenPoly()`
6. **Emit Target**: `zirgen/Main/Utils.cpp:emitTarget()`

### Dialect Implementations

Each dialect has consistent structure:

```
zirgen/Dialect/<dialect>/
├── <dialect>.td              # TableGen definitions
├── <dialect>.cpp             # Dialect registration
├── IR/                       # Operation implementations
│   ├── <dialect>.cpp
│   └── <dialect>Ops.cpp
├── Transforms/               # Transformation passes
│   └── *.cpp
└── Analysis/                 # Analysis passes
    └── *.cpp
```

### Code Generation

**High-level codegen**: `zirgen/Main/Utils.cpp`
- `emitTarget()` - Main codegen entry
- `emitDefs()`, `emitTypes()` - Generate definitions
- `emitOps()` - Generate operations

**Target-specific emitters**: `zirgen/compiler/codegen/`
- `gen_rust.cpp:RustEmitter` - Rust code generation
- `gen_cpp.cpp:CppEmitter` - C++ code generation
- `gen_gpu.cpp:GpuPolyContext` - CUDA/Metal generation

**Templates**: `zirgen/compiler/codegen/gpu/`
- `eval_check.tmpl.cu` - CUDA kernel template
- `eval_check.tmpl.metal.h` - Metal kernel template
- `recursion/eval_check.tmpl.cu` - Recursion variant
- Filled using `mustache.h` template engine

## Key Files by Task

### Working with DSL

**Adding syntax**:

1. `zirgen/dsl/lexer.cpp` - Add token
2. `zirgen/dsl/parser.cpp` - Add parsing rule
3. `zirgen/dsl/ast.h` - Add AST node
4. `zirgen/dsl/lower.cpp` - Add lowering to MLIR

**Example**: Component definition
- Lexer: `kw_component` token
- Parser: `parseComponent()` method
- AST: `Component` class
- Lower: `lowerComponent()` method

**Testing**:
- `zirgen/dsl/test/` - Test files
- Add `.zir` test file with expected behavior
- Run: `bazel test //zirgen/dsl:all`

### Working with MLIR Dialects

**Adding new operation**:

1. Define in `zirgen/Dialect/<dialect>/<dialect>.td`:
```tablegen
def MyOp : <Dialect>_Op<"my_op", [Traits]> {
  let summary = "Does something";
  let arguments = (ins ...);
  let results = (outs ...);
}
```

2. Build to generate C++ code:
```bash
bazel build //zirgen/Dialect/<dialect>:ir
```

3. Implement in `zirgen/Dialect/<dialect>/IR/<dialect>Ops.cpp`:
```cpp
LogicalResult MyOp::verify() { ... }
void MyOp::build(...) { ... }
```

4. Add lowering pass if needed:
`zirgen/Dialect/<dialect>/Transforms/LowerMyOp.cpp`

**Testing**:
- Create test in `zirgen/Dialect/<dialect>/test/`
- Use `lit` testing framework
- Run: `bazel test //zirgen/Dialect/<dialect>:all`

### Working with Code Generation

**Modifying output format**:

1. Find target emitter: `zirgen/compiler/codegen/gen_<target>.cpp`
2. Locate relevant `emit*` method
3. Modify code generation logic
4. Test with: `bazel build //zirgen/circuit/<circuit>:codegen`

**Adding new output file**:

1. Add to `FileEmitter` in `codegen.cpp`:
```cpp
emitter.addFile("newfile.rs", [](auto& os) {
  os << "content";
});
```

2. Update BUILD.bazel `outs` list:
```python
outs = [
    # ... existing files ...
    "newfile.rs",
]
```

3. Update `scripts/copy_codegen_to_risc0.sh` if integrating

**GPU kernel optimization**:

File: `zirgen/compiler/codegen/gen_gpu.cpp`

Key areas:
- `kFpSlotCount`, `kFpExtSlotCount` - Slot pool sizes
- `GpuPolyContext::emitPolyOpWithSlots()` - Operation emission
- `computeLastUse()` - Liveness analysis
- `allocSlot()`, `freeSlot()` - Slot management

Debug register usage:
```cpp
#define ZIRGEN_TRACK_PEAK_SLOTS 1  // In gen_gpu.cpp
```

### Working with Circuits

**DSL circuits** (`.zir` files):

Location: `zirgen/circuit/<circuit>/dsl/`

Key files:
- `top.zir` - Top-level circuit
- `BUILD.bazel` - Build configuration
- Other `.zir` files - Components

Modify circuit:
1. Edit `.zir` files
2. Build: `bazel build //zirgen/circuit/<circuit>/dsl:codegen`
3. Test locally: `bazel run //zirgen/dsl:zirgen -- circuit.zir --test`

**EDSL circuits** (C++ API):

Location: `zirgen/circuit/<circuit>/`

Key files:
- `*.cpp` - Circuit implementation
- `BUILD.bazel` - Build configuration

Example: `zirgen/circuit/fib/fib.cpp` (Fibonacci)

Modify circuit:
1. Edit C++ source
2. Build: `bazel build //zirgen/circuit/<circuit>:codegen`

**RV32IM v2 (main zkVM circuit)**:

Location: `zirgen/circuit/rv32im/v2/`

Key directories:
- `dsl/` - ZIR source files
  - `top.zir` - Top level
  - `decode.zir` - Instruction decode
  - `inst_*.zir` - Instruction implementations
- `kernel/` - Kernel implementations
- `platform/` - Platform-specific code
- `run/` - Runtime support
- `emu/` - Emulator

Build configuration:
- `dsl/BUILD.bazel` - Defines codegen target
- Uses `--validity-split-count=4`
- Protocol info: `RV32IM:v2rev2___`

## Directory Deep Dive

### `zirgen/` - Main Compiler

```
zirgen/
├── Dialect/              # MLIR dialects (IR definitions)
├── Conversions/          # Dialect lowering and conversion passes
├── Main/                 # Entry points and top-level pipeline
├── dsl/                  # DSL parser, lexer, AST, lowering
├── compiler/             # Code generation and utilities
├── circuit/              # Circuit implementations
└── components/           # Reusable ZIR components
```

### `zirgen/Dialect/` - MLIR Dialects

Each subdirectory is a dialect:

**Zll** - Low-level IR:
- `zirgen/Dialect/Zll/IR/` - Operation definitions
- `zirgen/Dialect/Zll/Transforms/` - Optimization passes
- Field arithmetic, buffer access, polynomial ops

**ZStruct** - Layout and structs:
- `zirgen/Dialect/ZStruct/IR/` - Layout operations
- Struct management, buffer allocation

**ZHLT** - High-level typed:
- `zirgen/Dialect/ZHLT/IR/` - CheckFuncOp, StepFuncOp
- Constraint and witness generation

**Others**: ZHL, BigInt, R1CS, IOP

### `zirgen/dsl/` - DSL Implementation

```
dsl/
├── parser.cpp/h          # Recursive descent parser
├── lexer.cpp/h           # Lexical analyzer
├── ast.cpp/h             # Abstract syntax tree
├── lower.cpp/h           # AST → MLIR lowering
├── driver.cpp            # Compilation driver
├── passes/               # DSL-level passes
├── Analysis/             # Layout analysis
├── examples/             # Example circuits
└── test/                 # Test suite
```

Key entry points:
- `Parser::parse()` - Parse ZIR source
- `lower()` - Lower AST to MLIR
- `Driver::compile()` - Full compilation

### `zirgen/compiler/` - Code Generation

```
compiler/
├── codegen/              # Code generation
│   ├── codegen.cpp/h     # Main codegen driver
│   ├── gen_rust.cpp      # Rust emitter
│   ├── gen_cpp.cpp       # C++ emitter
│   ├── gen_gpu.cpp       # CUDA/Metal emitter
│   ├── mustache.h        # Template engine
│   └── gpu/              # GPU templates
├── zkp/                  # ZKP primitives
│   ├── poseidon.cpp      # Poseidon hash
│   ├── sha256.cpp        # SHA256
│   └── baby_bear.cpp     # Field arithmetic
├── r1cs/                 # Circom integration
│   ├── r1csfile.cpp      # R1CS parser
│   ├── wtnsfile.cpp      # Witness parser
│   └── lower.cpp         # R1CS → BigInt
├── layout/               # Layout computation
├── stats/                # Circuit statistics
└── tools/                # Standalone tools
```

### `zirgen/circuit/` - Circuits

```
circuit/
├── rv32im/               # RISC-V zkVM
│   ├── v1/               # Version 1 (EDSL)
│   └── v2/               # Version 2 (DSL, current)
│       ├── dsl/          # ZIR source
│       ├── kernel/       # Implementations
│       ├── platform/     # Platform code
│       └── run/          # Runtime
├── recursion/            # Recursion circuit (EDSL)
│   ├── bits.cpp
│   ├── code.cpp
│   ├── poseidon2.cpp
│   └── ...
├── keccak/               # Keccak circuit (DSL)
│   ├── keccak.zir
│   └── ...
├── bigint/               # BigInt operations
├── fib/                  # Fibonacci example
└── verify/               # Verification circuits
```

### `zirgen/Main/` - Pipeline

```
Main/
├── gen_zirgen.cpp        # Main entry point
├── Main.cpp/h            # Pipeline setup
├── Utils.cpp/h           # Codegen utilities
├── Target.cpp/h          # Target emission
└── RunTests.cpp/h        # Test execution
```

Pipeline stages:
1. Parse (calls `Parser::parse`)
2. Lower (calls `dsl::lower`)
3. Typecheck (calls `Typing::typeCheck`)
4. Passes (calls `addTypingPasses`, etc.)
5. Emit poly (calls `emitCodeZirgenPoly`)
6. Emit target (calls `emitTarget`)

### `bazel/` - Build System

```
bazel/
├── rules/
│   ├── zirgen/
│   │   ├── dsl-defs.bzl      # DSL build rules
│   │   └── edsl-defs.bzl     # EDSL build rules
│   ├── lit/                  # LLVM lit testing
│   └── clang_format/         # Formatting
├── platform/                 # Platform detection
└── toolchain/                # Cross-compilation
```

Key rules:
- `zirgen_genfiles` - Generate multiple outputs
- `zirgen_build` - Single output
- `build_circuit` - Complete circuit build

### `scripts/` - Development Scripts

```
scripts/
├── copy_codegen_to_risc0.sh  # Main integration script
└── setup-git-hooks.sh         # Git hooks setup
```

See [Scripts Reference](SCRIPTS_REFERENCE.md) for details.

## Related Documentation

- [Architecture](../ARCHITECTURE.md) - System architecture overview
- [Bazel Build Guide](BAZEL_BUILD_GUIDE.md) - Building and testing
- [Scripts Reference](SCRIPTS_REFERENCE.md) - Development scripts
- [Getting Started](../zirgen/docs/01_Getting_Started.md) - User guide
- [Language Overview](../zirgen/docs/02_Conceptual_Overview.md) - ZIR concepts
- [AGENTS.md](../AGENTS.md) - Detailed technical reference
