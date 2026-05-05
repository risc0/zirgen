# Bazel Build Guide

This guide explains how to build Zirgen circuits using Bazel, including common build patterns, troubleshooting, and optimization techniques.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Build System Overview](#build-system-overview)
3. [Building Circuits](#building-circuits)
4. [Custom Build Rules](#custom-build-rules)
5. [Build Configuration](#build-configuration)
6. [Common Build Patterns](#common-build-patterns)
7. [Troubleshooting](#troubleshooting)

## Quick Start

```bash
# Install Bazel (or use Bazelisk for version management)
brew install bazelisk  # macOS
# or
brew install bazel

# Build the main codegen binary
bazel build //zirgen/Main:gen_zirgen

# Build RV32IM v2 circuit codegen
bazel build //zirgen/circuit/rv32im/v2/dsl:codegen

# Build Keccak circuit codegen
bazel build //zirgen/circuit/keccak:codegen

# Build recursion ZKR programs
bazel build //zirgen/circuit/predicates:recursion_zkr

# Run tests for a specific component
bazel test //zirgen/dsl:parser_test

# Run all tests
bazel test //...
```

## Build System Overview

### Workspace Configuration

The main `WORKSPACE` file defines external dependencies:

**LLVM Integration**:
- Specific commit: `39df4945e1888d407777e58fbf03f6ad1e859c11`
- Provides MLIR infrastructure

**Key Dependencies**:
- Bazel Skylib (utility functions)
- RISC-V toolchains
- Google Test (testing framework)
- Conda environment management
- Hermetic C++ toolchain (Zig-based)

### Platform Support

**Supported platforms** (`bazel/platform/`):
- **CPU**: x86_64, arm64
- **OS**: Linux, macOS

Build system automatically detects platform and selects appropriate toolchain.

### Toolchains

**RISC-V Toolchain** (`bazel/toolchain/risc0/`):
- RISC Zero compiler toolchain
- Cross-compilation support

**RV32IM Linux Toolchain** (`bazel/toolchain/rv32im-linux/`):
- RISC-V 32-bit target support

## Building Circuits

### DSL-Based Circuits

DSL circuits use the `.zir` file format:

```bash
# Build a DSL circuit
bazel build //zirgen/circuit/<circuit>/dsl:codegen

# Example: RV32IM v2
bazel build //zirgen/circuit/rv32im/v2/dsl:codegen

# Output location
ls bazel-bin/zirgen/circuit/rv32im/v2/dsl/
```

**Generated files** (depends on split configuration):
- `defs.*` - Constant definitions
- `types.*` - Type definitions
- `layout.*` - Memory layout
- `steps.*` - Step functions
- `poly_ext.rs` - Polynomial extensions
- `taps.rs` - Trace access patterns
- `info.rs` - Circuit metadata
- `rust_poly_fp_*.cpp` - Polynomial evaluation (split)
- `eval_check_*.cu` - CUDA validity kernels (split)
- `eval_check.cuh` - CUDA headers

### EDSL-Based Circuits

Embedded DSL circuits use C++ API:

```bash
# Build EDSL circuit
bazel build //zirgen/circuit/<circuit>:codegen

# Example: Recursion circuit
bazel build //zirgen/circuit/recursion:codegen

# Example: Fibonacci (simple example)
bazel build //zirgen/circuit/fib:codegen
```

### Testing Circuits Locally

Run circuit tests without full codegen:

```bash
# Test a ZIR file with built-in test framework
bazel run //zirgen/dsl:zirgen -- \
    path/to/circuit.zir \
    --test

# With include directories
bazel run //zirgen/dsl:zirgen -- \
    path/to/circuit.zir \
    -I path/to/components \
    --test
```

## Custom Build Rules

### zirgen_genfiles

Generates multiple files from a single ZIR input:

```python
load("//bazel/rules/zirgen:dsl-defs.bzl", "zirgen_genfiles")

zirgen_genfiles(
    name = "my_circuit",
    zir_file = "circuit.zir",
    data = [
        "//zirgen/components:all_components",
    ],
    zirgen_outs = [
        (["--opt1", "value"], "output1.rs"),
        (["--opt2", "value"], "output2.cpp"),
    ],
)
```

**Features**:
- Multiple output files with different options
- Generates `TestFresh<name>` rule to verify outputs are up-to-date
- Generates `Generate<name>` rule to regenerate files

### zirgen_build

Single output from ZIR file:

```python
load("//bazel/rules/zirgen:dsl-defs.bzl", "zirgen_build")

zirgen_build(
    name = "my_circuit_rust",
    zir_file = "circuit.zir",
    out = "circuit.rs",
    opts = ["--emit-rust"],
    data = ["//zirgen/components"],
)
```

### build_circuit

High-level rule for complete circuit builds (EDSL):

```python
load("//bazel/rules/zirgen:edsl-defs.bzl", "build_circuit")

build_circuit(
    name = "codegen",
    srcs = glob(["*.zir"]),
    main = "top.zir",
    outs = [
        "defs.rs.inc",
        "types.rs.inc",
        "layout.rs.inc",
        "steps.cpp",
        "steps.h",
        "eval_check_0.cu",
        "eval_check_1.cu",
        "eval_check.cuh",
        # ... more outputs
    ],
    circuit_name = "my_circuit",
    protocol_info = "MY_CIRCUIT:v1___",
    validity_split_count = 2,
    step_split_count = 0,  # 0 = no split
)
```

**Parameters**:
- `circuit_name` - Circuit identifier
- `protocol_info` - Protocol version string
- `validity_split_count` - Split validity checking (default: 1)
- `step_split_count` - Split step functions (default: 0)
- `outs` - Must match all generated files

**Important**: `outs` list must exactly match files produced by codegen with the given split counts.

## Build Configuration

### Validity Splitting

Large circuits split validity checking into multiple compilation units:

```bash
# RV32IM v2 uses 4-way split
--validity-split-count=4
```

**Generated files** (N = split count):
- `eval_check_0.cu` through `eval_check_{N-1}.cu`
- `rust_poly_fp_0.cpp` through `rust_poly_fp_{N-1}.cpp`

**Why split?**:
- Reduces compilation time (parallel compilation)
- Manages CUDA register pressure
- Smaller compilation units for better optimization

### Step Splitting

Optional splitting for step functions:

```bash
# Keccak uses 16-way split
--step-split-count=16
```

**Generated files** (N = split count):
- `steps_0.cu` through `steps_{N-1}.cu` (CUDA)
- `steps_0.cpp` through `steps_{N-1}.cpp` (C++)

**When to use**:
- Very large circuits with many step functions
- To parallelize compilation
- Keccak circuit uses this extensively

### Build Flags

**Low memory builds**:
```bash
bazel build --jobs=1 //zirgen/circuit/rv32im/v2/dsl:codegen
```

**Verbose output**:
```bash
bazel build --subcommands //zirgen/circuit/rv32im/v2/dsl:codegen
```

**Clean build**:
```bash
bazel clean
bazel build //zirgen/circuit/rv32im/v2/dsl:codegen
```

**Expunge (full clean)**:
```bash
bazel clean --expunge
```

## Common Build Patterns

### Building All Circuits

```bash
# Build all circuit codegens
bazel build //zirgen/circuit/...
```

### Incremental Development

```bash
# Build just the codegen binary
bazel build //zirgen/Main:gen_zirgen

# Manually run codegen (for quick iteration)
bazel-bin/zirgen/Main/gen_zirgen \
    --output-dir=./output \
    --circuit-name=test \
    -I zirgen/components \
    path/to/circuit.zir
```

### Running Tests

```bash
# Run all tests
bazel test //...

# Run specific test suite
bazel test //zirgen/dsl:all

# Run with test output
bazel test --test_output=all //zirgen/dsl:parser_test

# Run tests matching pattern
bazel test //zirgen/... --test_filter=*Parser*
```

### Building and Copying to RISC Zero

```bash
# Use the copy script (recommended)
./scripts/copy_codegen_to_risc0.sh

# Or manually:
bazel build //zirgen/circuit/rv32im/v2/dsl:codegen
bazel build //zirgen/circuit/keccak:codegen
bazel build //zirgen/circuit/predicates:recursion_zkr
# Then copy files per script logic
```

See [Scripts Reference](SCRIPTS_REFERENCE.md) for details.

### Verifying Output Freshness

```bash
# Check if generated files in repo are up-to-date
bazel test //zirgen/circuit/rv32im/v2/dsl:TestFreshcodegen

# Regenerate if stale
bazel run //zirgen/circuit/rv32im/v2/dsl:Generatecodegen
```

## Troubleshooting

### Build Failures

**LLVM/MLIR errors**:
```
Error: MLIR dialect not found
```

Solution: Clean and rebuild LLVM dependencies:
```bash
bazel clean
bazel build //zirgen/Main:gen_zirgen
```

**Missing outputs**:
```
Error: Output file not found: eval_check_3.cu
```

Solution: Check `outs` list in BUILD.bazel matches split count:
- validity-split-count=4 requires eval_check_0 through eval_check_3
- Update BUILD.bazel if split count changed

**Hermetic toolchain issues**:
```
Error: Cannot find compiler
```

Solution: Bazel uses hermetic Zig toolchain. Ensure WORKSPACE is properly configured:
```bash
bazel sync
bazel build //zirgen/Main:gen_zirgen
```

### Memory Issues

**Out of memory during build**:

Use `--jobs` flag to limit parallelism:
```bash
bazel build --jobs=1 //zirgen/circuit/rv32im/v2/dsl:codegen
```

Or use the low-memory script option:
```bash
./scripts/copy_codegen_to_risc0.sh --low-memory
```

**CUDA compilation OOM**:

When building RISC Zero with CUDA enabled:
```bash
cd risc0_repo
cargo build -F cuda -j 1  # Single job
```

### Caching Issues

**Stale build artifacts**:

```bash
# Clean build cache
bazel clean

# Full clean including external dependencies
bazel clean --expunge
```

**Remote cache issues** (if using):

```bash
# Disable remote cache temporarily
bazel build --remote_cache= //zirgen/circuit/rv32im/v2/dsl:codegen
```

### Test Failures

**Test doesn't reflect recent changes**:

```bash
# Run test without cache
bazel test --cache_test_results=no //zirgen/dsl:parser_test
```

**Test timeout**:

```bash
# Increase timeout
bazel test --test_timeout=300 //zirgen/dsl:parser_test
```

## Advanced Topics

### Custom Codegen Options

When using `build_circuit`, pass options via command-line flags in BUILD.bazel:

```python
build_circuit(
    name = "codegen",
    # ... other args ...
    opts = [
        "--validity-split-count=4",
        "--step-split-count=0",
        "--protocol-info=CUSTOM:v1___",
    ],
)
```

### Multiple Build Configurations

Build same circuit with different configurations:

```python
build_circuit(
    name = "codegen_cpu",
    main = "top.zir",
    # CPU-optimized config
)

build_circuit(
    name = "codegen_gpu",
    main = "top.zir",
    # GPU-optimized config
)
```

### Integration with Other Build Systems

Export Bazel outputs for use in Cargo/CMake:

```bash
# Build and export to known location
bazel build //zirgen/circuit/rv32im/v2/dsl:codegen
cp -r bazel-bin/zirgen/circuit/rv32im/v2/dsl/* /path/to/rust/project/
```

See `scripts/copy_codegen_to_risc0.sh` for a complete example.

## Related Documentation

- [Architecture](../ARCHITECTURE.md) - Compiler architecture
- [Scripts Reference](SCRIPTS_REFERENCE.md) - Development scripts
- [Testing in RISC Zero](TESTING_IN_RISC0.md) - Integration workflow
- [Code Navigation](CODE_NAVIGATION.md) - Codebase structure
