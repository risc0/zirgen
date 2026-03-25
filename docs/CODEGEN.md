# Zirgen Code Generation

This document describes the code generation subsystem in the Zirgen compiler, which transforms optimized Zll IR into executable code for multiple backends: Rust, C++, and GPU (CUDA/Metal).

## Overview

The Zirgen compiler supports multi-target code generation from a single optimized Zll IR. Each backend produces code optimized for its execution environment:

- **Rust**: Step functions, polynomial evaluation, memory layouts
- **C++**: Polynomial evaluators, constraint checking
- **GPU (CUDA/Metal)**: Massively parallel polynomial evaluation and step execution

## Architecture

```
Optimized Zll Module
    │
    ├──────────────┬──────────────┬─────────────┐
    │              │              │             │
    ▼              ▼              ▼             ▼
Rust Codegen   C++ Codegen   CUDA Codegen  Metal Codegen
    │              │              │             │
    ▼              ▼              ▼             ▼
.rs files      .cpp/.h        .cu files    .metal files
```

## Code Generation Pipeline

Located in `zirgen/compiler/codegen/codegen.cpp`, the pipeline consists of:

### Stage 1: Simple Optimization
```cpp
optimizeSimple(module)
├── Canonicalizer pass
└── CSE (Common Subexpression Elimination)
```

### Stage 2: Stage-Based Splitting
For each execution stage (exec, verify_mem, verify_bytes, compute_accum, verify_accum):
```cpp
optimizeSplit(module, stage, opts)
├── SplitStagePass(stage)      // Extract barrier-delimited section
├── Custom extra passes         // Optional stage-specific passes
├── Canonicalizer
└── CSE
```

**Emits**:
- `rust_step_{stage}.cpp` (Rust code in C++ wrapper)
- `step_{stage}.cu` (CUDA kernel)
- `step_{stage}.metal` (Metal kernel)

### Stage 3: Polynomial Generation
```cpp
optimizePoly(module, opts)
├── MakePolynomialPass         // Convert constraints to ConstraintType
├── Canonicalizer
├── CSE
└── ComputeTapsPass            // Finalize tap indices
```

**Emits**:
- `rust_poly_fp.cpp` (or split versions `rust_poly_fp_{i}.cpp`)
- `poly_ext.rs` (extension field polynomial)
- `taps.rs` / `taps.cpp` (tap definitions)
- `info.rs` (protocol info)
- `eval_check.cu` / `eval_check.metal` (GPU polynomial evaluation)

### Stage 4: Layout Emission
```cpp
emitAllLayouts(module)
```

**Emits**:
- `layout.rs.inc` (Rust syntax)
- `layout.cpp.inc` (C++ syntax)
- `layout.cu.inc` (CUDA syntax)

## Backend-Specific Details

### Rust Code Generation

**Location**: `zirgen/compiler/codegen/gen_rust.cpp`

**Key Class**: `RustStreamEmitter`

**Generated Files**:
1. **Step Functions** (`rust_step_{stage}.cpp`):
   - Wraps Rust code in C++ for FFI
   - Step execution logic per stage
   - Memory access patterns

2. **Polynomial Functions** (`rust_poly_fp.cpp`):
   - Polynomial constraint evaluation
   - May be split into multiple files for large circuits
   - Base field operations

3. **Extension Polynomial** (`poly_ext.rs`):
   - Extension field polynomial evaluation
   - Handles degree 2/4 extensions

4. **Tap Definitions** (`taps.rs`):
   - Register access patterns
   - Cycle offset information
   - Buffer tap indices

5. **Layout Definitions** (`layout.rs.inc`):
   - Memory layout structures
   - Buffer organization
   - Component placement

6. **Protocol Info** (`info.rs`):
   - Circuit metadata
   - Field parameters
   - Protocol constants

**Language Syntax**: `RustLanguageSyntax`
- Rust-specific operators and syntax
- Type emission: `Val::new(x)`, `ExtVal::new([a, b, c, d])`
- Memory access patterns

**Code Generation Options**:
```cpp
CodegenOptions getRustCodegenOpts() {
  static RustLanguageSyntax kRust;
  CodegenOptions opts(&kRust);
  addCommonSyntax(opts);
  addRustSyntax(opts);
  ZStruct::addRustSyntax(opts);
  Zhlt::addRustSyntax(opts);
  return opts;
}
```

### C++ Code Generation

**Location**: `zirgen/compiler/codegen/gen_cpp.cpp`

**Key Class**: `CppStreamEmitter`

**Generated Files**:
1. **Polynomial Evaluators**:
   - Constraint checking functions
   - Field arithmetic operations

2. **Tap Definitions** (`taps.cpp`):
   - C++ tap structures
   - Register access logic

3. **Layout Definitions** (`layout.cpp.inc`):
   - C++ layout structures
   - Memory organization

**Language Syntax**: `CppLanguageSyntax`
- C++-specific operators and syntax
- Type emission: `Val(x)`, `ExtVal(a, b, c, d)`
- Operator precedence handling

**Code Generation Options**:
```cpp
CodegenOptions getCppCodegenOpts() {
  static CppLanguageSyntax kCpp;
  CodegenOptions opts(&kCpp);
  addCommonSyntax(opts);
  addCppSyntax(opts);
  ZStruct::addCppSyntax(opts);
  Zhlt::addCppSyntax(opts);
  return opts;
}
```

### GPU Code Generation (CUDA/Metal)

**Location**: `zirgen/compiler/codegen/gen_gpu.cpp`

**Key Class**: `GpuStreamEmitter`

**Template-Based Generation**: Uses **Mustache templates** for flexible emission

**Template Files** (located in `zirgen/compiler/codegen/gpu/`):

**CUDA Templates**:
- `step.tmpl.cu.h` - Step function template
- `eval_check.tmpl.cu.h` - Polynomial evaluation template
- `recursion/step.tmpl.cu` - Recursion step
- `recursion/step_compute_accum.tmpl.cu` - Accumulator computation
- `recursion/step_verify_accum.tmpl.cu` - Accumulator verification
- `recursion/eval_check.tmpl.cu` - Recursion polynomial eval

**Metal Templates**:
- `step.tmpl.metal.h` - Step function template
- `eval_check.tmpl.metal.h` - Polynomial evaluation template
- `recursion/step_compute_accum.tmpl.metal` - Accumulator computation
- `recursion/step_verify_accum.tmpl.metal` - Accumulator verification

**Generated Files**:
1. **Step Kernels** (`step_{stage}.cu` / `.metal`):
   - GPU kernels for each execution stage
   - Massively parallel step execution
   - Optimized memory access patterns

2. **Polynomial Evaluation Kernels** (`eval_check.cu` / `.metal`):
   - GPU polynomial constraint checking
   - Parallel evaluation across witnesses
   - SIMD optimizations

3. **Layout Definitions** (`layout.cu.inc`):
   - GPU memory layout structures
   - Buffer organization for parallel access

**Language Syntax**: `CudaLanguageSyntax`
- CUDA/Metal-specific syntax
- Thread indexing (`threadIdx`, `blockIdx`)
- Shared memory declarations
- Barrier synchronization

**Code Generation Options**:
```cpp
CodegenOptions getCudaCodegenOpts() {
  static CudaLanguageSyntax kCuda;
  CodegenOptions opts(&kCuda);
  addCommonSyntax(opts);
  addCppSyntax(opts);  // CUDA uses C++ base syntax
  ZStruct::addCppSyntax(opts);
  Zhlt::addCppSyntax(opts);
  return opts;
}
```

**GPU Optimizations**:
- Coalesced memory access
- Shared memory for common values
- Thread-level parallelism
- Warp-level optimizations

## Code Generation Interfaces

### Operation Interfaces

**CodegenExprOpInterface** - Expression-level code emission
```cpp
// Emits code for expression operations (e.g., zll.add, zll.mul)
void emitExpr(CodegenEmitter& cg);
```

Operations implementing this interface:
- Arithmetic: `zll.add`, `zll.sub`, `zll.mul`, `zll.inv`
- Memory: `zll.get`, `zll.get_global`, `zll.const`
- Selection: `zll.select`
- Crypto: `zll.hash`, `zll.into_digest`
- Struct: `zstruct.lookup`, `zstruct.subscript`, `zstruct.load`

**CodegenStatementOpInterface** - Statement-level code emission
```cpp
// Emits code for statement operations (e.g., zll.if, zll.nondet)
void emitStatement(CodegenEmitter& cg);
```

Operations implementing this interface:
- Control flow: `zll.if`, `zll.nondet`
- Memory writes: `zll.set`, `zll.set_global`, `zstruct.store`
- Constraints: `zll.eqz`, `zll.and_eqz`

### Type Interfaces

**CodegenTypeInterface** - Type emission
```cpp
// Emit type name (e.g., "Val", "ExtVal", "Digest")
void emitTypeName(CodegenEmitter& cg);

// Emit type definition (struct/class definition)
void emitTypeDefinition(CodegenEmitter& cg);

// Emit literal value of this type
void emitLiteral(CodegenEmitter& cg, Attribute value);
```

Types implementing this interface:
- `Val` - Field elements
- `Digest` - Hash digests
- `StructType` - Component structures
- `ArrayType` - Fixed-size arrays
- `LayoutType` - Memory layouts

## CodegenEmitter Class

The `CodegenEmitter` class provides the primary interface for code emission:

**Key Methods**:
```cpp
// Output operators
CodegenEmitter& operator<<(StringRef str);
CodegenEmitter& operator<<(uint64_t val);

// Indentation control
void indent();
void outdent();
void emitIndent();

// String utilities
void emitEscapedString(StringAttr str);

// Comma-separated lists
template<typename Range, typename Func>
void interleaveComma(Range range, Func func);

// Emit operation as expression
void emitExpr(Operation* op);

// Emit operation as statement
void emitStatement(Operation* op);
```

**Language-Specific Emitters**:
- `RustStreamEmitter` - Rust-specific emission
- `CppStreamEmitter` - C++ specific emission
- `GpuStreamEmitter` - GPU (CUDA/Metal) emission

## Language Syntax Classes

Each backend has a language syntax class defining operators and conventions:

### RustLanguageSyntax
```cpp
class RustLanguageSyntax : public LanguageSyntax {
  // Binary operators: +, -, *, /, %, &, |, ^, <<, >>
  // Unary operators: -, !
  // Function calls: func(args)
  // Array subscript: array[index]
  // Field access: struct.field
};
```

### CppLanguageSyntax
```cpp
class CppLanguageSyntax : public LanguageSyntax {
  // Similar to Rust but with C++ conventions
  // Pointer access: ptr->field
  // Reference: &var
};
```

### CudaLanguageSyntax
```cpp
class CudaLanguageSyntax : public LanguageSyntax {
  // CUDA-specific: __device__, __global__, __shared__
  // Thread indexing: threadIdx.x, blockIdx.x
  // Synchronization: __syncthreads()
};
```

## Template System (GPU)

GPU code generation uses **Mustache** templating:

**Template Variables**:
- `{{function_body}}` - Generated function body
- `{{buffer_declarations}}` - Buffer declarations
- `{{field_params}}` - Field parameters
- `{{thread_index}}` - Thread indexing code

**Example Template** (simplified):
```cuda
__global__ void step_exec(
  {{buffer_declarations}}
) {
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  {{function_body}}
}
```

## Balanced Splitting

Large functions are split using the `BalancedSplit` pass for manageability:

**Default**: 1000 operations per function
**Configurable**: Via pass options

**Benefits**:
- Manageable compilation times
- Better optimization opportunities
- Avoids compiler limits

**Location**: `zirgen/Dialect/Zll/Transforms/BalancedSplit.cpp`

## Protocol Info Constants

**Location**: `zirgen/compiler/codegen/protocol_info_const.h`

Emits circuit metadata:
- Field type (BabyBear, Goldilocks)
- Extension degree (2, 4)
- Circuit parameters
- Buffer sizes

## File Emission

**FileEmitter Class** (in `codegen.cpp`):

Centralizes all output file management:
```cpp
class FileEmitter {
  void emitRustStep(stage, module);
  void emitGpuStep(stage, module);
  void emitPolyFunc(module, split_index);
  void emitLayout(module, syntax);
  void emitTaps(module, syntax);
  void emitInfo(module);
};
```

## Entry Points

### zirgen-translate Tool
**Location**: `zirgen/compiler/tools/zirgen-translate.cpp`

Registers translation passes:
- `zirgen-to-rust-step` - Rust step function generation
- `rust-codegen` - Full Rust code generation
- `cpp-codegen` - C++ code generation
- `zirgen-to-rust-poly-*` - Polynomial operations
- `zirgen-to-rust-taps` - Tap definitions

**Usage**:
```bash
zirgen-translate --rust-codegen input.mlir -o output_dir/
zirgen-translate --cpp-codegen input.mlir -o output_dir/
```

## Optimization Strategies

### Cross-Backend
1. **CSE (Common Subexpression Elimination)**: Reduces redundant computations
2. **Canonicalization**: Simplifies expressions
3. **Constant Folding**: Evaluates constants at compile time
4. **Dead Code Elimination**: Removes unused operations

### Rust-Specific
1. **Inline Hints**: Critical operations marked for inlining
2. **Zero-Copy**: Pass-by-reference for large structures
3. **Type Inference**: Leverage Rust's type system

### C++-Specific
1. **Template Specialization**: Field-specific templates
2. **Inline Functions**: Small operations inlined
3. **Const Correctness**: Immutable where possible

### GPU-Specific
1. **Memory Coalescing**: Aligned memory access patterns
2. **Shared Memory**: Cache common values per thread block
3. **Register Pressure**: Minimize local variables
4. **Occupancy**: Optimize thread/block configuration

## Testing

Code generation is tested via:
1. **Unit Tests**: Individual operation emission
2. **Integration Tests**: Full pipeline tests
3. **Golden Tests**: Compare against known-good output
4. **Execution Tests**: Run generated code and verify results

## Performance Characteristics

### Compilation Time
- Rust: ~seconds for small circuits, minutes for large
- C++: ~seconds for polynomial evaluators
- GPU: ~seconds (template-based is fast)

### Runtime Performance
- Rust: Excellent for CPU execution
- C++: Comparable to Rust, slightly faster for tight loops
- GPU: 10-100x speedup for large polynomial evaluations

### Code Size
- Rust: Moderate (split functions help)
- C++: Small (focused on polynomials)
- GPU: Small (template-generated kernels)

## Debugging Support

### Emitted Code Readability
- Human-readable variable names (where possible)
- Comments indicating source operations
- Indentation and formatting

### Debug Builds
- Optional debug information emission
- Line number tracking
- Assertion code generation

### Verification
- Type checking in generated code
- Runtime assertions (debug builds)
- Constraint verification code

## Future Directions

Potential improvements:
1. **Additional Backends**: WebAssembly, FPGA HDL
2. **Better GPU Utilization**: Multi-GPU support
3. **Incremental Compilation**: Only recompile changed functions
4. **Link-Time Optimization**: Cross-function optimization
5. **Profile-Guided Optimization**: Use runtime profiles to guide codegen

## See Also

- [Compiler Architecture](COMPILER_ARCHITECTURE.md) - Overall pipeline
- [Zll Dialect](../zirgen/Dialect/Zll/README.md) - Low-level IR being emitted
- [ZStruct Dialect](../zirgen/Dialect/ZStruct/README.md) - Structured types in codegen
- [BalancedSplit Pass](../zirgen/Dialect/Zll/Transforms/BalancedSplit.cpp) - Function splitting
