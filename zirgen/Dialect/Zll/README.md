# Zll Dialect

The **Zll (Zirgen Low-Level)** dialect is the core low-level intermediate representation for zero-knowledge proof circuit compilation in the Zirgen compiler. It represents circuits at a level suitable for constraint generation, polynomial evaluation, and code generation to multiple backends (Rust, C++, GPU).

## Purpose

Zll sits between higher-level dialects (ZStruct, ZHLT) and final code generation, serving as the central optimization and transformation layer. It provides:

- Field arithmetic operations over finite fields (BabyBear, Goldilocks)
- Constraint and polynomial expression representation
- Buffer and register management with tap-based access
- Cryptographic operations (hashing, digests)
- Control flow for conditional execution and non-deterministic witness generation
- Multi-target code generation support

## Position in Compilation Pipeline

```
ZStruct/ZHLT Dialects (high-level typed IR)
    ↓
[LowerComposites Pass]
    ↓
Zll Dialect ← YOU ARE HERE
    ↓
[Optimization Passes: compute-taps, make-polynomial, etc.]
    ↓
[Code Generation: Rust/C++/GPU]
```

## Key Operations

### Arithmetic Operations
- `zll.const` - Field element constant
- `zll.add`, `zll.sub`, `zll.mul` - Basic field arithmetic
- `zll.inv` - Multiplicative inverse
- `zll.neg` - Negation
- `zll.pow` - Constant exponentiation
- `zll.isz` - Is-zero check
- `zll.mod`, `zll.bit_and` - Modulo and bitwise operations

### Constraint Operations
- `zll.eqz` - Assert value equals zero (primary constraint)
- `zll.true` - Constraint representing truth
- `zll.and_eqz` - AND constraint with equality-to-zero
- `zll.and_cond` - AND constraint with condition

### Memory Operations
- `zll.get` - Read from buffer with cycle offset and optional tap
- `zll.set` - Write to buffer
- `zll.get_global` - Read from global/temporary buffers
- `zll.set_global` - Write to global/temporary buffers
- `zll.back` - Access previous cycle values
- `zll.slice` - Extract buffer slice
- `zll.temp_buffer` - Create temporary buffers

### Control Flow
- `zll.if` - Conditional execution region
- `zll.nondet` - Non-deterministic region (for witness generation)
- `zll.barrier` - Stage barrier (separates exec/verify stages)
- `zll.terminate` - Region terminator

### Cryptographic Operations
- `zll.hash` - Hash field elements
- `zll.hash_fold` - Combine two digests
- `zll.hash_assert_eq` - Verify digest equality
- `zll.into_digest` - Convert field elements to digest
- `zll.from_digest` - Convert digest to field elements
- `zll.tagged_struct` - Hash with proper padding
- `zll.hash_checked_bytes`, `zll.hash_checked_bytes_public` - Load, range-check, and hash bytes

### Utility Operations
- `zll.select` - Index-based selection
- `zll.extern` - Call external code
- `zll.variadic_pack` - Pack variadic parameters
- `zll.normalize` - Reduction to normal form (for extension fields)

## Type System

### Core Types

**Val** - Single field element
- Supports both base field (BabyBear: p=2^27*15+1, Goldilocks: p=2^64-2^32+1)
- Supports extension fields (degree 2 or 4)
- Syntax: `!zll.val<BabyBear>`, `!zll.val<Goldilocks, 4>`

**Buffer** - Array of field elements
- Parameters: element type, size, kind (Constant/Mutable/Global/Temporary)
- Used for register storage and temporary computations
- Syntax: `!zll.buffer<4x!zll.val<BabyBear>, Mutable>`

**Constraint** - Represents a constraint polynomial
- Used for constraint tracking during polynomial mix generation
- Combines with MixState to form final constraint polynomial

**Digest** - Cryptographic digest
- Kinds: Default (Poseidon2), SHA256, Poseidon2
- Syntax: `!zll.digest`, `!zll.digest<Sha256>`

**String** - String literals for code generation

**VariadicPack** - Packs variadic parameters for external calls

## Supported Fields

- **BabyBear**: Prime 2^27 * 15 + 1, extension degree 4
- **Goldilocks**: Prime 2^64 - 2^32 + 1, extension degree 2

## Key Transformation Passes

Located in `zirgen/Dialect/Zll/Transforms/`:

1. **ComputeTaps** - Identify and number tap registers for memory access
2. **MakePolynomial** - Combine constraints into polynomial mix
3. **SplitStage** - Extract specific circuit stage (exec, verify_mem, verify_bytes, etc.)
4. **AddReductions** - Add reduction operations to keep values in field range
5. **IfToMultiply** / **MultiplyToIf** - Transform conditionals
6. **DropConstraints** - Remove constraint checking for execution
7. **InlineFpExt** - Inline field extension operations
8. **BalancedSplit** - Split large operation blocks for manageability
9. **SortForReproducibility** - Ensure deterministic ordering

## Interfaces

### Operation Interfaces
- **EvalOp** - Evaluation in interpreter for simulation/verification
- **PolyOp** - Marks operations contributing to polynomial constraints
- **ReduceOpInterface** - Tracks value ranges for overflow detection
- **CodegenExprOpInterface** - Emits expression-level code
- **CodegenStatementOpInterface** - Emits statement-level code

### Type Interfaces
- **CodegenTypeInterface** - Custom type name/definition emission

## Analysis Infrastructure

Located in `zirgen/Dialect/Zll/Analysis/`:

- **DegreeAnalysis** - Tracks polynomial degree of constraints
- **TapsAnalysis** - Analyzes register accesses and cycle offsets
- **MixPowerAnalysis** - Analyzes constraint polynomial mixing

## Interpreter Support

The Zll dialect includes a full interpreter (`Interpreter.h/cpp`) for:
- Circuit simulation and verification
- Witness generation
- Debugging and testing
- Both base field and extension field arithmetic

## Code Generation

Zll operations translate directly to:
- **Rust**: Via template-based emission (see `zirgen/compiler/codegen/gen_rust.cpp`)
- **C++**: Direct emission of polynomial evaluators
- **GPU (CUDA/Metal)**: Via Mustache templates (see `zirgen/compiler/codegen/gpu/`)

Each codegen-capable operation implements `CodegenExprOpInterface` or `CodegenStatementOpInterface` to emit appropriate target code.

## Usage Example

```mlir
// Field arithmetic
%0 = zll.const 42 : !zll.val<BabyBear>
%1 = zll.const 7 : !zll.val<BabyBear>
%2 = zll.add %0, %1 : !zll.val<BabyBear>

// Memory access with tap
%3 = zll.get %buffer[%0] back 1 tap 2 : !zll.val<BabyBear>

// Constraint
%4 = zll.sub %2, %3 : !zll.val<BabyBear>
zll.eqz %4 : !zll.val<BabyBear>

// Conditional
zll.if %condition : !zll.val<BabyBear> {
  // ...
  zll.terminate
}
```

## Module-Level Attributes

Zll modules carry important metadata:
- `zll.taps` - Tap register definitions
- `zll.buffers` - Buffer layout information
- `zll.protocol_info` - Protocol metadata (field, degree, etc.)
- `zll.steps` - Circuit step definitions
- `zll.circuit_name` - Circuit identifier

## Design Philosophy

1. **Multi-field Support**: All operations work with configurable fields
2. **Buffer-centric Memory**: Clear separation of buffer kinds (mutable, constant, global, temporary)
3. **Constraint Tracking**: Constraints are first-class operations
4. **Tap-based Access**: Register access uses "tap" attributes for cycle offsets
5. **Dual Path**: Supports both interpretation and code generation
6. **Range Safety**: BigInt range tracking ensures values stay within field bounds

## File Organization

```
zirgen/Dialect/Zll/
├── IR/
│   ├── Dialect.td, Ops.td, Types.td, Attrs.td, Enums.td, Interfaces.td
│   ├── Field.h/cpp - Field arithmetic implementation
│   ├── BigInt.h - Big integer range tracking
│   ├── Interpreter.h/cpp - Circuit interpreter
│   ├── Codegen.h/cpp - Code generation support
│   └── test/ - MLIR tests
├── Transforms/ - Optimization passes
├── Conversion/ZStructToZll/ - Lowering from ZStruct
└── Analysis/ - Analysis passes

```

## See Also

- [ZHLT Dialect](../ZHLT/README.md) - Typed high-level dialect
- [ZStruct Dialect](../ZStruct/README.md) - Structured types
- [Compiler Architecture](../../docs/COMPILER_ARCHITECTURE.md) - Overall pipeline
- [Code Generation](../../docs/CODEGEN.md) - Multi-target emission
