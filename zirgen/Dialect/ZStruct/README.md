# ZStruct Dialect

The **ZStruct (Zirgen Structured Types)** dialect provides structured type operations for representing components, layouts, and arrays in the Zirgen compiler. It serves as an intermediate layer between the high-level typed ZHLT dialect and the low-level Zll dialect, focusing on structured data access and memory layout management.

## Purpose

ZStruct provides:

- **Structured types**: Structs, unions, and arrays for component representation
- **Layout management**: Explicit layout types describing register organization
- **Memory access operations**: Load/store with layout-aware addressing
- **Reference types**: Safe references to struct members
- **Type-preserving transformations**: Optimizations on structured data before lowering to Zll

## Position in Compilation Pipeline

```
ZHLT Dialect (typed high-level IR)
    ↓
[Lower to ZStruct]
    ↓
ZStruct Dialect ← YOU ARE HERE
    ↓
[ZStruct Transformation Passes]
    ↓
[LowerComposites: ZStruct → Zll]
    ↓
Zll Dialect (low-level IR)
```

## Key Operations

Defined in `zirgen/Dialect/ZStruct/IR/Ops.td`:

### Member Access
- `zstruct.lookup` - Look up reference to struct or union member
  - Syntax: `%ref = zstruct.lookup %base["member_name"]`
  - Returns reference or value depending on context

- `zstruct.subscript` - Select array element by index
  - Syntax: `%elem = zstruct.subscript %array[%index]`
  - Index can be `index` type or `!zll.val`

### Memory Operations
- `zstruct.load` - Load value from a reference
  - Syntax: `%val = zstruct.load %ref back %distance`
  - Supports extension field coalescing

- `zstruct.store` - Store value into a reference
  - Syntax: `zstruct.store %ref, %val`
  - Updates memory at referenced location

### Construction
- `zstruct.pack` - Pack operands into struct members
  - Syntax: `%struct = zstruct.pack (%m1, %m2, ...) : !zstruct.struct<...>`
  - Creates struct from individual values

- `zstruct.array` - Construct array from elements
  - Syntax: `%arr = zstruct.array [%e1, %e2, ...] : !zstruct.array<...>`

Additional operations likely include:
- Union construction and access
- Global buffer operations
- Reference manipulation

## Type System

Defined in `zirgen/Dialect/ZStruct/IR/Types.td`:

### Core Types

**StructType** - Component representation
- Parameters: ID string, field list
- Represents a circuit component with named fields
- Implements `CodegenTypeInterface` for code generation
- Syntax: `!zstruct.struct<"ComponentName", {field1: !zll.val, ...}>`

**LayoutType** - Register layout for components
- Parameters: ID string, field list, layout kind (Normal/Mux)
- Describes where component data is stored in registers
- Mux layouts share common "@super" registers across fields
- Syntax: `!zstruct.layout<"ComponentName", {...}>`

**UnionType** - Mux representation
- Parameters: ID string, field list (union arms)
- Represents conditional/multiplexed components
- Syntax: `!zstruct.union<"MuxName", {arm1: ..., arm2: ...}>`

**ArrayType** - Homogeneous arrays
- Parameters: Element type, size
- Fixed-size arrays of same-typed values
- Syntax: `!zstruct.array<!zll.val<BabyBear>, 16>`

**LayoutArrayType** - Arrays of layouts
- Parameters: Element type, size
- Arrays of layout references
- Pass-by-reference semantics
- Syntax: `!zstruct.layout_array<!zstruct.layout<...>, 8>`

**Ref** - Reference to struct member
- Provides safe, typed references for load/store operations
- Used internally for member access

## Key Transformation Passes

Located in `zirgen/Dialect/ZStruct/Transforms/`:

### Layout Optimization
1. **OptimizeLayout** - Reorder structure members for constraint compatibility
   - Optimizes field ordering for efficient constraint checking
   - Improves memory access patterns

2. **ExpandLayout** - Expand global layout constants
   - Each layout constant gets data for single component only
   - Simplifies layout management

3. **InlineLayout** - Inline offsets from layouts
   - Converts `zstruct.load`/`zstruct.store` to `zll.{get, get_global, set, set_global}`
   - Eliminates layout abstraction when possible

### Code Transformation
4. **Unroll** - Unroll zhlt.map and zhlt.reduce
   - Removes loop constructs by unrolling
   - Converts functional operations to explicit sequences

5. **BuffersToArgs** - Convert buffer references to function arguments
   - Replaces `zstruct.get_buffer` with function parameters
   - Arguments added in BufferAnalysis order
   - Propagated through call chains

## Interfaces

### Type Interfaces
- **CodegenTypeInterface** - Emit type definitions, names, and literals
- **ArrayLikeTypeInterface** - Uniform handling of array types
- **CodegenLayoutType** - Mark types as layouts (pass-by-reference)
- **CodegenNeedsCloneType** - Mark types needing clone operations

### Operation Interfaces
- **CodegenExprOpInterface** - Emit expression-level code
- **CodegenAlwaysInlineOp** - Operations always inlined in codegen
- **PolyOp** - Operations contributing to polynomial constraints
- **EvalOpAdaptor** - Evaluation in interpreter

## Layout Kinds

Defined in `zirgen/Dialect/ZStruct/IR/Enums.td`:

- **Normal** - Standard layout with independent field storage
- **Mux** - Multiplexed layout where arms share common registers
  - Hint to layout generator for shared "@super" registers
  - Optimizes storage for conditional components

## Example

```mlir
// Struct type definition
!zstruct.struct<"Point", {x: !zll.val<BabyBear>, y: !zll.val<BabyBear>}>

// Member lookup
%x_ref = zstruct.lookup %point["x"] : !zstruct.struct<...> -> !zstruct.ref<...>

// Load with back reference
%x_val = zstruct.load %x_ref back 0 : !zll.val<BabyBear>

// Array subscript
%elem = zstruct.subscript %array[%idx] : !zstruct.array<...> -> !zll.val<BabyBear>

// Pack into struct
%point = zstruct.pack (%x, %y) : !zstruct.struct<"Point", ...>

// Store to reference
zstruct.store %x_ref, %new_val : !zll.val<BabyBear> -> !zstruct.ref<...>
```

## Design Philosophy

1. **Structured Abstraction**: Maintains component structure before flattening to Zll
2. **Layout-Aware**: Explicit layout types enable layout optimization
3. **Safe References**: Reference types provide type-safe member access
4. **Progressive Lowering**: Gradually eliminates structure during transformation
5. **Codegen-Ready**: Types implement codegen interfaces for multi-target emission

## Lowering to Zll

The `LowerComposites` pass (`zirgen/Dialect/Zll/Conversion/ZStructToZll/`) converts:
- `zstruct.load` → `zll.get` or `zll.get_global`
- `zstruct.store` → `zll.set` or `zll.set_global`
- `zstruct.lookup` → direct field access or indexing
- Struct/array types → buffer types
- Layout abstractions → concrete buffer offsets

## File Organization

```
zirgen/Dialect/ZStruct/
├── IR/
│   ├── Dialect.td      - Dialect definition (name: "zstruct")
│   ├── Ops.td          - Operation definitions
│   ├── Types.td        - Type system (Struct, Layout, Union, Array, Ref)
│   ├── Attrs.td        - Attributes (FieldInfo, etc.)
│   ├── Enums.td        - Enumerations (LayoutKind)
│   ├── Interfaces.td   - Type/op interfaces
│   └── BUILD.bazel     - Build configuration
└── Transforms/
    ├── Passes.td       - Pass definitions
    ├── OptimizeLayout.cpp
    ├── Unroll.cpp
    ├── ExpandLayout.cpp
    ├── InlineLayout.cpp
    └── BuffersToArgs.cpp
```

## Relationship to Other Dialects

### From ZHLT
- **Input**: Typed components with high-level operations
- **Transformation**: Add explicit layout and structure operations
- **Output**: ZStruct with structured types and layout awareness

### To Zll
- **Input**: Structured operations and types
- **Transformation**: LowerComposites flattens structures
- **Output**: Low-level buffer operations in Zll

### Uses Zll Types
ZStruct operations use Zll scalar types (`!zll.val`) as field types, creating a unified scalar type system.

## See Also

- [ZHLT Dialect](../ZHLT/README.md) - Typed high-level predecessor
- [Zll Dialect](../Zll/README.md) - Low-level successor
- [Compiler Architecture](../../docs/COMPILER_ARCHITECTURE.md) - Overall pipeline
- [Code Generation](../../docs/CODEGEN.md) - Multi-target emission
