# ZHLT Dialect

The **ZHLT (Zirgen High-Level Typed)** dialect is the typed intermediate representation that results from type-checking the untyped ZHL dialect. It represents circuit components with full type information, enabling type-aware transformations and optimizations before lowering to ZStruct and eventually Zll.

## Purpose

ZHLT serves as the typed high-level IR in the Zirgen compiler, providing:

- **Full type information**: Unlike ZHL (which uses only `Expr`), ZHLT includes specific component types, field types (`Val`), and layout types
- **Type-aware transformations**: Enables optimizations that require type information
- **Layout management**: Introduces layout types and operations for memory organization
- **Step function generation**: Produces callable step functions for circuit execution
- **Bridge to structured IR**: Prepares for lowering to ZStruct dialect

## Position in Compilation Pipeline

```
ZHL Dialect (untyped IR)
    ↓
[Type Checking via ComponentManager]
    ↓
ZHLT Dialect ← YOU ARE HERE
    ↓
[ZHLT Transformation Passes]
    ↓
ZStruct Dialect (structured types)
    ↓
Zll Dialect (low-level)
```

Type checking happens in `zirgen/Conversions/Typing/ComponentManager.h` (invoked at `gen_zirgen.cpp:187`).

## Key Operations

ZHLT defines operations in `zirgen/Dialect/ZHLT/IR/Ops.td` and `ComponentOps.td`:

### Core Operations
- `zhlt.return` - Terminator marking component constructor result
- `zhlt.get_global_layout` - Returns the layout for a global buffer component
- `zhlt.back` - Retrieve value from previous cycle (with distance and layout)
- `zhlt.directive` - Compiler directive with downstream effects
- `zhlt.magic` - Construct magic value for error recovery (invalid, but allows continued compilation)

### Component Operations
ZHLT likely includes operations for:
- Component construction with typed parameters
- Member access with type information
- Array operations with element types
- Constraint checking with typed operands

## Type System

Unlike ZHL's single `Expr` type, ZHLT uses the full ZStruct type system:

### Core Types
- **Val** (`!zll.val`) - Field element values (from Zll dialect)
- **LayoutType** - Memory layout information for components
- **StructType** - Structured component types
- **ArrayType** - Typed arrays with element types
- **Constraint** - Constraint expressions

ZHLT reuses types from ZStruct and Zll dialects, providing a unified type system across the typed portion of the compiler.

## Key Transformation Passes

Located in `zirgen/Dialect/ZHLT/Transforms/`:

### Optimization Passes
1. **ElideRedundantMembers** - Prune struct members that are equal by construction
2. **HoistAllocs** - Hoist alloc-only params to callers, then merge (optimizes NondetReg calls)
3. **HoistCommonMuxCode** - Hoist code shared across all mux arms out of the mux
4. **OptimizeParWitgen** - Optimize for parallel witness generation (flattens, unrolls, inlines)

### Code Generation Passes
5. **GenerateSteps** - Generate top-level StepOps for all externally callable functions
   - `step$top`: Constructs "Top" component and checks constraints
   - `step$test$*`: StepOps to run each test
6. **StripTests** - Remove all tests for smaller generated code
7. **StripAliasLayoutOps** - Erase all AliasLayoutOps outside CheckLayoutFuncOps
8. **LowerDirectives** - Translate directives to lower-level code
9. **LowerStepFuncs** - Convert all functions to StepFuncs

### Analysis Passes
10. **AnalyzeBuffers** - Analyze buffers needed for circuit, saves BuffersAttr on module

### Advanced Optimization
11. **OutlineIfs** - Move bodies of `if` statements into separate function calls
    - Outlines zll.if operations inside zhlt.step_funcs
    - Captures or reconstructs values (prefers reconstruction from zll.get ops)

## Key Features

### Layout Management
ZHLT introduces layout operations and types:
- `zhlt.get_global_layout` retrieves layout information for global components
- Layout types describe memory organization
- AliasLayoutOps indicate layout constraints during lowering

### Back References
The `zhlt.back` operation provides typed access to previous cycle values:
- Specifies function name, cycle distance, and optional layout
- Returns typed values (not generic Expr)
- Critical for circuit state management

### Step Functions
ZHLT generates step functions that:
- Provide callable entry points for circuit execution
- Encapsulate component construction and constraint checking
- Enable modular circuit execution and testing

### Directive Processing
Directives provide compiler hints:
- `assume-range` and other directives guide optimization
- Lowered by LowerDirectives pass to concrete code
- Enable range analysis and other optimizations

## Example

```mlir
// Back reference with type and layout
%prev = zhlt.back @some_func(3, %layout : !zstruct.layout) -> !zll.val<BabyBear>

// Global layout access
%layout = zhlt.get_global_layout "global"["MyComponent"] : !zstruct.layout

// Directive
zhlt.directive "assume_range"(%value : !zll.val<BabyBear>)

// Return from component constructor
zhlt.return %result : !zstruct.struct<...>
```

## Compilation Flow

### From ZHL to ZHLT
1. **Parse ZHL** - Load untyped ZHL module
2. **Type Check** - ComponentManager resolves types:
   - `!zhl.expr` → specific types
   - Component types resolved
   - Field types determined
   - Layout information computed
3. **Emit ZHLT** - Fully typed ZHLT module

### ZHLT Pass Pipeline
From `gen_zirgen.cpp` (lines 199-227), the typical ZHLT pass pipeline:
1. Accumulation and global variable handling
2. ElideRedundantMembers
3. Field DCE and CSE passes
4. Type inference and checking
5. GenerateSteps (if needed)
6. InlinePure operations
7. HoistInvariants
8. LowerStepFuncs

### From ZHLT to ZStruct
After ZHLT passes, the IR is lowered to ZStruct dialect for further structural transformations.

## Design Philosophy

1. **Type Safety**: Full type information enables verification and optimization
2. **Gradual Lowering**: ZHLT maintains high-level structure while adding types
3. **Layout Awareness**: Explicit layout operations prepare for code generation
4. **Modular Steps**: Step functions provide clear execution boundaries
5. **Optimization-Friendly**: Type information enables aggressive optimization

## File Organization

```
zirgen/Dialect/ZHLT/
├── IR/
│   ├── Dialect.td           - Dialect definition (name: "zhlt")
│   ├── Ops.td               - Core operations
│   ├── ComponentOps.td      - Component-specific operations
│   ├── Types.td             - Type definitions (imports from ZStruct/Zll)
│   ├── Attrs.td             - Attributes
│   ├── Interfaces.td        - Operation/type interfaces
│   ├── NamedVariadic.td     - Named variadic parameter support
│   └── BUILD.bazel          - Build configuration
└── Transforms/
    ├── Passes.td            - Pass definitions
    ├── ElideRedundantMembers.cpp
    ├── HoistAllocs.cpp
    ├── HoistCommonMuxCode.cpp
    ├── StripTests.cpp
    ├── GenerateSteps.cpp
    ├── StripAliasLayoutOps.cpp
    ├── LowerDirectives.cpp
    ├── LowerStepFuncs.cpp
    ├── AnalyzeBuffers.cpp
    ├── OptimizeParWitgen.cpp
    └── OutlineIfs.cpp
```

## Relationship to Other Dialects

### From ZHL
- **Input**: Untyped ZHL with all `!zhl.expr` values
- **Transformation**: Type checking resolves all types
- **Output**: Typed ZHLT with specific types

### To ZStruct
- **Input**: ZHLT with typed components and layouts
- **Transformation**: Lower to structured types
- **Output**: ZStruct with struct/array operations

### Uses Zll Types
ZHLT operations directly use Zll types like `!zll.val<BabyBear>` for field values, creating a unified type system across compilation stages.

## See Also

- [ZHL Dialect](../ZHL/README.md) - Untyped predecessor
- [ZStruct Dialect](../ZStruct/README.md) - Structured type successor
- [Zll Dialect](../Zll/README.md) - Low-level circuit IR
- [Compiler Architecture](../../docs/COMPILER_ARCHITECTURE.md) - Overall pipeline
- [Type Checking](../../Conversions/Typing/README.md) - ZHL→ZHLT transformation (if docs exist)
