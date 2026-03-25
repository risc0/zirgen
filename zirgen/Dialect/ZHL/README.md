# ZHL Dialect

The **ZHL (Zirgen High-Level)** dialect is the initial untyped intermediate representation produced when lowering the Zirgen circuit DSL (.zir files) to MLIR. It serves as a bridge between the parsed Abstract Syntax Tree (AST) and the type-checked ZHLT dialect.

## Purpose

ZHL is the first IR after parsing the Zirgen DSL, providing:

- **Untyped representation**: All values are generic `Expr` types without specific field or component type information
- **DSL concept preservation**: Operations directly correspond to DSL constructs (components, arrays, lookups, constraints)
- **Bridge to typed IR**: Cleanly separates parsing from type checking
- **Modular architecture**: Enables independent AST lowering and type checking passes

## Position in Compilation Pipeline

```
Input (.zir file)
    ↓
DSL Parser
    ↓
Abstract Syntax Tree (AST)
    ↓
Lower to ZHL Dialect ← YOU ARE HERE
    ↓
Type Checking
    ↓
ZHLT Dialect (typed IR)
    ↓
Further transformations...
```

The lowering happens in `zirgen/dsl/lower.cpp` (invoked from `gen_zirgen.cpp:182`), and type checking converts ZHL to ZHLT via `ComponentManager.h` (invoked at `gen_zirgen.cpp:187`).

## Key Operations

ZHL defines 23 operations in `zirgen/Dialect/ZHL/IR/Ops.td`:

### Literals & Constants
- `zhl.literal` - Numeric constant (64-bit unsigned integer)
- `zhl.string` - String constant
- `zhl.global` - Reference to implicit global symbol

### Component & Type Operations
- `zhl.generic` (`TypeParamOp`) - Generic type template parameter value
- `zhl.parameter` (`ConstructorParamOp`) - Constructor function parameter value
- `zhl.component` - Component declaration with body region
- `zhl.extern` - Execute function external to the DSL

### Access Operations
- `zhl.lookup` - Access component member (like `.member` in DSL)
- `zhl.subscript` - Access array element (like `[index]` in DSL)

### Transformation Operations
- `zhl.specialize` - Apply type parameters (like `Component<T>`)
- `zhl.construct` - Construct a component instance
- `zhl.construct_global` - Construct a global component
- `zhl.get_global` - Returns the value of a global component

### Collection Operations
- `zhl.array` - Create array from list of values `[elem1, elem2, ...]`
- `zhl.range` - Create array spanning a range
- `zhl.map` - Apply function to array elements
- `zhl.reduce` - Reduce array elements with function

### Control Flow
- `zhl.block` - Block scope with statements (single region)
- `zhl.switch` - Evaluate expression array based on selector value

### Declaration & Definition
- `zhl.declare` - Declare member (can be public/private)
- `zhl.define` - Define/assign member value

### Constraint & Directive
- `zhl.constrain` - Declare constraint (LHS = RHS)
- `zhl.directive` - Compiler directive with name and arguments

### Special Operations
- `zhl.back` - Retrieve value from previous iteration (with distance and target)
- `zhl.super` - Super terminator (marks super constructor calls)

All expression-producing operations implement the `ReturnsExpr` trait using `InferTypeOpInterface`.

## Type System

ZHL has a minimal type system to maintain its untyped nature:

### Single Core Type

**Expr** - Generic expression node type (defined in `zirgen/Dialect/ZHL/IR/Types.td`)
- Mnemonic: `zhl.expr`
- This is the only type in ZHL - no field types, no component types
- All values in ZHL are represented as untyped `Expr` nodes

This minimal typing is intentional - it allows the DSL parser to emit IR without understanding types, delegating all type information to the subsequent type-checking pass.

## Key Characteristics

### Untyped by Design
ZHL intentionally lacks type information:
- All expressions are `!zhl.expr` type
- Component types, field types, and array element types are not represented
- Type checking is deferred to the ZHLT conversion pass

### Direct DSL Mapping
ZHL operations correspond directly to DSL constructs:
```
DSL: component Foo { x: Val; y: Val; }
ZHL: zhl.component "Foo" { ... }

DSL: let arr = [1, 2, 3];
ZHL: %arr = zhl.array %1, %2, %3 : !zhl.expr

DSL: foo.bar
ZHL: %result = zhl.lookup %foo["bar"] : !zhl.expr
```

### Separation of Concerns
By splitting parsing and type checking:
1. Parser focuses solely on syntax and structure
2. Type checker handles semantics and validation
3. Compiler architecture remains modular and maintainable

## Example

```mlir
// Component declaration
zhl.component @MyComponent {
  // Declare members
  %x = zhl.declare "x" : !zhl.expr
  %y = zhl.declare "y" : !zhl.expr

  // Array construction
  %arr = zhl.array %x, %y : !zhl.expr

  // Lookup
  %elem = zhl.lookup %arr["0"] : !zhl.expr

  // Constraint
  zhl.constrain %x, %elem : !zhl.expr, !zhl.expr
}
```

## Compilation Flow

### From DSL to ZHL
1. **Parse** (`zirgen/dsl/parser.cpp`): `.zir` file → AST
2. **Lower** (`zirgen/dsl/lower.cpp`): AST → ZHL Module (all `!zhl.expr`)

### From ZHL to ZHLT
3. **Type Check** (`zirgen/Conversions/Typing/ComponentManager.h`):
   - ZHL Module → ZHLT Module
   - `!zhl.expr` → specific types (`!zll.val<BabyBear>`, component types, etc.)
   - Validates constraints, resolves symbols, infers types

## Design Philosophy

1. **Minimal Typing**: Single `Expr` type keeps parsing simple
2. **DSL Fidelity**: Operations closely mirror DSL syntax
3. **Clean Separation**: Parser doesn't need type system knowledge
4. **Modularity**: Type checking is an independent transformation pass
5. **Explicit Structure**: All DSL constructs have corresponding operations

## File Organization

```
zirgen/Dialect/ZHL/
├── IR/
│   ├── Dialect.td      - Dialect definition (name: "zhl")
│   ├── Ops.td          - 23 operation definitions
│   ├── Types.td        - Single Expr type definition
│   ├── Attrs.td        - Attribute definitions (minimal)
│   ├── ZHL.h           - Main header
│   └── BUILD.bazel     - Build configuration
└── (No transformation passes - ZHL is immediately type-checked to ZHLT)
```

## Related Files

- **DSL Parser**: `zirgen/dsl/parser.h/cpp` - Parses .zir files to AST
- **AST Lowering**: `zirgen/dsl/lower.h/cpp` - Lowers AST to ZHL
- **Type Checking**: `zirgen/Conversions/Typing/ComponentManager.h` - Converts ZHL to ZHLT
- **Main Pipeline**: `zirgen/Main/gen_zirgen.cpp` - Orchestrates compilation

## See Also

- [ZHLT Dialect](../ZHLT/README.md) - The typed version of ZHL
- [Zll Dialect](../Zll/README.md) - Low-level circuit IR
- [Compiler Architecture](../../docs/COMPILER_ARCHITECTURE.md) - Overall pipeline
- [DSL Reference](../../dsl/README.md) - Zirgen DSL documentation (if available)
