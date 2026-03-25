# R1CS Dialect

The **R1CS (Rank-1 Constraint System)** dialect provides operations and types for representing and importing constraint systems in the R1CS format. Unlike other Zirgen dialects that are part of the main compilation pipeline, R1CS serves as an import-only dialect for converting external R1CS constraint systems into Zirgen's IR.

## Purpose

The R1CS dialect provides:

- **R1CS format support**: Import constraint systems from external tools
- **Standard constraint representation**: Rank-1 constraints of the form `a * b - c = 0`
- **Wire abstraction**: Represents circuit wires with labels and public/private designation
- **Conversion to Zirgen IR**: Transforms R1CS → BigInt → Zll for further compilation

## Position in Compilation Pipeline

```
R1CS Format (external file)
    ↓
[R1CS Import Tool: zirgen-r1cs]
    ↓
R1CS Dialect ← YOU ARE HERE
    ↓
[R1CSToBigInt Pass]
    ↓
BigInt Dialect
    ↓
[LowerZll Pass]
    ↓
Zll Dialect (joins main pipeline)
```

The R1CS dialect is **not** produced by Zirgen's DSL compiler - it exists solely for importing external R1CS constraint systems.

## Key Operations

Defined in `zirgen/Dialect/R1CS/IR/Ops.td` (only 4 operations):

### Wire Definition
- `r1cs.def` - Define a wire and its label ID
  - Syntax: `%wire = r1cs.def <label>, <isPublic> -> !r1cs.wire`
  - Parameters: label (64-bit ID), isPublic flag
  - Returns wire identifier

### Constraint Construction
- `r1cs.mul` - Multiply a wire by a constant
  - Syntax: `%factor = r1cs.mul %wire * <constant> -> !r1cs.factor`
  - Parameters: wire, constant value, prime modulus
  - Returns factor (wire * constant)

- `r1cs.sum` - Combine factors by summing
  - Syntax: `%sum = r1cs.sum %lhs + %rhs -> !r1cs.factor`
  - Adds two factors together
  - Returns combined factor

### Constraint Application
- `r1cs.constrain` - Assert constraint: a*b-c=0
  - Syntax: `r1cs.constrain %a, %b[, %c]`
  - Represents Rank-1 constraint: `a * b - c = 0`
  - If `c` omitted, constraint is `a * b = 0`

## Type System

Defined in `zirgen/Dialect/R1CS/IR/Types.td` (only 2 types):

### Core Types

**Wire** - Wire identifier
- Represents a circuit wire (variable)
- Labeled with 64-bit ID
- Can be public or private
- Syntax: `!r1cs.wire`

**Factor** - Wire multiplied by constant
- Represents `wire * constant` in finite field
- Used to build linear combinations
- Syntax: `!r1cs.factor`

## R1CS Format

The R1CS dialect represents the standard Rank-1 Constraint System format where constraints have the form:

```
a * b = c
```

or equivalently:

```
a * b - c = 0
```

where `a`, `b`, and `c` are linear combinations of wires:
- `a = Σ(constant_i * wire_i)`
- `b = Σ(constant_j * wire_j)`
- `c = Σ(constant_k * wire_k)`

## Conversion Pipeline

### R1CSToBigInt Pass
Located in `zirgen/Dialect/R1CS/Conversion/R1CSToBigInt/`:

The `R1CSToBigInt` pass converts R1CS operations to BigInt dialect operations:
1. Wires → BigInt variables
2. Factors → BigInt arithmetic expressions
3. Constraints → BigInt constraint operations

### Complete Flow
```
External R1CS File
    ↓
zirgen-r1cs tool (parses R1CS format)
    ↓
R1CS Dialect IR
    ↓
R1CSToBigInt Pass
    ↓
BigInt Dialect IR
    ↓
LowerZll Pass (BigInt → Zll)
    ↓
Zll Dialect IR
    ↓
Standard Zirgen compilation pipeline
```

## Example

```mlir
// Define wires
%w0 = r1cs.def 0, false -> !r1cs.wire  // Private wire 0
%w1 = r1cs.def 1, true -> !r1cs.wire   // Public wire 1
%w2 = r1cs.def 2, false -> !r1cs.wire  // Private wire 2

// Build factors (linear combinations)
%f_a = r1cs.mul %w0 * 5 -> !r1cs.factor  // a = 5*w0
%f_b = r1cs.mul %w1 * 3 -> !r1cs.factor  // b = 3*w1
%f_c1 = r1cs.mul %w2 * 1 -> !r1cs.factor // c = 1*w2
%f_c2 = r1cs.mul %w0 * 2 -> !r1cs.factor //     + 2*w0
%f_c = r1cs.sum %f_c1 + %f_c2 -> !r1cs.factor

// Apply constraint: a * b = c
// i.e., (5*w0) * (3*w1) = (1*w2 + 2*w0)
r1cs.constrain %f_a, %f_b, %f_c
```

## Entry Point

The main entry point for R1CS import is `zirgen/compiler/tools/zirgen-r1cs.cpp`:
- Parses R1CS format files
- Constructs R1CS dialect IR
- Applies R1CSToBigInt conversion
- Outputs Zll IR for further processing

## Use Cases

1. **External Tool Integration**: Import circuits from other ZK toolchains
2. **Format Conversion**: Convert R1CS to Zirgen's native representation
3. **Legacy Support**: Work with existing R1CS constraint systems
4. **Interoperability**: Bridge between different ZK frameworks

## Design Philosophy

1. **Minimal**: Only 4 operations and 2 types - just enough to represent R1CS
2. **Standard Format**: Adheres to widely-used R1CS specification
3. **Import-Only**: Not produced by Zirgen's compiler, only consumed
4. **Clean Conversion**: Clear transformation path to BigInt/Zll
5. **Interoperability**: Enables integration with external tools

## Limitations

- **Import-only**: Cannot generate R1CS from Zirgen circuits
- **No optimization**: R1CS is immediately lowered to BigInt
- **Limited abstraction**: Flat constraint representation without structure
- **External dependency**: Requires R1CS format files

## File Organization

```
zirgen/Dialect/R1CS/
├── IR/
│   ├── Dialect.td      - Dialect definition (name: "r1cs")
│   ├── Ops.td          - 4 operation definitions
│   ├── Types.td        - 2 type definitions (Wire, Factor)
│   ├── Attrs.td        - Attributes
│   └── BUILD.bazel     - Build configuration
└── Conversion/
    └── R1CSToBigInt/
        ├── Passes.td           - Pass definition
        ├── R1CSToBigInt.cpp    - Conversion implementation
        └── BUILD.bazel

```

## Related Tools

- **zirgen-r1cs**: CLI tool for R1CS import (`zirgen/compiler/tools/zirgen-r1cs.cpp`)
- **R1CS Parser**: Parses external R1CS files
- **R1CSToBigInt**: Conversion pass to BigInt dialect

## See Also

- [BigInt Dialect](../BigInt/README.md) - Target of R1CS conversion (if docs exist)
- [Zll Dialect](../Zll/README.md) - Final target after BigInt lowering
- [Compiler Architecture](../../docs/COMPILER_ARCHITECTURE.md) - Overall pipeline
- [zirgen-r1cs Tool](../../compiler/tools/README.md) - Import tool documentation (if exists)
