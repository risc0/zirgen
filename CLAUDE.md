# Zirgen — Claude Code Orientation

## Project Purpose

Zirgen is a domain-specific language (DSL) and compiler for writing arithmetic circuits for the [RISC Zero](https://www.risczero.com/) proof system. Circuits are STARK-based and reason over a grid of finite-field elements (the *witness*) subject to polynomial constraints of degree ≤ 5.

Primary use cases:
- zk accelerators (hashing, bigint, crypto primitives)
- zkVMs (the RISC-V rv32im VM is included)
- Recursion circuits
- Arbitrary zero-knowledge applications

## Build System

Zirgen uses **Bazel**. Key commands:

```sh
# Build the zirgen compiler binary
bazel build //zirgen/Main:zirgen

# Run all tests
bazel test //...

# Run a single .zir file through the compiler
bazel run //zirgen/Main:zirgen -- --emit=zstruct path/to/circuit.zir

# Run built-in tests in a .zir file
bazel run //zirgen/Main:zirgen -- --test path/to/circuit.zir

# Run tests with a specific number of cycles
bazel run //zirgen/Main:zirgen -- --test --test-cycles=6 path/to/circuit.zir
```

Bazel rules for Zirgen circuits live in `bazel/rules/zirgen/`. The `zirgen_genfiles()` macro generates Rust/C++/CUDA from `.zir` source files.

Emission targets: `--emit=zhlt`, `--emit=zstruct`, `--emit=rust`, `--emit=stats`.

## Directory Layout

```
zirgen/
├── Main/               # CLI entry point (driver, test runner, codegen)
├── dsl/                # DSL front-end: lexer, parser, AST, lowering
│   ├── test/           # 120+ .zir unit tests
│   └── examples/       # fibonacci.zir, calculator/, …
├── Dialect/            # MLIR dialect hierarchy (see Dialect/README.md)
│   ├── ZHL/            # High-level dialect (close to source)
│   ├── ZHLT/           # Transformed ZHL with optimizations
│   ├── ZStruct/        # Structured/layout dialect
│   ├── Zll/            # Low-level dialect
│   ├── BigInt/         # Arbitrary-precision integer ops
│   ├── IOP/            # Interactive Oracle Proof dialect
│   └── R1CS/           # Rank-1 Constraint System (Circom integration)
├── components/         # Reusable C++ circuit components (Bit, Byte, RAM, …)
├── circuit/
│   ├── recursion/      # The recursion circuit
│   └── rv32im/         # The RISC-V zkVM circuit
├── compiler/           # Additional compiler tooling (zirgen-r1cs, etc.)
└── docs/               # User-facing language documentation
```

## Key Subsystems

### DSL (Front-end)

Source files have a `.zir` extension. The front-end pipeline is:

1. **Lexer** (`dsl/lexer.h`) — tokenises source; supports `@` (back-reference), `->` (mux), `..` (range), `:=` (definition), `=` (constraint)
2. **Parser** (`dsl/parser.h`) — recursive descent, produces an AST (`dsl/ast.h`)
3. **Lowering** (`dsl/lower.h`) — AST → ZHL MLIR dialect

See [`zirgen/dsl/README.md`](zirgen/dsl/README.md) for a full pipeline description.

### MLIR Dialect Hierarchy

Compilation flows through several MLIR dialects:

```
Source (.zir) → ZHL → ZHLT → ZStruct → Zll → Backend (Rust/C++/CUDA)
                                 ↕
                           BigInt / IOP / R1CS
```

See [`zirgen/Dialect/README.md`](zirgen/Dialect/README.md) for details on each dialect.

### Components (C++ library)

`zirgen/components/` contains pre-built circuit primitives used when writing circuits in C++ rather than ZIR:

| Header | Purpose |
|--------|---------|
| `reg.h` | Single witness column (register) |
| `bits.h` | Single-bit register; 2-bit (twit) register |
| `bytes.h` | Byte-range-checked register via PLONK |
| `u32.h` | 32-bit unsigned integer value and register |
| `ram.h` | RAM read/write with sorting argument |
| `fpext.h` | Extension-field element register |
| `iszero.h` | Zero-test component |
| `onehot.h` | One-hot encoded selector |
| `mux.h` | Template multiplexer over component arms |
| `plonk.h` | Generic PLONK permutation argument infrastructure |

### Circuit Implementations

- `zirgen/circuit/rv32im/` — RISC-V RV32IM zkVM
- `zirgen/circuit/recursion/` — Recursion verifier circuit

## Language Quick Reference

```zir
// Component definition
component Foo(x: Val, y: Val) {
  sum := x + y;        // :=  defines a local name
  sum = x + y;         // =   adds a polynomial constraint
  public r : Reg;      // declares a public register field
  r := Reg(sum);
}

// Back-references (previous cycle value)
x@1          // value of x from 1 cycle ago
arr@1[i]     // element i of arr from 1 cycle ago

// Arrays and loops
arr := for i : 0..4 { Reg(i) };   // creates Array<Reg, 4>
result := reduce arr init 0 with Add;

// Mux (selector must be a one-hot array)
[sel, 1-sel] -> (arm_if_true, arm_if_false)

// Externs (nondeterministic host calls)
extern GetCycle() : Val;
extern Output(v: Val);
```

## Documentation

User-facing language docs live in `zirgen/docs/`. Start with:

1. [Getting Started](zirgen/docs/01_Getting_Started.md)
2. [Conceptual Overview](zirgen/docs/02_Conceptual_Overview.md)
3. [Building a Fibonacci Circuit](zirgen/docs/03_Building_a_Fibonacci_Circuit.md)
4. [Components](zirgen/docs/04_Components.md)
5. [Muxes](zirgen/docs/05_Muxes.md)
6. [Built-in Components](zirgen/docs/A1_Builtin_Components.md)
