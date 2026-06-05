# Zirgen — Claude Code Contributor Guide

## Build Commands

### Bazel (primary build system)

```bash
# Build the Zirgen compiler
bazel build //zirgen/dsl:zirgen

# Run all tests
bazel test //...

# Run DSL interpreter tests only
bazel test //zirgen/dsl/...

# Run a single .zir file in test mode
bazel run //zirgen/dsl:zirgen -- $(pwd)/zirgen/dsl/test/hello_world.zir --test

# Run a file with multiple cycles (for multi-cycle circuits)
bazel run //zirgen/dsl:zirgen -- $(pwd)/zirgen/dsl/test/back_counter.zir --test --test-cycles 4

# Emit C++ or Rust output
bazel run //zirgen/dsl:zirgen -- $(pwd)/path/to/circuit.zir --emit=cpp
bazel run //zirgen/dsl:zirgen -- $(pwd)/path/to/circuit.zir --emit=rust
```

### Cargo (Rust DSL crate)

```bash
# Build the DSL Rust crate
cargo build -p risc0-zirgen-dsl

# Run Rust tests
cargo test -p risc0-zirgen-dsl
```

## Repo Layout

```
zirgen/
├── circuit/          # Concrete circuits written in .zir
│   ├── rv32im/       # RISC-V zkVM circuit
│   ├── recursion/    # Recursion circuit
│   └── fib/          # Fibonacci example circuit
├── compiler/         # MLIR passes and code-generation tools
│   └── tools/        # CLI tools including zirgen-r1cs (Circom integration)
├── components/       # Reusable circuit component libraries
├── Dialect/          # MLIR dialect definitions (compiler IR layers)
│   ├── ZHL/          # High-level ZHL dialect (parsed from .zir source)
│   ├── ZHLT/         # High-level typed dialect
│   ├── ZStruct/      # Struct-lowering dialect
│   └── Zll/          # Low-level field-arithmetic dialect
└── dsl/              # Rust crate: lexer, parser, interpreter, test runner
    ├── src/          # Rust source
    ├── test/         # .zir test files (used by Bazel LIT tests)
    └── examples/     # Standalone .zir example circuits
docs/                 # Language documentation (Markdown)
```

### Compiler IR Pipeline

Source `.zir` files are compiled through a cascade of MLIR dialects:

```
.zir source
   ↓  parser (zirgen/dsl/)
 ZHL   (high-level, untyped)
   ↓  type-checking / lowering
ZHLT  (high-level, typed)
   ↓  struct layout
ZStruct
   ↓  field arithmetic lowering
 Zll   (low-level; target for Rust/C++ codegen)
```

## Running Tests

### LIT tests (Bazel)

Most `.zir` files in `zirgen/dsl/test/` carry `// RUN:` directives that are
executed by the LLVM Integrated Tester (LIT) via Bazel:

```bash
bazel test //zirgen/dsl:zirgen_tests
```

### Interpreter quick-check

For rapid iteration on a circuit file:

```bash
bazel run //zirgen/dsl:zirgen -- $(pwd)/zirgen/dsl/test/map.zir --test
```

## Key Language Concepts (Quick Reference)

| Concept | Where to read |
|---------|--------------|
| Getting started / Hello World | `zirgen/docs/01_Getting_Started.md` |
| Conceptual overview (trace, constraints, witness) | `zirgen/docs/02_Conceptual_Overview.md` |
| Components (struct-like) | `zirgen/docs/04_Components.md` |
| Muxes (sum types / control flow) | `zirgen/docs/05_Muxes.md` |
| Arrays and `for`/`reduce` loops | `zirgen/docs/99_Arrays_and_Loops.md` |
| Back-references (`@1`, multi-cycle state) | `zirgen/docs/99_Backs.md` |
| Externs (host-provided values and callbacks) | `zirgen/docs/99_Externs.md` |
| Built-in components (`Val`, `Reg`, `NondetReg`, …) | `zirgen/docs/A1_Builtin_Components.md` |

## Notes for Contributors

* **Never commit directly to `main`** — open a PR for all changes.
* **Formatting**: run `clang-format` (via `clang-format.py`) on C++ files before
  committing. There is no auto-formatter for `.zir` files yet.
* **License headers**: new source files should include the Apache-2.0 SPDX
  header; `license-check.py` verifies this.
* **Constraint degree limit**: the proof system enforces a maximum constraint
  degree of 5. Mux selectors multiply into arm constraints, so be mindful of
  combined degree when writing mux-heavy code.
