# Zirgen MLIR Dialect Hierarchy

Zirgen uses a layered stack of MLIR dialects to progressively lower a `.zir` source file into backend output (Rust, C++, CUDA). Each dialect represents a different level of abstraction, and a sequence of conversion/rewriting passes moves the IR from one level to the next.

## Compilation Flow

```
.zir source
    │
    ▼  (dsl/lower.cpp)
  ZHL          — High-level; mirrors source structure
    │
    ▼  (ZHLT/Transforms/)
  ZHLT         — Transformed/optimized ZHL; layout computed
    │
    ▼  (ZStruct/Transforms/)
  ZStruct      — Explicit structure layout; memory model resolved
    │
    ▼  (Zll/Transforms/ + Zll/Conversion/)
  Zll          — Low-level polynomial constraints and witness ops
    │
    ├──► Rust / C++ / CUDA backend emitters
    │
    ├──► BigInt  (if circuit uses bigint ops)
    ├──► IOP     (if circuit needs oracle proof ops)
    └──► R1CS    (if targeting Circom/R1CS export)
```

## ZHL — ZIRgen High-Level Dialect

**Location**: `ZHL/IR/`

ZHL is produced directly by the DSL lowering pass (`dsl/lower.cpp`). It closely mirrors the component/mux/for structure of the source language. At this stage:

- Components are still named types with explicit parameter lists.
- `for` loops, `reduce`, and mux expressions exist as first-class operations.
- Back-references (`@N`) are represented symbolically.

ZHL is primarily a target for type inference and component resolution.

## ZHLT — ZHL Transformed Dialect

**Location**: `ZHLT/IR/`, `ZHLT/Transforms/`

ZHLT is the result of the first major transformation pass over ZHL. Key changes:

- Component types are monomorphised (type parameters are substituted).
- Layout constraints are resolved: each component knows exactly which witness columns it owns.
- Mux arms are made explicit.

The `--emit=zhlt` flag stops compilation here and dumps the ZHLT IR.

## ZStruct — Structured Dialect

**Location**: `ZStruct/IR/`, `ZStruct/Transforms/`, `ZStruct/Analysis/`

ZStruct introduces an explicit memory model for the witness:

- Witness columns are described as a flat `struct`-like layout.
- Field accesses become explicit buffer indexing.
- Back-references are resolved to concrete column offsets.
- Analysis passes (in `ZStruct/Analysis/`) check for alias layout conflicts and perform alias-hint propagation.

The `--emit=zstruct` flag stops here. This is a useful debugging target when investigating layout or constraint issues.

## Zll — ZIRgen Low-Level Dialect

**Location**: `Zll/IR/`, `Zll/Transforms/`, `Zll/Analysis/`, `Zll/Conversion/`

Zll is the final common IR before backend emission. At this level:

- All circuit operations are expressed as polynomial arithmetic over the BabyBear field (or Goldilocks, depending on configuration).
- The `zll.val` type represents a single finite-field element.
- Constraints are `zll.eq` operations (asserting two values are equal as field elements).
- `zll.extern` represents nondeterministic calls to the prover host.
- Witness generation logic and constraint logic are kept in separate function bodies (`exec_func` vs `validity_func`).

The interpreter in `Zll/IR/Interpreter.h` can execute Zll IR directly, which is how `--test` mode works.

## BigInt Dialect

**Location**: `BigInt/IR/`, `BigInt/Transforms/`, `BigInt/Bytecode/`

The BigInt dialect supports arbitrary-precision integer operations inside ZK circuits, useful for modular arithmetic over fields larger than BabyBear (e.g., RSA, secp256k1). See `BigInt/Overview.md` and `BigInt/Precompiles.md` for details.

## IOP Dialect

**Location**: `IOP/IR/`

The IOP (Interactive Oracle Proof) dialect models interactive proof protocol steps, such as Fiat-Shamir challenges and Merkle queries. It is used when building recursion circuits that must verify the inner STARK proof.

## R1CS Dialect

**Location**: `R1CS/IR/`, `R1CS/Conversion/`

The R1CS dialect enables export of circuits as Rank-1 Constraint Systems, compatible with Circom tooling. The `zirgen-r1cs` tool (`compiler/tools/zirgen-r1cs.cpp`) drives this path and allows Circom witnesses to be verified inside RISC Zero recursion programs.

## Adding a New Dialect

1. Create a directory under `zirgen/Dialect/<Name>/`.
2. Define ops in `IR/Ops.td` (TableGen) and types in `IR/Types.td`.
3. Register the dialect in `IR/Dialect.cpp` and add it to the MLIR context in the driver.
4. Add conversion passes to `Conversion/` or `Transforms/` as needed.
5. Wire the new passes into `Main/Main.cpp`.
