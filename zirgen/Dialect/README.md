# zirgen/Dialect

Each subdirectory defines one MLIR dialect used in the zirgen compiler pipeline. Dialects are listed here in roughly pipeline order from high-level to low-level.

## ZHL — Zirgen High-Level

`ZHL/` contains the direct output of `zirgen::dsl::lower()`. It mirrors the ZIR DSL closely: components, fields, `if` expressions, `back` references, and generic type parameters are all represented as ZHL ops. This dialect is still untyped in the sense that generic components have not been instantiated yet.

## ZHLT — Zirgen High-Level Typed

`ZHLT/` is the typed successor to ZHL, produced by `zirgen::Typing::typeCheck()` via the `ComponentManager`. All generic components are monomorphized, and the module now contains `CheckFuncOp` (the validity polynomial body) and one or more `StepFuncOp`s (witness generation). Most subsequent optimization passes operate on ZHLT.

## ZStruct — Layout and Buffers

`ZStruct/` models the physical witness layout: named buffers, struct types with explicit column offsets, and array types. Layout passes assign concrete column indices to fields. The `BuffersToArgs` pass converts implicit buffer accesses into function arguments before step-function emission.

## Zll — Zirgen Low-Level

`Zll/` is the low-level polynomial dialect that sits just above target code emission. It exposes field arithmetic (`MulOp`, `AddOp`, `SubOp`), buffer access (`GetOp`, `SetOp`), and extension-field ops. The `MakePolynomial` pass lowers ZHLT check functions into Zll ops. GPU codegen reads Zll ops directly to emit the `eval_check` CUDA/Metal kernel.

## BigInt

`BigInt/` provides big-integer arithmetic operations used by accelerator circuits (e.g., elliptic-curve point addition, SHA-256 precompiles). These ops are lowered separately from the main Zll path and emit specialized constraint code.

## IOP — Interactive Oracle Proof

`IOP/` defines ops that represent IOP-level interactions such as Merkle queries and FRI folding steps. Used primarily by the recursion circuit to express verifier logic inside the circuit.

## R1CS

`R1CS/` contains an MLIR dialect that represents Rank-1 Constraint Systems, used as an intermediate form when targeting Circom via the `zirgen-r1cs` tool.

## Analysis

`Analysis/` contains MLIR analyses that are shared across multiple dialects, such as dominator-tree queries and data-flow analyses used by optimization passes.
