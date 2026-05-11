# recursion — Proof Composition Circuit

The recursion circuit enables proof aggregation in RISC Zero. It takes a
verification predicate as input and produces a circuit that can verify proofs
of other circuits, allowing multiple proofs to be combined into a single
succinct proof.

## How it works

The recursion circuit implements a RISC Zero VM that executes a small
"recursion program" — a verification predicate encoded as RISC Zero bytecode.
Running this program inside a zkVM circuit produces a proof that the inner
proof was valid, which can itself be fed into another recursion step. This is
the mechanism that allows RISC Zero to aggregate proofs from the rv32im and
bigint circuits into compact, fixed-size receipts.

## Key source files

| File | Role |
|------|------|
| `top.cpp` / `top.h` | Top-level circuit definition; owns the mux over micro/macro ops |
| `code.cpp` / `code.h` | Instruction encoding and opcode decoding |
| `sha.cpp` / `sha.h` | SHA-256 acceleration used in Merkle path verification |
| `poseidon2.cpp` / `poseidon2.h` | Poseidon2 hash used for FRI commitments and the WOM |
| `wom.cpp` / `wom.h` | Write-Once Memory used to pass state between recursion cycles |
| `macro.cpp` / `macro.h` | Macro-op implementations (memory load/store, control flow) |
| `micro.cpp` / `micro.h` | Micro-op implementations (arithmetic, bitwise) |
| `checked_bytes.cpp` / `checked_bytes.h` | Range-checked byte decomposition |

## Relationship to other circuits

The recursion circuit is designed to compose with:

* **rv32im** — Recursion verifies rv32im execution proofs as its primary use case.
* **bigint** — Bigint operation proofs can be folded into a recursion proof to
  produce an aggregated receipt.
