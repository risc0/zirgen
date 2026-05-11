# rv32im — RISC-V zkVM Circuit

This is the primary RISC-V zkVM circuit used in RISC Zero. It implements the
RV32IM instruction set as a zero-knowledge proof circuit, allowing RISC Zero to
generate proofs of arbitrary RISC-V program execution.

## Versions

The circuit exists in two versions:

* **v1** (`v1/`) — The original implementation written using the EDSL (C++
  embedded DSL). This is the legacy version.
* **v2** (`v2/`) — A rewrite in the ZIR DSL. This is the current production
  version.

When making changes or adding features, target `v2/`.

## Shared infrastructure (`shared/`)

The `shared/` subdirectory contains platform code used by both versions:

* **`emu/`** — A software emulator for the RV32IM ISA used to generate
  execution witnesses.
* **`kernel/`** — The zkVM kernel, which handles syscalls and I/O between the
  guest program and the host.
* **`run/`** — The circuit runner that drives multi-cycle execution.
* **`platform/`** — Platform-specific definitions (memory layout, page tables,
  system call numbers).

## Further reading

* [ARCHITECTURE.md](../../../ARCHITECTURE.md) — Describes the full codegen
  pipeline from ZIR source through MLIR dialects to generated C++/CUDA.
* [docs/TESTING_IN_RISC0.md](../../../docs/TESTING_IN_RISC0.md) — How to
  integrate and test changes to this circuit inside the risc0 repository.
