# keccak — Keccak/SHA3 Accelerator Circuit

This circuit implements the Keccak-f[1600] permutation as a ZIR circuit,
providing a zkVM precompile for Ethereum-compatible Keccak-256 and SHA3
hashing. Guest programs that call the Keccak precompile have their hash
computations verified by this circuit rather than executed inside the rv32im
circuit, significantly reducing proof size and generation time.

## Key source files

| File | Role |
|------|------|
| `keccak.zir` | Core Keccak-f[1600] permutation (Theta, Rho, Pi, Chi, Iota steps) |
| `top.zir` | Top-level circuit; manages the control flow across Keccak rounds |
| `bits.zir` | Bit decomposition and reconstruction helpers |
| `xor5.zir` | Optimized 5-input XOR used in the Theta step |
| `arr.zir` | Array utilities |
| `pack.zir` | Packing/unpacking between field elements and 64-bit lane representation |
| `cycle_counter.zir` | Tracks which round within a Keccak permutation is executing |

## Build notes

Due to the size of the Keccak circuit, the step function is split across
multiple generated files to keep each compilation unit manageable. The
BUILD.bazel file sets:

```
--step-split-count=16
```

which produces 16 separate `steps_N.cpp` / `steps_N.cu` files instead of a
single large step function. This is required for the circuit to compile in
reasonable time and memory.

The `--parallel-witgen` flag is also set to enable parallel witness generation
across the 16 step shards.
