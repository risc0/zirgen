# Bootstrapping zirgen changes into risc0 and risc0-ethereum

How to carry a change to the zirgen circuits into the generated code in `risc0` and the
verifier contracts in `risc0-ethereum`. Use checkouts of all three repositories side by side,
on branches of the same release line.

What to run depends on what changed:

| Change in zirgen | Run |
| --- | --- |
| rv32im circuit (`zirgen/circuit/rv32im/v2/dsl`) | A |
| recursion programs (`zirgen/circuit/predicates`) | A from A2 |
| recursion circuit (`zirgen/circuit/recursion`) | D, then A from A2, then B |
| Groth16 wrapper (`zirgen/circuit/verify/circom`) | B |
| Poseidon254 hash | C, then B |

Any change to the control root also changes `ControlID.sol` in risc0-ethereum (B5).

**Prerequisites:** `bazelisk`, `clang-format` (without it the generated C++ is unformatted and
the diffs are huge), Docker, Foundry's `forge` on `PATH`, and for the CUDA prover the CUDA
toolkit with `nvcc` on `PATH`. Part B needs about 30 GB of disk and 64 GB of RAM.

## A. rv32im circuit and recursion programs

1. Regenerate the rv32im kernels and Rust code: `cargo bootstrap -- rv32im-v2 --output ../risc0/`.
2. Build the recursion programs: `bazelisk build //zirgen/circuit/predicates:gen_predicates`.
   They are in `bazel-out/k8-opt/bin/zirgen/circuit/predicates/*.zkr`. If they changed,
   update `zirgen/circuit/predicates/golden_hashes.txt` from `actual_hashes.txt`.
3. In risc0, pack them into the `recursion-zkrs` and `recursion-control-zkrs` crates:
   `cargo xtask pack-zkrs ../zirgen/bazel-out/k8-opt/bin/zirgen/circuit/predicates`.
   Packing is reproducible: only the tarballs whose programs changed should show as modified.
4. Regenerate the control IDs: `cargo xtask bootstrap`.
5. Update the stability canaries, which pin digests that follow the control IDs, once the
   change is confirmed intentional: `cargo test -F prove -p risc0-zkvm --lib -- is_stable stable_root`.
6. Test: `cargo test -F prove -p risc0-zkvm --no-fail-fast`.

Gotchas:

- Witness generation errors print `set(row: …)` through a `"%s(row: …"` format string; grep
  for `(row: `. They mean preflight and witgen disagree about a column, and the C++ exception
  usually surfaces as a crash.
- A constraint failure during parallel witgen crashes instead of returning an error. Tests
  that expect witgen to reject a witness must use `StepMode::SeqForward` or `SeqReverse`.
- Never `AliasLayout!` a lookup-argument column onto a data column, and remember that
  `NondetU16Reg(x)` does not constrain its value to `x` (use `U16Reg`).
- Adding an rv32im global changes the seal layout: bump `RV32IM_SEAL_VERSION`, and set the
  global in both risc0's `build_global_vec` and zirgen's `run.cpp`.
- Tests through the client/server path use `RISC0_SERVER_PATH`, then an rzup `r0vm`. A released
  `r0vm` has the old circuit, so point `RISC0_SERVER_PATH` at a local build.

## B. Groth16 circuit and trusted setup

1. Regenerate the circuit: `cargo bootstrap -- verify --output ../risc0/`. Delete the copy it
   leaves at `risc0/groth16/src/seal_format.rs` (the real one is in `src/prove/`). The circom
   generator is nondeterministic, so expect a large `stark_verify.circom` diff.
2. Before a ceremony, check that an honest witness satisfies the new circuit. Run anything
   that proves Groth16 with `RISC0_WORK_DIR` set, which leaves an `input.json` there, then run
   the witness generator stage of `groth16_proof/docker/prover.Dockerfile` (`--target witgen`)
   on it.
3. Run the ceremony: `docker build -f docker/ceremony.Dockerfile . -t snark-ceremony`, then run
   it with `groth16_proof/groth16` mounted. It needs a 2^23 phase-1 ptau file; verify it
   against its publisher before use. The ceremony image and the prover image must build the
   same r1cs: compare the two `stark_verify.r1cs` hashes. Check the result with
   `snarkjs zkey verify`.
4. Upload `stark_verify_final.zkey.gz` under a new `zkey/<yyyy-mm-dd.rev>/` path, pin its
   checksum in `prover.Dockerfile`, build and push `risczero/risc0-groth16-prover:v<yyyy-mm-dd.rev>`,
   and update the tag in `risc0/groth16/src/prove/docker.rs`.
5. Update the verifiers and the test receipt, from risc0, with `forge` on `PATH`:
   `RISC0_ETHEREUM_PATH=../risc0-ethereum cargo xtask bootstrap-groth16`. Point
   `SOLIDITY_TEST_RECEIPT_PATH` in `xtask/src/bootstrap_groth16.rs` at the release's
   `TestReceiptV<x>_<y>.sol` first. Then run it again with `--test-receipt-only`: the first run
   computes the receipt's selector from the old verifying key.
6. Update the canaries again (A5), and `testSelectorIsStable` in risc0-ethereum's
   `RiscZeroGroth16Verifier.t.sol`. Then run `forge test` in `risc0-ethereum/contracts`.
7. For the CUDA prover, rebuild the rzup `risc0-groth16` component with
   `cargo xtask setup-groth16` after placing the new zkey in the component directory, and bump
   its version and the pin in `risc0/groth16/src/prove/cuda.rs` (see E).

Test with `cargo test -F prove,docker -p risc0-zkvm --lib -- --test-threads=1 compress` (and
`-F prove,cuda`). Groth16 tests are memory-heavy; run them one at a time.

## C. Poseidon254 hash

The hash has four implementations that must change together: `risc0/zkp/src/core/hash/poseidon_254/`
(CPU), `risc0/sys/kernels/zkp/cuda/supra/poseidon254.cuh` (CUDA),
`zirgen/compiler/zkp/poseidon_254.cpp`, and `zirgen/circuit/verify/circom/include/risc0.circom`,
which risc0 copies to `groth16_proof/groth16/risc0.circom`. Update the known-answer vector in
`zirgen/circuit/verify/circom/test/AB.cpp` from the Rust implementation, run
`cargo xtask bootstrap` (it recomputes `BN254_IDENTITY_CONTROL_ID`), then run B from step 2.

## D. Recursion circuit

1. Test in zirgen: `bazelisk test //zirgen/circuit/recursion/... //zirgen/circuit/verify/... //zirgen/circuit/predicates/...`.
2. Check the cycle budget. Every recursion program must fit in `2^18 - 1024` rows, and
   `join_povw` is at the limit, so a new constraint must cost almost nothing in the verifier.
   Measure with `bazelisk build //zirgen/circuit/predicates:gen_predicates`; a program's rows
   are its `.zkr` size in bytes divided by 92.
3. Regenerate: `cargo bootstrap -- recursion --output ../risc0/`. It does not copy the
   hand-maintained `recursion-sys/kernels/cuda/eval_check.cuh`; update its `poly_mix[N]` if
   `NUM_POLY_MIX_POWERS` in the regenerated `info.rs` changed.
4. Continue with A from step 2, then B.

## E. The CUDA Groth16 component

The CUDA prover reads the rzup component `risc0-groth16`, whose version `cuda.rs` pins. With an
rzup built from this repository:

```
rzup publish create-artifact --input <component dir> --output risc0-groth16-<ver>.tar.xz
rzup publish upload risc0-groth16 <ver> --target-agnostic risc0-groth16-<ver>.tar.xz
rzup publish set-latest risc0-groth16 <ver>   # on release day
```

`upload` and `set-latest` need `RZUP_PUBLISH_URL` (the bucket's S3 API URL),
`AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` for it, and `RZUP_PUBLISH_KEY`, the key that
signs the component manifest. Uploading early is safe: only `set-latest` changes what users get.
