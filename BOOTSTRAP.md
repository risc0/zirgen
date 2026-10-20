# Bootstrapping zirgen circuit changes into risc0 and risc0-ethereum

This document describes how to take a change to the zirgen circuits and carry it through
to the `risc0` and `risc0-ethereum` repositories, including a new Groth16 trusted setup
when the STARK-to-SNARK circuit changes. It was written while backporting two soundness
fixes to the `risc0` v3.0.x line (risc0 `3.0.6`, risc0-ethereum `v3.0.1`) and records
every problem hit along the way.

There are two independent pipelines:

- **Part A — rv32im + recursion predicates.** Needed whenever the rv32im circuit
  (`zirgen/circuit/rv32im/v2/dsl/*.zir`) or the recursion programs change. Changes the
  rv32im kernels, the lift ZKRs, the recursion control IDs and the control root.
- **Part B — Groth16 (`stark_verify.circom`).** Needed whenever the circom generator
  (`zirgen/circuit/verify/circom/`) or the recursion circuit changes. Requires a new
  Groth16 trusted setup, a new prover image, and new Rust/Solidity verifiers.

If Part A changes the control root, the Solidity `ControlID.sol` in risc0-ethereum must be
updated too (Part B, step B9 does this), even if the circom circuit is unchanged.

## Where the fixes live

| Fix | File(s) |
| --- | --- |
| Host-read words not range checked | zirgen: `zirgen/circuit/rv32im/v2/dsl/inst_ecall.zir` (`ECallHostReadWords`) |
| Non-canonical Groth16 output decomposition | zirgen: `zirgen/circuit/verify/circom/circom.cpp` (`CircomGenerator::emit`) |
| Poseidon254 S-box `x^8` not a permutation | zirgen: `zirgen/compiler/zkp/poseidon_254.cpp`, `zirgen/circuit/verify/circom/include/risc0.circom`; **risc0**: `risc0/zkp/src/core/hash/poseidon_254/mod.rs`, `risc0/sys/kernels/zkp/cuda/supra/poseidon254.cuh`, `groth16_proof/groth16/risc0.circom` (see Part C) |

The first two fixes live entirely in zirgen. The Poseidon254 S-box fix does not: the prover
side of that hash is hand-written in risc0 (Rust and CUDA), and risc0 keeps its own copy of
`risc0.circom`, so those are real source changes in risc0 rather than generated output.

Apart from the risc0 files listed above, everything in `risc0` and `risc0-ethereum` is
generated output, a hash/ID/key derived from that output, a stability canary updated to
match, or build tooling that needed fixing to run at all. A third party can reproduce
equivalent deltas from the zirgen changes plus this document. They will not be
byte-identical, because:

- the trusted setup contribution is random, so the zkey, verifying key, `verifier.sol`,
  Groth16 selector, and verifier-parameter digests will differ; and
- the circom generator is nondeterministic (see B1), so a fresh bootstrap produces an
  equivalent but textually different `stark_verify.circom` unless the splice procedure
  below is used.

The rv32im/recursion side (Part A) should reproduce the same control IDs given the same
zirgen source; this was not independently verified.

### What each fix does

**rv32im host-read words.** The memory argument stores each 32-bit word as two field
elements (`dataLow`, `dataHigh`) that are assumed to be u16. `MemoryWriteUnconstrained`
(used by `ECallHostReadWords`) let the host write arbitrary values. In `MemoryIO`,
`oldTxn` holds the *previous* value and `newTxn` holds the value being written, so the
host-supplied data is `newTxn`. The fix is:

```
writeVal := MemoryWriteUnconstrained(cycle, addr);
// Verify the host-provided word is a valid pair of u16s
U16Reg(writeVal.io.newTxn.dataLow);
U16Reg(writeVal.io.newTxn.dataHigh);
```

This mirrors `ECallHostReadBytes`, which already constrains `io.newTxn.*` via `SplitWord`.

**Groth16 output decomposition.** `stark_verify.circom` packs the 32 recursion output
elements `out_tmp[0..32)` into four public outputs `out[i] = Σ_j out_tmp[8i+j] · 2^(16j)`.
Nothing constrained the `out_tmp` values, so the packing had many decompositions. The output
is two 16-element digests (see `risc0/zkvm/src/receipt/succinct.rs`, which decodes them):

- `out_tmp[0..16)`: the Poseidon2 control root, 8 BabyBear words each followed by a zero
  padding element;
- `out_tmp[16..32)`: the SHA-256 claim digest as 16 half-words.

The generator now emits, for each index `i`:

- even `i < 16`: `to_bits_exact(31)` followed by `valid_baby_bear()` (canonical BabyBear);
- odd `i < 16`: `out_tmp[i] === 0`;
- `i >= 16`: `to_bits_exact(16)`.

Note that `valid_baby_bear` takes the bit decomposition as input and does not decompose
itself, so it must be paired with `to_bits_exact(31)` (the same idiom `normalize_impl`,
`bit_and_impl` and `inv_impl` use).

**Poseidon254 S-box.** The S-box was `x^8`, computed as `x2 = x*x; x4 = x2*x2; out = x4*x4`.
`x^n` permutes `F_r` only if `gcd(n, r-1) = 1`; the BN254 scalar field has 2-adicity 28, so
`x^8` is 8-to-1. (`x^8` is not a permutation in *any* odd prime field, since `p-1` is always
even, so this was an implementation slip rather than a bad parameter import.) Because 8th
roots of unity exist, collisions are free: replacing each lane `x_i` with
`ζ_i·(x_i + c_i) - c_i` for any `ζ_i^8 = 1` leaves every S-box output, and therefore the
whole permutation output, unchanged. In `hash_pair` the capacity lane is pinned to 0, which
forces `ζ_0 = 1`, but the two data lanes are free, giving 64 colliding `(a, b)` pairs per
digest for two multiplications of work. That breaks Merkle-commitment binding and
Fiat-Shamir in the identity_p254 STARK that the SNARK wrapper verifies.

The fix is `x^7` (`x2 = x*x; x4 = x2*x2; x6 = x4*x2; out = x6*x`). `x^5` is the smallest
permutation exponent for this field, but `x^7` was chosen because the existing parameters
(`t=3`, `R_F=8`, `R_P=42`) are already the standard 128-bit set for α=7, while α=5 would
need `R_P=57` and therefore new round constants and a new MDS matrix. The reference
Grain-LFSR parameter generator seeds on field, S-box *type*, field size, `t`, `R_F` and
`R_P`, not on the exponent, so moving 8 → 7 needs no constant changes at all. Cost: `x^7` is
4 R1CS constraints per S-box instead of 3, which took `stark_verify` from 5,677,093 to
6,052,250 constraints (+6.6%), still far under the 2^23 ptau ceiling.

## Prerequisites

- Sibling checkouts: `zirgen/`, `risc0/`, `risc0-ethereum/` in the same directory.
- `bazelisk` (zirgen builds), Rust toolchain per each repo's `rust-toolchain.toml`.
- Docker (ceremony, Groth16 prover image, witness generator). No sudo required if the
  user is in the `docker` group.
- `forge` (Foundry): `curl -L https://foundry.paradigm.xyz | bash && ~/.foundry/bin/foundryup`.
  `cargo xtask bootstrap-groth16` shells out to `forge fmt`, so it must be on `PATH`.
- For the CUDA prover only: the CUDA toolkit, with `/usr/local/cuda/bin` on `PATH`.
- Disk: about 30 GB free for Part B (the ptau file is 9.7 GB, each zkey 3.6 GB).
- RAM: 64 GB was enough, provided the Groth16 tests run one at a time (see Part B tests).

### Branches used for the v3.0.x backport

- `zirgen`: branch `jeremy-bugfix`.
- `risc0`: tag/commit `1cc70cf05` (Release 3.0.6) plus the changes below.
- `risc0-ethereum`: a branch off the `v3.0.1` tag. Do **not** use `main`, which tracks a
  newer risc0 (its `ControlID.sol` and test receipts belong to a different version). Check
  that `contracts/src/groth16/ControlID.sol` on the chosen base has the same
  `CONTROL_ROOT` as `ALLOWED_CONTROL_ROOT` in risc0's
  `risc0/circuit/recursion/src/control_id.rs` before the change.

---

## Part A — rv32im circuit and recursion predicates

All zirgen commands run from the zirgen repo root; all risc0 commands from the risc0 root.

### A1. Regenerate the rv32im kernels and Rust code

```
cargo bootstrap -- rv32im-v2 --output ../risc0/
```

This rewrites:

- `risc0/circuit/rv32im-sys/kernels/cxx/*` and `kernels/cuda/*` (witness generation,
  constraint evaluation, layout);
- `risc0/circuit/rv32im/src/zirgen/*` (`layout.rs.inc`, `poly_ext.rs`, `types.rs.inc`,
  `info.rs`).

Gotcha: if `clang-format` is not installed, the bootstrap prints
`unable to format ... with clang-format` and leaves the generated C++/CUDA unformatted.
The code is fine, but the diff is much larger than necessary. Install `clang-format` to get
a reviewable diff.

### A2. Regenerate the recursion ZKRs

```
cargo bootstrap -- predicates --output ../risc0/
```

This rewrites `risc0/circuit/recursion/src/recursion_zkr.zip` (the lift programs embed the
rv32im verifier, so they change whenever the rv32im circuit does). This target builds from
zirgen source directly and does not depend on A1 having run.

### A3. Update the ZKR archive hash

```
shasum -a 256 risc0/circuit/recursion/src/recursion_zkr.zip
```

Put the hash in `SHA256_HASH` in `risc0/circuit/recursion/build.rs`.

`build.rs` uses the in-repo zip only if its hash matches `SHA256_HASH`; otherwise it
downloads `https://risc0-artifacts.s3.us-west-2.amazonaws.com/zkr/<SHA256_HASH>.zip`. Local
builds therefore work without uploading anything, but a release must upload the zip to that
S3 path (crates.io builds do not have the zip).

### A4. Regenerate control IDs

```
cargo xtask bootstrap
```

This rewrites `risc0/circuit/recursion/src/control_id.rs` (all ZKR control IDs,
`ALLOWED_CONTROL_ROOT`, `BN254_IDENTITY_CONTROL_ID`) and
`risc0/circuit/keccak/src/control_id.rs`.

### A5. Update the stability canaries

These tests intentionally pin digests that change with the control IDs. Run them, and
replace the expected values with the new ones, only when the change is intentional:

```
cargo test -F prove -p risc0-zkvm --lib -- is_stable stable_root
```

- `risc0/zkvm/src/host/recursion/tests.rs`: `stable_root` (= new `ALLOWED_CONTROL_ROOT`)
- `risc0/zkvm/src/receipt/succinct.rs`: `succinct_receipt_verifier_parameters_is_stable`
- `risc0/zkvm/src/receipt/composite.rs`: `composite_receipt_verifier_parameters_is_stable`
- `risc0/zkvm/src/receipt/groth16.rs`: `groth16_receipt_verifier_parameters_is_stable`

The composite and Groth16 digests change again after Part B (they include the Groth16
verifying key), so re-run this step at the end of Part B.

### A6. Test

```
cargo test -F prove -p risc0-zkvm --no-fail-fast
```

Use `--no-fail-fast`; otherwise a lib test failure hides the integration and doc tests.
Expected results with this backport: everything passes except
`host::server::exec::gdb::tests::end_to_end`, which needs the rzup `gdb` component
(`rzup install gdb`) and is unrelated to the circuit.

A useful focused test is
`cargo test -F prove -p risc0-zkvm -- --test-threads=1 test_recursion_lift_join_identity_p254_e2e`,
which exercises rv32im proving, lift, join and identity_p254.

### Part A gotchas

- **Misleading debug output.** Witness generation errors print
  `set(row: …, col: …, val: …) cur: …` from `rv32im-sys/kernels/cxx/buffers.h`, but the
  format string is `"%s(row: …"` with `name = "set"`, so grepping for the literal
  `set(row:` does not find it. Such a message means the preflight/witgen actually disagree
  about a column. It is not a sign of a stale build or of the external `r0vm` being used.
  The C++ exception it throws typically surfaces as a SIGSEGV.
- **Do not `AliasLayout!` lookup-argument columns onto data columns.** Arguments (for
  example `ArgU16`) are hoisted out of muxes, so their slots are shared by every mux arm.
  Aliasing an argument's `val` onto a data column in one arm clobbers that column in every
  other arm, and witness generation fails with conflicting `set`s. The alias saves nothing
  anyway: each argument slot already has its own `val` column.
- **`NondetU16Reg(x)` does not constrain its value to equal `x`.** Use `U16Reg(x)`, which
  adds `ret = val`.
- **`r0vm`.** Tests with `-F prove` prove in-process and do not use the installed `r0vm`.
  Tests that go through the client/server path use `RISC0_SERVER_PATH`, then an rzup `r0vm`
  matching the crate version, then `r0vm` on `PATH`. A released `r0vm` contains the old
  circuit, so point `RISC0_SERVER_PATH` at a locally built one for those tests.

---

## Part B — Groth16 circuit (`stark_verify.circom`) and trusted setup

### B1. Regenerate the circom circuit

```
cargo bootstrap -- verify --output ../risc0/
```

This copies three files into risc0. Two of them need fixing up afterwards:

1. `groth16_proof/groth16/stark_verify.circom`: the circuit (stored in git LFS; the diff
   only shows the LFS pointer).
2. `groth16_proof/groth16/risc0.circom`: **restore risc0's version**
   (`git checkout groth16_proof/groth16/risc0.circom`). risc0 carries an intentional
   divergence from zirgen in `inv_impl` (`var inv_res = inverse(in[0], P); out <-- inv_res;`
   instead of `out <-- inverse(in[0], P);`), added with the CUDA Groth16 prover in
   risc0 #3296, most likely for circom-witnesscalc's `build-circuit`. It does not change any
   constraint. Only port `risc0.circom` changes that you actually made in zirgen.
3. `risc0/groth16/src/seal_format.rs`: **delete it.** The bootstrap copies it to a stale
   path. The real file is `risc0/groth16/src/prove/seal_format.rs`. Diff the two to confirm
   the seal format did not change (only cosmetic differences are expected).

**The circom generator is nondeterministic.** Two builds of the same zirgen source produce
different `stark_verify.circom` files: different CSE/normalization choices, different
`iop[]` orderings, and different line counts (about 1M differing lines). Each is presumably
equivalent, but it makes review of a security change impractical. The procedure used here
keeps risc0's committed circuit and appends only the new lines:

```
# From the risc0 repo. The committed circom is in git LFS; get its object path:
git lfs ls-files -l | grep stark_verify.circom      # shows the committed oid
O=.git/lfs/objects/<aa>/<bb>/<full oid>              # committed circom
B=$(cd ../zirgen && bazelisk info --config bootstrap_linux_amd64 bazel-bin)/zirgen/circuit/verify/circom/stark_verify.circom
# (build it first: bazelisk build --config bootstrap_linux_amd64 //zirgen/circuit/verify/circom:circuit)

grep "out_tmp_bits_\|out_tmp_valid_\|out_tmp\[[0-9]*\] === 0" "$B" > checks.lines
N=$(grep -n "^  codeRoot <== comp_40.out;$" "$O" | cut -d: -f1)
{ head -n "$N" "$O"; cat checks.lines; tail -n +$((N+1)) "$O"; } > groth16_proof/groth16/stark_verify.circom

diff "$O" groth16_proof/groth16/stark_verify.circom | grep -c '^>'   # expect 72
diff "$O" groth16_proof/groth16/stark_verify.circom | grep -c '^<'   # expect 0
```

The generator emits the new lines immediately after `codeRoot <== …` and before the
`out[i] <== …` packing lines, in the same place in every build. Confirm the extracted lines
are contiguous in `$B`, and that the grep pattern matches exactly the lines the generator
change adds. For this backport it produced 72 lines (8 × 4 for even root words, 8 × 1 for
padding, 16 × 2 for SHA halves). The resulting file has sha256
`a0f6141b7b0ab35b9fb9566f96029b58f7189d705277744d72263590087a41e6`.

### B2. Sanity check the new circuit on an honest proof (cheap, do this first)

A ceremony takes about 20 minutes and a prover image several more, so check that honest
witnesses still satisfy the new constraints before running one. This check caught a first
version of the fix (16-bit checks on all 32 outputs) that rejected every honest proof,
because the control root words are 31-bit.

1. Get an honest `input.json`. Any Groth16 proving attempt writes one: set `RISC0_WORK_DIR`
   to a directory and run anything that proves Groth16 (for example
   `cargo xtask bootstrap-groth16 --test-receipt-only`). It writes `seal.r0` and
   `input.json` there before invoking the prover, even if the prover then fails.
2. Build only the witness generator stage of the prover image and run it:

```
cd groth16_proof
docker build -f docker/prover.Dockerfile --target witgen -t r0-witgen .
docker run --rm -v "$WORK_DIR":/mnt r0-witgen bash -c \
  "ulimit -s 16384; /usr/src/stark_verify_cpp/stark_verify /mnt/input.json /tmp/w.wtns && ls -la /tmp/w.wtns"
```

A failed constraint shows up as
`Failed assert in template/function … Followed trace of components: main.<component>`.
Only the honest path is checked this way. A negative test is hard to isolate, because
tampering with the seal trips the STARK verification asserts first.

### B3. Get a phase-1 powers of tau file

The stark_verify circuit has about 5.68M constraints, so it needs a 2^23 ptau.

**Gotcha:** the Hermez file referenced in `groth16_proof/README.md`
(`https://hermez.s3-eu-west-1.amazonaws.com/powersOfTau28_hez_final_23.ptau`) now returns
403, and so does the well-known mirror at
`https://storage.googleapis.com/zkevm/ptau/powersOfTau28_hez_final_23.ptau`. This backport's
test ceremony used the PSE Perpetual Powers of Tau file instead, which has the same format:

```
curl -f -o groth16_proof/groth16/pot23.ptau \
  https://pse-trusted-setup-ppot.s3.eu-central-1.amazonaws.com/pot28_0080/ppot_0080_23.ptau
# 9663769746 bytes, sha256 05085994c8aa34e4925585202e178e09fb8acc04e9565311751e8a04dec81cb1
```

Choosing the phase-1 transcript for a production ceremony is a deliberate decision. Verify
the file against its publisher's checksums before relying on it.

`pot23.ptau` is gitignored in `groth16_proof/groth16/`.

### B4. Build the ceremony image

**Gotcha:** the stock `groth16_proof/docker/ceremony.Dockerfile` no longer builds. It runs
`cargo install --path circom` without `--locked` on `rust:1.74.0`, so current crates.io
dependencies are pulled in, and they require edition 2024. It also used a different circom
commit (`e60c4ab`) and default optimization flags from the prover image, which risks a zkey
generated for a different r1cs than the prover loads.

Changes made to `ceremony.Dockerfile` for this backport:

- `FROM rust:1.84-bookworm AS dependencies` (was `rust:1.74.0`);
- circom commit `e410b0d5cd2948a15931df0bc50d79ce56fa8c32` (circom 2.2.2, the same as
  `prover.Dockerfile`) with `cargo install --locked --path circom`;
- `circom --r1cs --O2 stark_verify.circom` (the same flags as the prover image).

`prover.Dockerfile` also got `--locked` on its `cargo install` for the same bitrot reason.
Note that the final ceremony stage is `FROM node` (unpinned) with `snarkjs@0.7.5`.

```
cd groth16_proof
docker build -f docker/ceremony.Dockerfile . -t snark-ceremony
```

The build log reports the r1cs size (for this backport: 5,677,093 non-linear constraints,
5 public outputs).

### B5. Run the ceremony

```
cd groth16_proof
cp groth16/verifier.sol /somewhere/verifier.sol.orig   # optional, for comparison
docker run --rm -v "$(pwd)/groth16:/ceremony/groth16" snark-ceremony
```

This runs `scripts/run_ceremony.sh`: `snarkjs g16s`, then a single contribution with
fixed entropy `'Entropy'`, then a fixed beacon. **That is a test setup only.** A production
setup needs real independent contributions. It took about 21 minutes on a 24-core machine
and peaked well under 64 GB of RAM (`NODE_OPTIONS` allows 32 GB).

Outputs in `groth16/`: `stark_verify_000{0,1}.zkey` (intermediates, can be deleted),
`stark_verify_final.zkey`, `stark_verification_key.json`, and `verifier.sol` (overwritten).
The container runs as root, so fix ownership afterwards, for example
`docker run --rm -v "$(pwd)/groth16:/g" busybox chown -R "$(id -u):$(id -g)" /g`.

Then:

```
forge fmt groth16/verifier.sol       # the diff should touch only the verifying-key constants
sha256sum groth16/stark_verify_final.zkey
gzip -k groth16/stark_verify_final.zkey
```

### B6. Build the Groth16 prover image

`prover.Dockerfile` normally `ADD`s the proving key from S3 with a checksum. To keep the key
local (as here, before release), replace that with:

```
COPY groth16/stark_verify_final.zkey.gz groth16/stark_verify_final.zkey.gz
```

For release, upload `stark_verify_final.zkey.gz` to
`s3://risc0-artifacts/zkey/<yyyy-mm-dd.rev>/stark_verify_final.zkey.gz` and restore the
`ADD --checksum=sha256:<gz sha256> <url>` form.

```
cd groth16_proof
docker build -f docker/prover.Dockerfile . -t risczero/risc0-groth16-prover:v<yyyy-mm-dd.rev>
```

Update the tag in `risc0/groth16/src/prove/docker.rs` to match. This backport used
`v2026-09-16.1`, built locally and not pushed. For release, push the image to Docker Hub and
mirror the tag change in any other consumers of the image.

**Check that the ceremony and prover r1cs match.** The gnark converter in the prover image
combines the prover's r1cs with the ceremony's zkey. They must come from the same circom
build:

```
docker build -f docker/prover.Dockerfile --target circom -t r0-prover-circom .
c=$(docker create r0-prover-circom); docker cp $c:/usr/src/groth16/stark_verify.r1cs prover.r1cs; docker rm $c
c=$(docker create snark-ceremony);   docker cp $c:/ceremony/stark_verify.r1cs ceremony.r1cs; docker rm $c
sha256sum prover.r1cs ceremony.r1cs   # must be identical
```

### B7. Update the verifiers (Rust + Solidity) and the test receipt

Run from the risc0 root, with `forge` on `PATH`:

```
PATH=$PATH:$HOME/.foundry/bin RISC0_ETHEREUM_PATH=../risc0-ethereum cargo xtask bootstrap-groth16
```

This:

- copies `groth16_proof/groth16/verifier.sol` to
  `risc0-ethereum/contracts/src/groth16/Groth16Verifier.sol`;
- rewrites the verifying-key constants in `risc0/groth16/src/verifier.rs`;
- writes `ControlID.sol` (`CONTROL_ROOT`, `BN254_CONTROL_ID`);
- proves a Groth16 receipt with the Docker prover image and writes the test receipt
  Solidity file.

Gotchas:

- **The test receipt path is stale.** The xtask writes `contracts/test/TestReceipt.sol`, but
  risc0-ethereum v3.0.x uses versioned files (`TestReceiptV3_0.sol` and so on). For this
  backport, `SOLIDITY_TEST_RECEIPT_PATH` in `xtask/src/bootstrap_groth16.rs` was changed to
  `contracts/test/TestReceiptV3_1.sol`. `EndToEnd.t.sol` and `RiscZeroGroth16Verifier.t.sol`
  now import `./TestReceiptV3_1.sol`, and `TestReceiptV3_0.sol` stays unchanged for the
  deployment tests (`contracts/deployment-test/`), which check already-deployed verifiers.
  The file name is a placeholder; choose whatever the release versioning calls for.
- **Run it twice.** The xtask binary is compiled before it rewrites `verifier.rs`, so its
  first run computes the test receipt's selector (the first 4 bytes of the Groth16
  verifier-parameters digest) from the *old* verifying key. After the first run, run
  `cargo xtask bootstrap-groth16 --test-receipt-only` so the rebuilt xtask writes the
  correct selector. Check that the `SEAL` hex in the test receipt starts with the new
  selector (the first 8 hex digits of the new Groth16 verifier-parameters digest, see B8).
- The receipt proof uses the Docker prover image at the tag in `docker.rs`, so B6 must be
  done first.

### B8. Update the canaries again

The verifying key changes the Groth16 and composite verifier-parameter digests:

```
cargo test -F prove -p risc0-zkvm --lib -- is_stable stable_root
```

Update `groth16_receipt_verifier_parameters_is_stable` and
`composite_receipt_verifier_parameters_is_stable` (see A5). In risc0-ethereum, update
`testSelectorIsStable` in `contracts/test/RiscZeroGroth16Verifier.t.sol` to the new
selector.

### B9. risc0-ethereum

Beyond the files written in B7:

```
cd risc0-ethereum
git submodule update --init --recursive
cd contracts && forge test
```

With this backport all 35 tests pass. `EndToEnd` verifies the freshly generated Groth16
receipt against the new `Groth16Verifier.sol` and `ControlID.sol`.

### B10. CUDA Groth16 prover (optional, for `-F cuda`)

The CUDA prover reads the rzup component `risc0-groth16` version `0.1.0` from
`~/.risc0/extensions/v0.1.0-risc0-groth16/`. `cargo xtask setup-groth16` builds it, and it
downloads the **old** zkey from S3 only if `stark_verify_final.zkey` is missing. So
pre-place the new key:

```
mkdir -p ~/.risc0/extensions/v0.1.0-risc0-groth16
cp groth16_proof/groth16/stark_verify_final.zkey ~/.risc0/extensions/v0.1.0-risc0-groth16/
PATH=/usr/local/cuda/bin:$PATH cargo run --release -p xtask -F setup-groth16 -- \
  setup-groth16 $HOME/.risc0/extensions
```

This produces `stark_verify_graph.bin` (via circom-witnesscalc `build-circuit`),
`preprocessed_coeffs.bin` and `fuzzed_msm_results.bin`. `rzup show` then lists
`risc0-groth16 0.1.0`.

Gotchas:

- `nvcc` must be on `PATH`. If sppark was ever built without it, its build script caches a
  non-CUDA configuration and linking later fails with
  `undefined symbol: select_gpu(int)` / `gpu_props(int)` / `ngpus()`. Fix with
  `cargo clean -p sppark -p risc0-groth16-sys` (add `--release` for the xtask build) and
  rebuild with `nvcc` on `PATH`. This applies separately to the debug and release profiles.
- `setup-groth16` uses its argument as a work directory and leaves a `circom-witnesscalc`
  clone and `log_input_signals.txt` in it. Move those out of `~/.risc0/extensions/`
  afterwards.

### Part B tests

```
# Docker prover (run serially; each Groth16 prove is memory-heavy)
cargo test -F prove,docker -p risc0-zkvm --lib -- --test-threads=1 compress

# CUDA prover
PATH=/usr/local/cuda/bin:$PATH cargo test -F prove,cuda -p risc0-zkvm --lib -- \
  --test-threads=1 compress test_recursion_lift_join_identity_p254_e2e
```

The Groth16 tests are gated on `feature = "docker"` or `feature = "cuda"` and do **not** run
under plain `-F prove`. Running the Groth16 cases in parallel launches one prover container
per case and exhausted 64 GB of RAM; use `--test-threads=1`. When a test run is killed, its
prover containers keep running; clean them up with
`docker ps -q --filter ancestor=risczero/risc0-groth16-prover:<tag> | xargs -r docker kill`.

Results for this backport: Docker 12/12, CUDA 13/13.

---

## Part C — changing the Poseidon254 hash (S-box, constants, rounds)

This hash is implemented **four times**, and all four must change together. A mismatch is a
liveness failure, not a soundness one: proofs stop verifying and the tests below catch it
immediately.

1. `risc0/zkp/src/core/hash/poseidon_254/mod.rs` — `sbox()`, the CPU prover.
2. `risc0/sys/kernels/zkp/cuda/supra/poseidon254.cuh` — `sbox()`, the CUDA prover. Uses
   sppark's exponentiation operator, so the exponent is the literal in `fr ^= 7`.
3. `zirgen/zirgen/compiler/zkp/poseidon_254.cpp` — `sbox()`, used by zirgen's tests.
4. `zirgen/zirgen/circuit/verify/circom/include/risc0.circom` — `template sbox()`, the
   in-SNARK verifier.

**Do not bootstrap #4 into risc0 with `cargo bootstrap -- verify`.** That would also
overwrite risc0's intentionally divergent `inv_impl` (see B1). Apply the same edit by hand to
`risc0/groth16_proof/groth16/risc0.circom` and diff the two `sbox` templates to confirm they
match.

Changing round counts or constants (needed for α=5, not for α=7) additionally touches
`poseidon_254/consts.rs` (`ROUNDS_PARTIAL`, constants, MDS),
`poseidon254_constants.cuh`, zirgen's C++ constants, and in `risc0.circom` both the
`ROUND_CONSTANTS`/`MDS` tables and the hardcoded `mid[42]` / `rc[150]` sizes in
`template poseidon()`.

Note that `stark_verify.circom` itself does **not** change: the S-box lives in the
`risc0.circom` include, which the generated circuit only instantiates. The r1cs changes
anyway, so a new ceremony is still required.

### Steps

1. Patch all four implementations (plus risc0's `risc0.circom` copy).
2. Update the known-answer vector in
   `zirgen/zirgen/circuit/verify/circom/test/AB.cpp` (`TEST(SNARK, Poseidon)`). Compute the
   new value from the Rust implementation, which cross-checks Rust against C++:

   ```rust
   // risc0/zkp/tests/p254_tmp.rs, run with:
   //   cargo test -p risc0-zkp -F prove --test p254_tmp -- --nocapture
   // (poseidon_254 is gated behind the `prove` feature)
   let suite = Poseidon254HashSuite::new_suite();
   let data: Vec<BabyBearElem> = (0..5u32).map(BabyBearElem::new).collect();
   println!("KAT={}", suite.hashfn.hash_elem_slice(&data));
   ```

   Then `bazelisk test //zirgen/circuit/verify/circom/test:test --test_filter=SNARK.Poseidon`
   must pass. (Build it without `-c opt`; the test uses plain `assert`, which `NDEBUG`
   would strip. The other tests in that file shell out to `circom`, so they need it
   installed.)
3. `cargo xtask bootstrap` in risc0, which recomputes `BN254_IDENTITY_CONTROL_ID` (the
   identity ZKR hashed under the Poseidon254 suite). `ALLOWED_CONTROL_ROOT` is a Poseidon2
   BabyBear value and does **not** change.
4. Sanity-check the new circuit against an honest witness, as in B2 — but the seal must come
   from the **new** Rust prover, since the hash changed. Running
   `RISC0_WORK_DIR=<dir> cargo xtask bootstrap-groth16 --test-receipt-only` writes a fresh
   `input.json` before failing at the (still stale) prover image.
5. Run the whole of Part B from B4 onward: new ceremony, prover image, `bootstrap-groth16`
   twice, canaries, risc0-ethereum, CUDA component, tests. For the CUDA component, delete
   the stale `stark_verify_graph.bin`, `preprocessed_coeffs.bin` and
   `fuzzed_msm_results.bin` before re-running `setup-groth16`; the graph is derived from the
   circuit and the preprocessing from the zkey.

Downstream values that change: `BN254_IDENTITY_CONTROL_ID`, the verifying key and hence
`verifier.sol` / `verifier.rs` / `Groth16Verifier.sol`, the Groth16 selector, the Groth16 and
composite verifier-parameter canaries, `ControlID.sol`'s `BN254_CONTROL_ID`, and the test
receipt. The rv32im circuit, the lift/join control IDs and the control root are untouched.

---

## Release checklist (not done for the local test run)

- Run a real multi-party Groth16 ceremony (B5) with a chosen phase-1 transcript (B3), and
  redo B6–B10 with its zkey.
- Upload `recursion_zkr.zip` to `s3://risc0-artifacts/zkr/<SHA256_HASH>.zip` (A3).
- Upload `stark_verify_final.zkey.gz` to
  `s3://risc0-artifacts/zkey/<yyyy-mm-dd.rev>/stark_verify_final.zkey.gz` and restore the
  `ADD --checksum` line in `prover.Dockerfile` (B6).
- Push `risczero/risc0-groth16-prover:v<yyyy-mm-dd.rev>` and update every consumer of the
  tag.
- Publish the `risc0-groth16` rzup component built from the new zkey (B10). Until then,
  `rzup install risc0-groth16` fetches a component built from the old key.
- Point `xtask/src/setup_groth16.rs` at the new zkey URL (it hardcodes the 2024-05-17 key).
- Deploy the new Groth16 verifier. Its selector and control root differ from the deployed
  v3.0 verifier's.

## Gotcha summary

1. Witgen `set(row …)` errors print via `"%s(row: …"`; grep for `(row: ` instead.
2. Never `AliasLayout!` a lookup-argument column onto a data column.
3. `NondetU16Reg` does not constrain equality; use `U16Reg`.
4. `clang-format` missing means unformatted generated kernels and huge diffs.
5. `build.rs` downloads the ZKR zip from S3 only when the local hash does not match.
6. `cargo test --no-fail-fast`, or lib failures hide the other test targets.
7. The `verify` bootstrap overwrites risc0's intentionally divergent `risc0.circom` and
   drops a stale `seal_format.rs`.
8. The circom generator is nondeterministic; splice new lines into the committed circuit.
9. Check new circom constraints against an honest witness before running a ceremony.
10. The Hermez ptau URLs return 403; the PSE ppot file works.
11. `ceremony.Dockerfile` bitrot (Rust 1.74, no `--locked`); align its circom version and
    flags with the prover image and compare the r1cs hashes.
12. `bootstrap-groth16` writes a stale test receipt path and a stale selector on its first
    run after a ceremony; fix the path and run it again with `--test-receipt-only`.
13. Use a risc0-ethereum branch matching the risc0 release line, not `main`.
14. CUDA: `nvcc` on `PATH`, and clean sppark if it was ever built without it.
15. Run Groth16 tests with `--test-threads=1`; clean up orphaned prover containers.
16. The Poseidon254 hash has four implementations (Rust, CUDA, zirgen C++, circom) that must
    be changed together, and risc0 keeps its own `risc0.circom` copy to patch by hand.
