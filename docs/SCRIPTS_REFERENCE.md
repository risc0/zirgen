# Scripts Reference

This guide documents the development scripts available in the Zirgen repository.

## Table of Contents

1. [Overview](#overview)
2. [copy_codegen_to_risc0.sh](#copy_codegen_to_risc0sh)
3. [setup-git-hooks.sh](#setup-git-hookssh)

## Overview

Development scripts are located in the `scripts/` directory:

```
scripts/
├── copy_codegen_to_risc0.sh    # Build and copy codegen to RISC Zero
└── setup-git-hooks.sh          # Install git hooks
```

## copy_codegen_to_risc0.sh

**Purpose**: Build Zirgen codegen and copy generated files into the RISC Zero repository, then optionally run bootstrap and benchmarks.

This is the main integration script for testing Zirgen changes with RISC Zero.

### Usage

```bash
./scripts/copy_codegen_to_risc0.sh [OPTIONS]
```

### Workflow

The script performs these steps:

1. **Build codegen** (unless `--copy-only`)
   - Builds `//zirgen/circuit/rv32im/v2/dsl:codegen`
   - Builds `//zirgen/circuit/keccak:codegen` (unless `--no-keccak`)

2. **Copy generated files** to RISC Zero repository
   - Rust files → `risc0/circuit/rv32im/src/zirgen/`
   - C++ files → `risc0/circuit/rv32im-sys/kernels/cxx/`
   - CUDA files → `risc0/circuit/rv32im-sys/kernels/cuda/`
   - Same for Keccak circuit files

3. **Rebuild recursion_zkr.zip** (unless `--no-rebuild-zkr`)
   - Builds `//zirgen/circuit/predicates:recursion_zkr`
   - Copies to `risc0/circuit/recursion/src/recursion_zkr.zip`
   - Updates SHA256 hash in `build.rs`

4. **Run bootstrap** (unless `--no-bootstrap`)
   - Executes `cargo xtask bootstrap` in RISC Zero repo
   - Regenerates `control_id.rs` with new circuit digests

5. **Run datasheet benchmark** (unless `--no-datasheet`)
   - Executes datasheet example with prove feature
   - Optionally with CUDA (`--cuda`)

### Options

#### Build Control

**`--copy-only`**
Skip building codegen; just copy from existing output directory.

Example:
```bash
./scripts/copy_codegen_to_risc0.sh --copy-only
```

**`--no-keccak`**
Skip building and copying Keccak circuit.

Example:
```bash
./scripts/copy_codegen_to_risc0.sh --no-keccak
```

**`--low-memory`**
Serialize build jobs to reduce RAM usage. Sets `--jobs=1` for Bazel and `-j 1` for Cargo.

Example:
```bash
./scripts/copy_codegen_to_risc0.sh --low-memory
```

Use this if builds are failing with OOM errors.

#### Path Configuration

**`--codegen-dir DIR`**
Use custom directory for rv32im codegen output instead of default `bazel-bin` location.

Example:
```bash
./scripts/copy_codegen_to_risc0.sh --codegen-dir ./my-output
```

**`--keccak-codegen-dir DIR`**
Use custom directory for Keccak codegen output.

Example:
```bash
./scripts/copy_codegen_to_risc0.sh --keccak-codegen-dir ./keccak-output
```

**`--risc0-root DIR`**
Override RISC Zero repository path (default: `risc0_repo/` in Zirgen root).

Example:
```bash
./scripts/copy_codegen_to_risc0.sh --risc0-root ~/projects/risc0
```

#### Execution Control

**`--no-rebuild-zkr`**
Skip rebuilding `recursion_zkr.zip`.

Example:
```bash
./scripts/copy_codegen_to_risc0.sh --no-rebuild-zkr
```

Note: `--copy-only` implies `--no-rebuild-zkr` unless you explicitly pass `--rebuild-zkr`.

**`--rebuild-zkr`**
Force rebuild of `recursion_zkr.zip` even with `--copy-only`.

Example:
```bash
./scripts/copy_codegen_to_risc0.sh --copy-only --rebuild-zkr
```

**`--no-bootstrap`**
Skip running `cargo xtask bootstrap`.

Example:
```bash
./scripts/copy_codegen_to_risc0.sh --no-bootstrap
```

**`--no-datasheet`**
Skip running the datasheet benchmark.

Example:
```bash
./scripts/copy_codegen_to_risc0.sh --no-datasheet
```

**`--cuda`**
Enable CUDA feature for datasheet benchmark (requires NVIDIA GPU and nvcc).

Example:
```bash
./scripts/copy_codegen_to_risc0.sh --cuda
```

#### Datasheet Configuration

**`--max-po2 N`**
Set maximum po2 for composite benchmarks (default: 20, range: 15-24).

Example:
```bash
./scripts/copy_codegen_to_risc0.sh --max-po2 22
```

**`--json FILE`**
Write datasheet results to JSON file.

Example:
```bash
./scripts/copy_codegen_to_risc0.sh --json results.json
```

**`--datasheet-cmd CMD`**
Run only specific datasheet subcommand.

Valid commands:
- `execute`
- `composite`
- `lift`
- `join`
- `succinct`
- `identity`
- `bigint2`

Example:
```bash
./scripts/copy_codegen_to_risc0.sh --datasheet-cmd composite
```

### Environment Variables

**`RISC0_ROOT`**
Override RISC Zero repository path (can also use `--risc0-root` flag).

Example:
```bash
RISC0_ROOT=~/risc0 ./scripts/copy_codegen_to_risc0.sh
```

### Generated Files

#### RV32IM v2 Circuit

The script copies these files (with `SPLIT_VALIDITY=4`, `SPLIT_STEP=1`):

**Rust files**:
- `defs.rs.inc`
- `info.rs`
- `layout.rs.inc`
- `poly_ext.rs`
- `taps.rs`
- `taps_provider_impl.rs.inc`
- `types.rs.inc`

**C++ files**:
- `defs.cpp.inc`
- `layout.cpp.inc`
- `layout.h.inc`
- `rust_poly_fp_0.cpp` through `rust_poly_fp_3.cpp` (4 files)
- `steps.cpp`
- `steps.h`
- `types.h.inc`

**CUDA files**:
- `defs.cu.inc`
- `eval_check_0.cu` through `eval_check_3.cu` (4 files)
- `eval_check.cuh`
- `layout.cu.inc`
- `layout.cuh.inc`
- `steps.cu`
- `steps.cuh`
- `types.cuh.inc`

#### Keccak Circuit

With `SPLIT_VALIDITY=5`, `SPLIT_STEP=16`:

**Rust files**: Same as RV32IM

**C++ files**:
- Standard files (defs, layout, types)
- `rust_poly_fp_0.cpp` through `rust_poly_fp_4.cpp` (5 files)
- `steps_0.cpp` through `steps_15.cpp` (16 files)
- `steps.h`

**CUDA files**:
- Standard files (defs, layout, types)
- `eval_check_0.cu` through `eval_check_4.cu` (5 files)
- `steps_0.cu` through `steps_15.cu` (16 files)
- `steps.cuh`

### Common Workflows

#### Full CI Run

Build everything, copy, bootstrap, and test:

```bash
./scripts/copy_codegen_to_risc0.sh
```

#### Quick Iteration

Copy already-built files and skip tests:

```bash
./scripts/copy_codegen_to_risc0.sh --copy-only --no-datasheet
```

#### CUDA Development

Build and test with CUDA:

```bash
./scripts/copy_codegen_to_risc0.sh --cuda
```

#### Low Memory System

Build serially to avoid OOM:

```bash
./scripts/copy_codegen_to_risc0.sh --low-memory
```

#### Test Specific Benchmark

Run only composite benchmark:

```bash
./scripts/copy_codegen_to_risc0.sh --datasheet-cmd composite --max-po2 20
```

#### RV32IM Only

Skip Keccak to save time:

```bash
./scripts/copy_codegen_to_risc0.sh --no-keccak
```

### Troubleshooting

#### Script Fails to Find RISC Zero

```
ERROR: Cannot find risc0 repo at /path/to/risc0_repo
```

Solution: Set `RISC0_ROOT` or use `--risc0-root`:

```bash
./scripts/copy_codegen_to_risc0.sh --risc0-root ~/my-risc0
```

#### Missing Generated Files

```
WARNING: bazel-bin/zirgen/circuit/rv32im/v2/dsl/eval_check_3.cu not found
```

Solution: Ensure Bazel build completed successfully. Check BUILD.bazel has correct `outs` list for split counts.

#### Out of Memory During Build

```
c++: fatal error: Killed signal terminated program cc1plus
```

Solution: Use `--low-memory` flag:

```bash
./scripts/copy_codegen_to_risc0.sh --low-memory
```

For CUDA compilation in RISC Zero:
```bash
cd risc0_repo
cargo build -F cuda -j 1
```

#### Bootstrap Failures

If bootstrap fails, try manually:

```bash
cd risc0_repo
cargo clean
cargo xtask bootstrap
```

#### Datasheet Test Failures

Known lift failures can be accepted:

```bash
ACCEPT_LIFT_FAILURES=1 ./scripts/copy_codegen_to_risc0.sh
```

See [Testing in RISC Zero](TESTING_IN_RISC0.md) for details on lift failures and ZKR zip limitations.

## setup-git-hooks.sh

**Purpose**: Install git hooks and commit message template for the Zirgen repository.

### Usage

```bash
./scripts/setup-git-hooks.sh
```

Run once after cloning the repository.

### What It Does

1. **Installs commit-msg hook**
   - Copies `hooks/commit-msg` to `.git/hooks/commit-msg`
   - Makes it executable
   - Validates commit messages follow `ZIR-XXX: Description` format

2. **Installs commit message template**
   - Copies `hooks/commit-msg-template` to `.git/commit-msg-template`
   - Configures git to use template

3. **Configures git**
   - Sets `commit.template` to point to template file

### Requirements

- Must be run from within a git repository
- Requires bash shell

### Commit Message Format

After setup, commits must follow this format:

```
ZIR-XXX: Brief description of change

Optional longer description explaining why this change was made
and any relevant context.

Co-Authored-By: Name <email@example.com>
```

Where `XXX` is:
- Issue number (e.g., `ZIR-387`)
- `000` for minor changes without an issue

The hook validates:
- First line starts with `ZIR-` followed by number
- First line has a colon and description
- Description is meaningful (not just "fix" or "update")

### Examples

Valid commit messages:

```
ZIR-387: Fix for building circuits from out-of-tree

Adds support for specifying include directories when building
circuits that are not in the main zirgen repository.
```

```
ZIR-000: Update README with new documentation links
```

Invalid commit messages:

```
Fix bug                    # Missing ZIR-XXX prefix
ZIR-123 Fix bug           # Missing colon
ZIR-123:                  # No description
ZIR-: Fix bug             # No issue number
```

### Troubleshooting

#### Hook Not Running

Check hook is executable:

```bash
ls -l .git/hooks/commit-msg
chmod +x .git/hooks/commit-msg
```

#### Hook Rejects Valid Message

Verify format exactly matches:

```bash
# Correct format
git commit -m "ZIR-000: Description here"

# Not this
git commit -m "ZIR-000 Description"
git commit -m "zir-000: Description"
```

#### Template Not Showing

Verify git configuration:

```bash
git config commit.template
# Should output: /path/to/zirgen/.git/commit-msg-template
```

Reload configuration:

```bash
git config --unset commit.template
./scripts/setup-git-hooks.sh
```

## Related Documentation

- [CONTRIBUTING.md](../CONTRIBUTING.md) - Contribution guidelines
- [Bazel Build Guide](BAZEL_BUILD_GUIDE.md) - Build system reference
- [Testing in RISC Zero](TESTING_IN_RISC0.md) - Integration testing
- [Architecture](../ARCHITECTURE.md) - Compiler architecture
