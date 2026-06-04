# CLAUDE.md — Zirgen Repository

## Commit Message Convention

All commits in this repository must follow the `ZIR-NNN: Description` format.

### Pattern

```
ZIR-<number>: <short imperative description>
```

- `ZIR-<number>` — the Linear ticket number (e.g. `ZIR-123`). Use `ZIR-000` when there is no associated ticket.
- `:` followed by a single space
- A short, imperative description (capitalise the first word; no trailing period)

### Valid examples

```
ZIR-123: Add Poseidon2 hash support to recursion circuit
ZIR-456: Fix memory leak in connection pool
ZIR-000: Update CI runner pinning
```

### Invalid examples

```
fix stuff              # missing ZIR- prefix
ZIR123: Description    # missing hyphen
ZIR-123 Description    # missing colon
zir-123: Description   # prefix must be uppercase
```

### Exceptions

Merge commits (`Merge …`) and revert commits (`Revert …`) are exempt from the format check and are allowed through as-is.

### Local enforcement

The `.git/hooks/commit-msg` hook rejects non-conforming messages at commit time.
The `.git/hooks/prepare-commit-msg` hook automatically prepends `ZIR-000: ` to any
message that lacks a `ZIR-` prefix, so bare messages are normalised rather than
rejected outright.

These hooks live in `.git/hooks/`, which is **not tracked by git** and is therefore
**not installed automatically on clone**. See [CONTRIBUTING.md](CONTRIBUTING.md) for
manual installation instructions.
