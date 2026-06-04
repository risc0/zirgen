# Contributing to Zirgen

## Commit message format

All commits must follow this format:

```
ZIR-NNN: Short imperative description
```

| Part | Rule |
|------|------|
| `ZIR-NNN` | Linear ticket number. Use `ZIR-000` when no ticket exists. |
| `: ` | Literal colon + space. |
| Description | Imperative mood, sentence case, no trailing period. |

### Valid examples

```
ZIR-387: Fix circuit builder for out-of-tree targets
ZIR-123: Add poseidon2 transcript hashing to keccak circuit
ZIR-000: Update README with sccache build instructions
```

### Invalid examples

```
fix a bug                     # missing ZIR- prefix
ZIR123: Add feature           # missing hyphen
ZIR-: forgot the number       # no number
ZIR-45 Missing colon          # missing ': '
```

### Exceptions

Commits whose subject starts with `Merge` or `Revert` are exempt from the format rule.

### Ticket-unknown fallback

When you don't have a ticket number yet, use `ZIR-000` as a placeholder:

```
ZIR-000: WIP — prototype keccak optimization
```

## Git hooks

The repository includes two hooks in `.git/hooks/`:

| Hook | Behavior |
|------|----------|
| `prepare-commit-msg` | Auto-prepends `ZIR-000: ` to messages that lack a `ZIR-` prefix. Runs before `commit-msg`. Skips merge and squash sources. |
| `commit-msg` | Rejects the commit if the first non-comment line does not match `^ZIR-[0-9]+: .+`. Exits 0 for Merge and Revert commits. |

These hooks are installed automatically when you clone the repository. If they are not active, check that `.git/hooks/commit-msg` and `.git/hooks/prepare-commit-msg` are executable (`chmod +x`).

## Development workflow

1. Identify or create a Linear ticket for the work (`ZIR-NNN`).
2. Commit with the `ZIR-NNN: Description` subject line.
3. Open a PR against `main`.

For build instructions, see `README.md` and `AGENTS.md`.
