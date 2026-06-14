# Contributing

## Commit message format

Every commit must start with a Linear ticket prefix:

```
ZIR-NNN: Short imperative description
```

Examples:
- `ZIR-387: Fix building circuits from out-of-tree`
- `ZIR-000: Add placeholder change without a ticket`

**Exemptions:** `Merge ...` and `Revert ...` commits are exempt.

Use `ZIR-000` as a placeholder when a commit genuinely has no associated ticket.

## Setting up git hooks

The repository ships tracked hooks in `.githooks/`. Run the setup script once after cloning:

```sh
bash scripts/setup-hooks.sh
```

This sets `core.hooksPath = .githooks` for your local clone and makes the hooks executable. After setup:

- **prepare-commit-msg** — automatically prepends `ZIR-000: ` when you forget the prefix.
- **commit-msg** — rejects the commit if the final message still does not conform.

CI (`.github/workflows/commit-lint.yml`) enforces the same rule on every pull request.
