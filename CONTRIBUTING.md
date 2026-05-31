# Contributing to Zirgen

## Commit Message Format

All commits must follow the format:

```
ZIR-XXX: Short description of the change
```

- `ZIR-XXX` is the issue or task number (e.g. `ZIR-123`)
- The description follows after a colon and space
- Keep the subject line under 72 characters

**Examples:**
```
ZIR-123: Add user authentication middleware
ZIR-456: Fix memory leak in connection pool
ZIR-000: Miscellaneous or un-ticketed change
```

**Exceptions:** Merge commits (`Merge ...`) and revert commits (`Revert ...`) are exempt from this requirement.

## Setting Up Git Hooks

Run the setup script once after cloning to activate commit message validation and auto-normalization:

```sh
bash scripts/setup-hooks.sh
```

This configures git to use the hooks in `.githooks/`:

- **`commit-msg`** — rejects commits that don't follow the `ZIR-XXX:` format
- **`prepare-commit-msg`** — prepends `ZIR-000: ` automatically when a message lacks the prefix, so you can fix the ticket number before finalizing
