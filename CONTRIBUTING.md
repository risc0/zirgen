# Contributing to Zirgen

## Commit Message Format

All commits must follow this format:

```
ZIR-XXX: Short description of the change
```

- `ZIR-XXX` is the Linear issue number (e.g. `ZIR-123`)
- The colon and space after the issue number are required
- The description should be a concise summary of the change

**Examples:**

```
ZIR-123: Add new feature for user authentication
ZIR-456: Fix memory leak in connection pool
```

**Exceptions:** Merge commits (`Merge ...`) and revert commits (`Revert ...`) are exempt from this requirement.

## Setting Up Git Hooks

The repository includes commit hooks in `.githooks/` that validate and auto-normalize commit messages. To activate them, run once after cloning:

```sh
./scripts/setup-hooks.sh
```

This configures git to use the tracked hooks via `core.hooksPath`. The hooks do the following:

- **`prepare-commit-msg`**: Automatically prepends `ZIR-000: ` to messages that lack a `ZIR-` prefix, so you can fix the issue number before committing.
- **`commit-msg`**: Validates the final message matches `ZIR-XXX: Description` and rejects it with a clear error if not.
