# Contributing to Zirgen

## Commit Message Format

All commits must follow this format:

```
ZIR-NNN: Short description of what changed
```

- **ZIR-NNN** — the Linear ticket number for this work (e.g. `ZIR-123`). Use `ZIR-000` as a placeholder if no ticket exists.
- **Colon and space** after the ticket number.
- **Description** — imperative mood, sentence case, no trailing period.

**Examples:**

```
ZIR-123: Add new feature for user authentication
ZIR-456: Fix memory leak in connection pool
ZIR-000: Update README with build instructions
```

Merge commits (`Merge ...`) and revert commits (`Revert ...`) are exempt from this rule.

Ticket numbers come from the [Linear](https://linear.app) project. If you do not have access, ask a maintainer to create a ticket or use `ZIR-000`.

## Git Hooks

The repository ships tracked hook scripts in `.githooks/`. Opt in by running the setup script once after cloning:

```sh
bash scripts/setup-hooks.sh
```

This sets `core.hooksPath` to `.githooks/` so git finds the tracked hooks. The `commit-msg` hook rejects commits whose subject line does not match the `ZIR-###:` convention (or an allowed bypass prefix). Merge commits and commits prefixed with `ci:` or `chore:` are exempt.
