# Contributing to Zirgen

## Commit Message Format

All commits must follow the `ZIR-XXX:` format:

```
ZIR-123: Brief description of the change
```

- `ZIR-XXX` — the issue or task number (e.g. `ZIR-123`)
- Colon and space after the number
- A short, imperative description on the first line
- Optional body after a blank line

### Exceptions

Merge commits (`Merge ...`) and revert commits (`Revert ...`) are exempt from this requirement.

### Auto-normalization

The `prepare-commit-msg` hook automatically prepends `ZIR-000:` to any message that lacks a `ZIR-` prefix, so you won't be blocked — but you should replace `000` with the correct issue number before finalizing.

## Setting Up Hooks

Run the following once after cloning:

```sh
bash scripts/setup-hooks.sh
```

This sets `core.hooksPath = .githooks` in your local git config, activating the tracked hooks in `.githooks/`.

Alternatively, the `scripts/setup-git-hooks.sh` script copies the hooks directly into `.git/hooks/` and also installs a commit message template.

## Examples

```
ZIR-123: Add user authentication feature
ZIR-456: Fix memory leak in connection pool
ZIR-789: Update API documentation
```
