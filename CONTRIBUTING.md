# Contributing to Zirgen

## Commit message format

Every commit must follow this pattern:

```
ZIR-<number>: <short imperative description>
```

| Part | Requirement |
|---|---|
| `ZIR-<number>` | Linear ticket number. Use `ZIR-000` if there is no ticket. |
| `:` | Literal colon, followed by exactly one space. |
| Description | Imperative mood, capitalised, no trailing period. |

### Valid

```
ZIR-387: Fix build for out-of-tree circuits
ZIR-000: Tidy CI runner pinning
```

### Invalid

```
fix stuff                  # no ZIR- prefix
ZIR123: description        # missing hyphen
ZIR-123 description        # missing colon+space
zir-123: description       # prefix must be uppercase ZIR
```

### Exceptions

- **Merge commits** — messages starting with `Merge` are allowed through unchanged.
- **Revert commits** — messages starting with `Revert` are allowed through unchanged.

---

## Local git hooks

Two hooks enforce the convention locally:

| Hook | Purpose |
|---|---|
| `commit-msg` | Rejects any non-conforming subject line at commit time. |
| `prepare-commit-msg` | Auto-prepends `ZIR-000: ` to messages that lack a `ZIR-` prefix, so bare messages are normalised rather than immediately rejected. |

### Installing the hooks

The `.git/hooks/` directory is **not tracked by git** and is therefore **not copied
when you clone the repository**. You must install the hooks manually after cloning.

If the repository ships a tracked hooks directory (e.g. `hooks/` or `.githooks/`),
set git's hook path:

```sh
git config core.hooksPath hooks
```

Otherwise, copy or symlink the hook scripts into your local `.git/hooks/` directory
and make them executable:

```sh
cp hooks/commit-msg .git/hooks/commit-msg
cp hooks/prepare-commit-msg .git/hooks/prepare-commit-msg
chmod +x .git/hooks/commit-msg .git/hooks/prepare-commit-msg
```

> **Note:** Until a tracked hooks directory is added to this repository, the hooks
> exist only in the local `.git/hooks/` of contributors who have set them up
> manually. CI is the authoritative enforcement point for commit format.
