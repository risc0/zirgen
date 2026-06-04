# Contributing to zirgen

## Commit Message Format

All commits must follow this format:

```
ZIR-NNN: Short description of the change
```

Where `ZIR-NNN` is the Linear issue number associated with the work.

### Pattern

```
^ZIR-[0-9]+: .+
```

Only the **first line** (subject line) is validated. The body and trailers are
unrestricted.

### Valid examples

```
ZIR-387: Fix for building circuits from out-of-tree
ZIR-123: Add new feature for user authentication
ZIR-456: Fix memory leak in connection pool
ZIR-000: Document commit message convention
```

### Invalid examples

```
fix something             # missing ZIR- prefix
ZIR-123 Add feature       # missing colon-space separator
ZIR-: Add feature         # missing issue number
[ZIR-123] Add feature     # wrong bracket syntax
```

### No ticket? Use ZIR-000

When no Linear issue exists yet, or you are doing exploratory or infrastructure
work that does not map to a ticket, use `ZIR-000` as the prefix:

```
ZIR-000: Update CI runner configuration
```

### Exceptions

Two commit types are exempt from the `ZIR-NNN:` requirement:

| Type | Example |
|------|---------|
| Merge commits | `Merge pull request #123 from user/branch` |
| Revert commits | `Revert "ZIR-123: some change"` |

Git generates these subject lines automatically; do not modify them to add a
`ZIR-` prefix.

## Enforcement

The `ZIR-NNN:` convention is a project standard, but it is **not enforced
automatically** for new contributors:

- The `.git/hooks/` directory is not tracked by git and is not included in a
  clone. Any hooks that team members have locally were installed manually and
  are not shared through the repository.
- CI does not currently validate commit message format.

Contributors are asked to follow the convention manually. If you want local
enforcement, you can write your own `commit-msg` hook that validates the
pattern `^ZIR-[0-9]+: .+` against the commit subject line.
