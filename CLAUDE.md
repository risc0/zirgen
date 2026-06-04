# CLAUDE.md — AI Agent Guidance for zirgen

## Commit Message Convention

All commits must follow this format on the subject line:

```
ZIR-NNN: Short description of the change
```

- `ZIR-NNN` is the Linear issue number (e.g. `ZIR-123`, `ZIR-42`).
- Use `ZIR-000` when no ticket exists or the ticket number is not yet known.
- The colon and space after the ticket number are required.
- The description should be a concise imperative-mood summary.

### Valid examples

```
ZIR-387: Fix for building circuits from out-of-tree
ZIR-123: Add new feature for user authentication
ZIR-000: Document commit message convention
```

### Invalid examples

```
fix something          # missing ZIR- prefix
ZIR-123 Add feature    # missing colon-space separator
ZIR-: Add feature      # missing issue number
```

### Exceptions

- **Merge commits** (subject starts with `Merge`) — exempt from the pattern.
- **Revert commits** (subject starts with `Revert`) — exempt from the pattern.

### Enforcement

The convention is currently enforced only by local git hooks (`commit-msg` and
`prepare-commit-msg`) that some contributors have installed in their `.git/hooks/`
directory. These hooks are **not tracked** in the repository and are **not
installed automatically** when you clone. There is no CI check for commit message
format. Contributors are asked to follow the convention manually, or to install
local hooks if they want automated enforcement.
