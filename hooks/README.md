# Git Hooks

This directory contains git hooks to enforce project standards and conventions.

## Commit Message Hook

The `commit-msg` hook enforces a standardized commit message format for all commits in this repository.

### Required Format

All commit messages must follow this format:

```
ZIR-XXX: Description
```

Where:
- `ZIR-XXX` is the issue/task number (XXX must be one or more digits)
- `:` followed by a space
- `Description` is a brief description of the change

### Examples

Valid commit messages:
```
ZIR-123: Add new feature for user authentication
ZIR-456: Fix memory leak in connection pool
ZIR-42: Update documentation for API endpoints
```

Invalid commit messages:
```
Add new feature          # Missing ZIR-XXX prefix
ZIR-123 Add feature     # Missing colon
ZIR-: Add feature       # Missing issue number
ZIR-ABC: Add feature    # Issue number must be numeric
```

### Exceptions

The hook allows the following commit types without validation:
- Merge commits (messages starting with "Merge")
- Revert commits (messages starting with "Revert")

### Installation

To install the commit-msg hook, run:

```bash
cp hooks/commit-msg .git/hooks/commit-msg
chmod +x .git/hooks/commit-msg
```

Or use a symlink to keep it synchronized:

```bash
ln -sf ../../hooks/commit-msg .git/hooks/commit-msg
```

Alternatively, configure git to use the hooks directory directly:

```bash
git config core.hooksPath hooks/
```

### Testing

You can test the hook by running the test suite:

```bash
./test_hook.sh
```

This will validate that the hook correctly accepts valid commit messages and rejects invalid ones.
