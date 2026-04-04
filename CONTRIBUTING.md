# Contributing to Zirgen

Thank you for contributing to Zirgen! This document provides guidelines for contributing to the project.

## Commit Message Format

All commit messages must follow a standardized format to maintain consistency and traceability across the project.

### Format

```
ZIR-XXX: Brief description of the change

Optional detailed explanation of what changed and why.
Can span multiple lines if needed.

Additional context, references, or notes can go here.
```

### Requirements

1. **Subject Line (First Line)**:
   - Must start with `ZIR-` followed by the issue/task number
   - Followed by a colon and space `: `
   - Then a brief, descriptive summary
   - Example: `ZIR-123: Add user authentication feature`

2. **Issue Number**:
   - Must be a valid issue/task number (one or more digits)
   - References the corresponding issue in your issue tracking system

3. **Description**:
   - Should be clear and concise
   - Written in imperative mood (e.g., "Add feature" not "Added feature")
   - Capitalize the first letter after the colon

### Examples

#### Good Commit Messages

```
ZIR-123: Add user authentication feature

Implements OAuth2 authentication with JWT tokens.
Includes unit tests and integration tests.
```

```
ZIR-456: Fix memory leak in connection pool

The connection pool was not properly closing idle connections.
This fix ensures connections are closed after the timeout period.
```

```
ZIR-789: Update documentation for API endpoints
```

#### Bad Commit Messages

```
Fix bug
```

```
ZIR-123 Add feature
```
(Missing colon after issue number)

```
Add new feature
```
(Missing ZIR prefix)

### Special Cases

The commit message hook automatically allows the following types of commits without format validation:

- **Merge commits**: Messages starting with "Merge"
- **Revert commits**: Messages starting with "Revert"
- **Empty commits**: Commits with no message or only comments

## Setting Up Commit Message Validation

To enforce the commit message format locally, install the git commit-msg hook:

### Installation

Run the installation script from the repository root:

```bash
./scripts/install-commit-msg-hook.sh
```

This will:
1. Copy the commit-msg hook to your `.git/hooks/` directory
2. Make it executable
3. Set up a commit message template for your convenience

### Manual Installation

If you prefer to install manually:

1. Copy the hook:
   ```bash
   cp scripts/commit-msg .git/hooks/commit-msg
   chmod +x .git/hooks/commit-msg
   ```

2. Set up the commit message template (optional):
   ```bash
   git config commit.template .git/commit-msg-template
   ```

### Using the Commit Message Template

Once configured, when you run `git commit` without the `-m` flag, your editor will open with a pre-filled template:

```
ZIR-XXX:

# Please enter the commit message for your changes. Lines starting
# with '#' will be ignored, and an empty message aborts the commit.
#
# Format: ZIR-XXX: Brief description
#
# Example:
#   ZIR-123: Add user authentication feature
```

Simply replace `XXX` with your issue number and add your description.

## Testing Your Commit Messages

To test if your commit message is valid without actually committing:

```bash
echo "ZIR-123: Test commit message" | .git/hooks/commit-msg /dev/stdin
```

If the message is valid, you'll see no output. If invalid, you'll see an error message.

## Questions?

If you have questions about commit message formatting or other contribution guidelines, please open an issue or reach out to the maintainers.
