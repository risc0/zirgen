# Contributing to Zirgen

Thank you for your interest in contributing to Zirgen! This document provides guidelines and standards for contributing to the project.

## Commit Message Format

All commit messages in this repository must follow a standardized format to ensure consistency and traceability.

### Required Format

```
ZIR-XXX: Brief description of the change
```

Where:
- `ZIR-XXX` is the issue or task number (XXX must be one or more digits)
- A colon and space (`: `) follow the issue number
- A brief, clear description in imperative mood follows

### Examples

**Good commit messages:**
```
ZIR-123: Add user authentication feature
ZIR-456: Fix memory leak in connection pool
ZIR-789: Update API documentation for new endpoints
```

**Bad commit messages:**
```
Fix bug                          # Missing ZIR-XXX prefix
ZIR-123 Add feature              # Missing colon
ZIR-: Update docs                # Missing issue number
Added new feature                # Use imperative mood, not past tense
```

### Extended Commit Messages

You can add additional details after a blank line:

```
ZIR-123: Add user authentication feature

Implements OAuth2 authentication with JWT tokens.
Includes unit tests and integration tests.
Addresses security requirements from design review.
```

### Automatic Validation

The repository includes a git hook that automatically validates commit messages. If your commit message doesn't follow the format, the commit will be rejected with a helpful error message.

### Setup

The commit message template and validation hook are automatically configured when you clone the repository. If you need to set them up manually:

```bash
# Set the commit message template
git config commit.template .git/commit-msg-template

# Ensure the commit-msg hook is executable
chmod +x .git/hooks/commit-msg
```

### Exceptions

The following types of commits are exempt from validation:
- Merge commits (starting with "Merge")
- Revert commits (starting with "Revert")

For more detailed information about commit message guidelines, see [docs/commit-guidelines.md](docs/commit-guidelines.md).

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes following the commit message format
3. Ensure all tests pass
4. Submit a pull request to `main`
5. Address any review feedback

## Questions or Issues

If you have questions about contributing or encounter issues with the commit message format, please open an issue in the repository.
