# Contributing to Zirgen

Thank you for your interest in contributing to Zirgen! This document provides guidelines and instructions for contributing to the project.

## Commit Message Format

This repository requires all commit messages to follow a specific format. **Commits that don't follow this format will be rejected by the commit-msg hook.**

### Required Format

```
ZIR-XXX: Brief description of changes
```

Where:
- `ZIR-XXX` is the issue/task number (XXX must be digits)
- Followed by a colon and space (`: `)
- Then a clear, concise description

### Examples

Good commit messages:
```
ZIR-123: Add user authentication feature
ZIR-456: Fix memory leak in connection pool
ZIR-789: Update documentation for API endpoints
```

Bad commit messages:
```
Add new feature                    # Missing ZIR-XXX prefix
ZIR-123 Add feature               # Missing colon
ZIR-123:Add feature              # Missing space after colon
```

### Setup Commit Template

To make commit formatting easier, configure git to use the provided template:

```bash
git config commit.template .gitmessage
```

Then use `git commit` (without `-m`) to open your editor with the template pre-filled.

### Detailed Guidelines

For comprehensive information about commit message formatting, including:
- Best practices
- Common errors and solutions
- Additional examples
- Troubleshooting

Please see **[COMMIT_GUIDELINES.md](COMMIT_GUIDELINES.md)**.

## General Workflow

1. **Fork and Clone**: Fork the repository and clone it locally
2. **Create a Branch**: Create a feature branch for your changes
3. **Make Changes**: Implement your changes with clear, logical commits
4. **Follow Format**: Ensure all commit messages follow the ZIR-XXX format
5. **Test**: Verify your changes work as expected
6. **Submit PR**: Open a pull request to the main branch

## Code Style

- Follow existing code style and conventions in the project
- Write clear, self-documenting code
- Add comments for complex logic
- Include tests for new functionality

## Questions or Issues?

If you encounter problems with the commit message format or have questions about contributing:

1. Review [COMMIT_GUIDELINES.md](COMMIT_GUIDELINES.md)
2. Check recent commits for examples: `git log --oneline -10`
3. Reach out to the team for clarification

Thank you for contributing to Zirgen!
