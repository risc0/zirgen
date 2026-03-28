# Commit Message Guidelines

This repository enforces a standardized commit message format to maintain consistency and traceability across the project.

## Required Format

All commit messages must follow this format:

```
ZIR-XXX: Brief description of changes

Optional detailed explanation of the changes, why they were made,
and any additional context that would be helpful for reviewers or
future maintainers.
```

### Components

1. **Issue Reference**: `ZIR-XXX` where XXX is the issue or task number
2. **Separator**: Colon followed by a space (`: `)
3. **Description**: Brief, clear description of what the change does

### Rules

- The first line (subject line) must start with `ZIR-XXX: `
- The issue number (XXX) must be one or more digits
- There must be a space after the colon
- The description should be concise but descriptive
- Use imperative mood (e.g., "Add feature" not "Added feature")
- Keep the first line under 72 characters when possible

## Examples

### Good Commit Messages

```
ZIR-123: Add user authentication feature
```

```
ZIR-456: Fix memory leak in connection pool

The connection pool was not properly releasing connections after
use, leading to resource exhaustion. This commit ensures all
connections are returned to the pool after operations complete.
```

```
ZIR-789: Update documentation for API endpoints

- Add examples for POST /users endpoint
- Clarify authentication requirements
- Fix typos in response schemas
```

### Bad Commit Messages

```
Add new feature
```
*Missing ZIR-XXX prefix*

```
ZIR-123 Add feature
```
*Missing colon after issue number*

```
ZIR-123:Add feature
```
*Missing space after colon*

```
ZIR-: Add feature
```
*Missing issue number*

## Exceptions

The following types of commits bypass the format check:

- **Merge commits**: Messages starting with "Merge"
- **Revert commits**: Messages starting with "Revert"

Git will automatically format these correctly when you use `git merge` or `git revert`.

## Setup

### Using the Commit Message Template

The repository includes a `.gitmessage` template file to help you format commits correctly. To use it:

```bash
git config commit.template .gitmessage
```

Now when you run `git commit` (without the `-m` flag), your editor will open with the template pre-filled.

### First-Time Setup

1. Configure the template (run once):
   ```bash
   git config commit.template .gitmessage
   ```

2. When making a commit:
   ```bash
   git add <files>
   git commit
   ```

3. Replace `ZIR-XXX` with your actual issue number and update the description

### Quick Commits

For quick commits, you can still use the `-m` flag:

```bash
git commit -m "ZIR-123: Add new feature"
```

## Validation

A `commit-msg` hook automatically validates your commit message format. If your message doesn't match the required format, the commit will be rejected with an error message explaining what's wrong.

### Common Errors

**Error: Commit message does not follow the required format**

This means your commit message doesn't start with `ZIR-XXX: `. Check that:
- You have the ZIR- prefix
- The issue number contains only digits
- You have a colon and space after the number
- You have a description after the colon

## Best Practices

1. **Reference the Issue**: Always include the correct ZIR issue number
2. **Be Descriptive**: The description should clearly explain what changed
3. **Use Imperative Mood**: Write as if giving a command ("Add", "Fix", "Update")
4. **Keep It Concise**: The first line should be brief but informative
5. **Add Context**: Use the body for detailed explanations when needed
6. **One Logical Change**: Each commit should represent one logical change

## Getting Help

If you have questions about commit message formatting or encounter issues with the commit hook:

1. Check this guide for examples
2. Review recent commits: `git log --oneline -10`
3. Consult with the team if unsure about issue numbering

## Additional Resources

- The commit message template: `.gitmessage`
- The validation hook: `.git/hooks/commit-msg`
- Recent commit history: `git log`
