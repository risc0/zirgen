# Contributing

## Commit message format

All commits must follow this format:

```
ZIR-NNN: Short description of the change
```

- `ZIR-NNN` is the Linear issue number (e.g. `ZIR-387`)
- The description must be non-empty
- `Merge` and `Revert` commits are exempt

### Setting up local hooks

Run the setup script once after cloning:

```sh
bash scripts/setup-hooks.sh
```

This configures git to use the shared hooks in `.githooks/`. The
`prepare-commit-msg` hook will automatically prepend `ZIR-000: ` to your
commit message when the prefix is absent, so you only need to replace `000`
with the real issue number. Merge, squash, and template commits are left
unchanged.

### CI enforcement

Every PR is validated by the `Commit Lint` GitHub Actions workflow. If any
commit on the branch fails the format check, the workflow will report which
SHA violated it.
