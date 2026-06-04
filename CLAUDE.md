# CLAUDE.md — Zirgen

## Commit message convention

All commit subject lines **must** follow the `ZIR-NNN: Description` format:

```
ZIR-NNN: Short imperative description
```

- `ZIR-NNN` — Linear ticket number (e.g. `ZIR-123`). Use `ZIR-000` when no ticket exists.
- Colon + space after the number.
- Description: imperative mood, sentence case, no trailing period.

**Valid:**
```
ZIR-387: Fix circuit builder for out-of-tree targets
ZIR-000: Update README with sccache instructions
```

**Invalid:**
```
fix a bug
ZIR123 Add feature
ZIR-: forgot the number
```

**Exceptions:** Merge commits (`Merge …`) and revert commits (`Revert …`) are exempt.

The `prepare-commit-msg` hook auto-prefixes `ZIR-000: ` when your message has no `ZIR-` prefix. The `commit-msg` hook rejects messages that still don't conform after that normalization.

See `CONTRIBUTING.md` for full contributor guidance and `.git/hooks/` for the hook implementations.
