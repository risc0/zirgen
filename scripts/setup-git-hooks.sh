#!/bin/bash
# Install tracked git hooks into .git/hooks/.
# Idempotent: safe to run multiple times.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HOOKS_SRC="$REPO_ROOT/hooks"
HOOKS_DST="$REPO_ROOT/.git/hooks"

if [ ! -d "$HOOKS_DST" ]; then
    echo "ERROR: .git/hooks directory not found. Are you inside a git repository?"
    exit 1
fi

for hook in commit-msg prepare-commit-msg; do
    src="$HOOKS_SRC/$hook"
    dst="$HOOKS_DST/$hook"

    if [ ! -f "$src" ]; then
        echo "WARNING: $src not found, skipping."
        continue
    fi

    ln -sf "$src" "$dst"
    echo "Installed $hook -> $dst"
done

echo "Git hooks installed successfully."
