#!/bin/bash
# Installs tracked .githooks/ scripts into .git/hooks/ as symlinks.
# Idempotent: safe to run multiple times.

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
HOOKS_SRC="$REPO_ROOT/.githooks"
HOOKS_DEST="$REPO_ROOT/.git/hooks"

if [ ! -d "$HOOKS_SRC" ]; then
    echo "ERROR: .githooks/ directory not found at $HOOKS_SRC" >&2
    exit 1
fi

for hook in "$HOOKS_SRC"/*; do
    name="$(basename "$hook")"
    dest="$HOOKS_DEST/$name"

    if [ -L "$dest" ] && [ "$(readlink "$dest")" = "$hook" ]; then
        echo "  already installed: $name"
        continue
    fi

    if [ -e "$dest" ] && [ ! -L "$dest" ]; then
        echo "  backing up existing $name -> $name.bak"
        mv "$dest" "$dest.bak"
    fi

    ln -sf "$hook" "$dest"
    chmod +x "$hook"
    echo "  installed: $name"
done

echo "Git hooks installed successfully."
