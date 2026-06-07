#!/bin/bash
# Install versioned git hooks by symlinking them into .git/hooks/.
# Run from the repository root: bash hooks/install.sh

set -e

REPO_ROOT=$(git rev-parse --show-toplevel)
HOOKS_DIR="$REPO_ROOT/hooks"
GIT_HOOKS_DIR="$REPO_ROOT/.git/hooks"

for hook in prepare-commit-msg commit-msg; do
    src="$HOOKS_DIR/$hook"
    dst="$GIT_HOOKS_DIR/$hook"

    if [ -e "$dst" ] && [ ! -L "$dst" ]; then
        echo "Backing up existing $hook to $hook.bak"
        mv "$dst" "$dst.bak"
    fi

    ln -sf "$src" "$dst"
    chmod +x "$src"
    echo "Installed $hook"
done

echo "Done. Hooks installed in $GIT_HOOKS_DIR"
