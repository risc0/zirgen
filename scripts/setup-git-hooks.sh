#!/bin/bash
# Setup script for git hooks and commit message template

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
HOOKS_DIR="$REPO_ROOT/hooks"
GIT_HOOKS_DIR="$REPO_ROOT/.git/hooks"

echo "Setting up git hooks and commit message template..."

# Check if we're in a git repository
if [ ! -d "$REPO_ROOT/.git" ]; then
    echo "Error: Not in a git repository"
    exit 1
fi

# Copy commit-msg hook (validates ZIR-XXX: format)
echo "Installing commit-msg hook..."
cp "$HOOKS_DIR/commit-msg" "$GIT_HOOKS_DIR/commit-msg"
chmod +x "$GIT_HOOKS_DIR/commit-msg"

# Copy prepare-commit-msg hook (auto-prepends ZIR-000: when prefix is missing)
echo "Installing prepare-commit-msg hook..."
cp "$HOOKS_DIR/prepare-commit-msg" "$GIT_HOOKS_DIR/prepare-commit-msg"
chmod +x "$GIT_HOOKS_DIR/prepare-commit-msg"

# Copy commit message template
echo "Installing commit message template..."
cp "$HOOKS_DIR/commit-msg-template" "$REPO_ROOT/.git/commit-msg-template"

# Configure git to use the template
echo "Configuring git to use commit message template..."
git config commit.template "$REPO_ROOT/.git/commit-msg-template"

echo ""
echo "Git hooks setup complete!"
echo ""
echo "The following have been configured:"
echo "  - prepare-commit-msg hook (auto-prepends ZIR-000: to non-conforming messages)"
echo "  - commit-msg hook (validates ZIR-XXX: format)"
echo "  - commit message template"
echo ""
echo "You can now make commits following the standardized format."
echo "See CONTRIBUTING.md for more details."
