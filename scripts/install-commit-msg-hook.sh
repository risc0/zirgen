#!/bin/bash
# Installation script for commit-msg hook and commit message template
# This script sets up local git commit message validation

set -e

# Get the repository root directory
REPO_ROOT="$(git rev-parse --show-toplevel)"
HOOK_SOURCE="$REPO_ROOT/scripts/commit-msg"
HOOK_DEST="$REPO_ROOT/.git/hooks/commit-msg"
TEMPLATE_FILE="$REPO_ROOT/.git/commit-msg-template"

echo "Installing commit message validation hook..."

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "Error: Not in a git repository"
    exit 1
fi

# Check if the hook source exists in the repository
if [ ! -f "$HOOK_SOURCE" ]; then
    echo "Error: commit-msg hook not found at $HOOK_SOURCE"
    echo "The hook should be in the repository's scripts/ directory"
    exit 1
fi

# Copy the hook to .git/hooks/
echo "Copying hook from $HOOK_SOURCE to $HOOK_DEST"
cp "$HOOK_SOURCE" "$HOOK_DEST"
chmod +x "$HOOK_DEST"
echo "✓ Installed commit message hook to $HOOK_DEST"

# Create commit message template
echo "Creating commit message template..."
cat > "$TEMPLATE_FILE" << 'EOF'
ZIR-XXX:

# Please enter the commit message for your changes. Lines starting
# with '#' will be ignored, and an empty message aborts the commit.
#
# Commit Message Format:
#   ZIR-XXX: Brief description of the change
#
# Requirements:
#   - Start with ZIR- followed by issue number
#   - Add colon and space after issue number
#   - Write a clear, concise description in imperative mood
#
# Examples:
#   ZIR-123: Add user authentication feature
#   ZIR-456: Fix memory leak in connection pool
#   ZIR-789: Update API documentation
#
# You can add additional details on subsequent lines after a blank line:
#
# ZIR-123: Add user authentication feature
#
# Implements OAuth2 authentication with JWT tokens.
# Includes unit tests and integration tests.
EOF

echo "✓ Created commit message template at $TEMPLATE_FILE"

# Configure git to use the template
git config commit.template "$TEMPLATE_FILE"
echo "✓ Configured git to use commit message template"

echo ""
echo "Installation complete!"
echo ""
echo "The commit-msg hook will now validate your commit messages."
echo "When you run 'git commit' (without -m), your editor will open"
echo "with a template to help you write properly formatted messages."
echo ""
echo "Commit message format: ZIR-XXX: Description"
echo ""
echo "To test the hook, try:"
echo "  echo 'ZIR-123: Test message' | $HOOK_DEST /dev/stdin"
