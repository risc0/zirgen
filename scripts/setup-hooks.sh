#!/bin/bash
# Activate the tracked git hooks in .githooks/ via core.hooksPath

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ ! -d "$REPO_ROOT/.git" ]; then
    echo "Error: Not in a git repository"
    exit 1
fi

git -C "$REPO_ROOT" config core.hooksPath .githooks
echo "Git hooks configured: core.hooksPath = .githooks"
echo "See CONTRIBUTING.md for the commit message format."
