#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"

# Point git at the tracked hooks directory
git config core.hooksPath .githooks

# Ensure hooks are executable
chmod +x "$REPO_ROOT/.githooks/commit-msg"
chmod +x "$REPO_ROOT/.githooks/prepare-commit-msg"

echo "Git hooks configured. Using .githooks/ for this repository."
