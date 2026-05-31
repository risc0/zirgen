#!/bin/bash
# Configure git to use the tracked hooks in .githooks/
# Run this once after cloning the repository.

set -e

REPO_ROOT=$(git rev-parse --show-toplevel)

git config core.hooksPath .githooks

echo "Git hooks configured. Hooks in ${REPO_ROOT}/.githooks/ are now active."
