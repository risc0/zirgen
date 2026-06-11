#!/bin/bash
# One-time setup: configure git to use the tracked hooks in .githooks/.
# Run once after cloning: bash scripts/setup-hooks.sh

set -euo pipefail

git config core.hooksPath .githooks
git config commit.template .githooks/commit-msg-template

echo "Git hooks and commit template configured. Commit messages will now be validated against the ZIR-###: format."
