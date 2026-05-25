#!/bin/bash
# One-time setup: point git at the tracked hooks directory and commit template.
# Run this once after cloning: bash .githooks/install.sh

set -euo pipefail

git config core.hooksPath .githooks
git config commit.template .githooks/commit-msg-template

echo "Hooks and commit template configured."
