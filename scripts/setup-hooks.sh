#!/bin/bash
# One-time setup: configure git to use the tracked hooks in .githooks/.
# Run once after cloning: bash scripts/setup-hooks.sh

set -euo pipefail

git config core.hooksPath .githooks

echo "Git hooks configured. Commit messages will now be validated against the ZIR-###: format."
