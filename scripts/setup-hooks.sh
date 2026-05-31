#!/bin/bash
# Configure git to use the tracked hooks in .githooks/.
set -e

git config core.hooksPath .githooks
echo "Git hooks configured. Commit messages will be validated against ZIR-XXX: format."
