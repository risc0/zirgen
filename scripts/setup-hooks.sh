#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

chmod +x "$REPO_ROOT/.githooks/commit-msg"
chmod +x "$REPO_ROOT/.githooks/prepare-commit-msg"

git -C "$REPO_ROOT" config core.hooksPath .githooks

echo "Git hooks configured. Commit messages must follow: ZIR-NNN: description"
