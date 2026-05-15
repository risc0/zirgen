#!/bin/bash
# Normalize a commit message file to enforce ZIR-XXX: Description convention.
# Usage: normalize-commit-msg.sh <commit-msg-file>

set -e

COMMIT_MSG_FILE="$1"

if [ -z "$COMMIT_MSG_FILE" ] || [ ! -f "$COMMIT_MSG_FILE" ]; then
    echo "Usage: normalize-commit-msg.sh <commit-msg-file>" >&2
    exit 1
fi

FIRST_LINE=$(head -n 1 "$COMMIT_MSG_FILE")

# Skip normalization for merge, squash, fixup, and revert commits
if echo "$FIRST_LINE" | grep -qE "^(Merge|Revert|fixup!|squash!)"; then
    exit 0
fi

# Skip empty messages or pure-comment messages
STRIPPED=$(echo "$FIRST_LINE" | sed 's/^[[:space:]]*//' | sed 's/[[:space:]]*$//')
if [ -z "$STRIPPED" ] || echo "$STRIPPED" | grep -q "^#"; then
    exit 0
fi

# --- Apply normalization rules to the subject line ---

SUBJECT="$STRIPPED"

# 1. Strip trailing whitespace (already done above)

# 2. If no ZIR-XXX: prefix, prepend ZIR-000:
if ! echo "$SUBJECT" | grep -qE "^ZIR-[0-9]+: "; then
    SUBJECT="ZIR-000: $SUBJECT"
fi

# 3. Capitalize first letter of description (the part after "ZIR-XXX: ")
PREFIX=$(echo "$SUBJECT" | grep -oE "^ZIR-[0-9]+: ")
DESCRIPTION="${SUBJECT#$PREFIX}"
FIRST_CHAR=$(echo "$DESCRIPTION" | cut -c1)
REST="${DESCRIPTION#?}"
UPPER_CHAR=$(echo "$FIRST_CHAR" | tr '[:lower:]' '[:upper:]')
DESCRIPTION="${UPPER_CHAR}${REST}"
SUBJECT="${PREFIX}${DESCRIPTION}"

# 4. Remove trailing period from subject line
SUBJECT=$(echo "$SUBJECT" | sed 's/\.$//')

# Reconstruct: replace first line, keep the rest unchanged
TAIL=$(tail -n +2 "$COMMIT_MSG_FILE")
if [ -n "$TAIL" ]; then
    printf '%s\n%s' "$SUBJECT" "$TAIL" > "$COMMIT_MSG_FILE"
else
    printf '%s' "$SUBJECT" > "$COMMIT_MSG_FILE"
fi
