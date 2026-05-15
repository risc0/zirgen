#!/bin/bash
# Tests for scripts/normalize-commit-msg.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
NORMALIZER="$REPO_ROOT/scripts/normalize-commit-msg.sh"

PASS=0
FAIL=0

run_case() {
    local description="$1"
    local input="$2"
    local expected="$3"

    TMPFILE=$(mktemp)
    printf '%s' "$input" > "$TMPFILE"
    bash "$NORMALIZER" "$TMPFILE"
    RESULT=$(head -n 1 "$TMPFILE")
    rm -f "$TMPFILE"

    if [ "$RESULT" = "$expected" ]; then
        echo "PASS: $description"
        PASS=$((PASS + 1))
    else
        echo "FAIL: $description"
        echo "  Input:    $input"
        echo "  Expected: $expected"
        echo "  Got:      $RESULT"
        FAIL=$((FAIL + 1))
    fi
}

run_passthrough() {
    local description="$1"
    local input="$2"

    TMPFILE=$(mktemp)
    printf '%s' "$input" > "$TMPFILE"
    bash "$NORMALIZER" "$TMPFILE"
    RESULT=$(head -n 1 "$TMPFILE")
    rm -f "$TMPFILE"

    if [ "$RESULT" = "$input" ]; then
        echo "PASS: $description"
        PASS=$((PASS + 1))
    else
        echo "FAIL: $description (should be unchanged)"
        echo "  Input:    $input"
        echo "  Got:      $RESULT"
        FAIL=$((FAIL + 1))
    fi
}

# Case 1: Message with no ZIR prefix gets ZIR-000 prepended
run_case "no prefix gets ZIR-000" \
    "add something useful" \
    "ZIR-000: Add something useful"

# Case 2: Correct prefix is preserved as-is
run_case "correct prefix preserved" \
    "ZIR-123: Add new feature" \
    "ZIR-123: Add new feature"

# Case 3: Lowercase first word of description is capitalized
run_case "lowercase description capitalized" \
    "ZIR-42: fix the bug" \
    "ZIR-42: Fix the bug"

# Case 4: Trailing period is removed
run_case "trailing period removed" \
    "ZIR-7: Update configuration." \
    "ZIR-7: Update configuration"

# Case 5: Extra trailing whitespace on subject line is stripped
run_case "trailing whitespace stripped" \
    "ZIR-99: Clean up code   " \
    "ZIR-99: Clean up code"

# Case 6: Merge commit is passed through unchanged
run_passthrough "merge commit passthrough" \
    "Merge branch 'main' into feature/foo"

# Case 7: Revert commit is passed through unchanged
run_passthrough "revert commit passthrough" \
    "Revert \"ZIR-10: Some change\""

# Case 8: Multi-line message: body lines are preserved
TMPFILE=$(mktemp)
printf 'ZIR-5: do the thing.\n\nSome body text.\n' > "$TMPFILE"
bash "$NORMALIZER" "$TMPFILE"
SUBJECT=$(head -n 1 "$TMPFILE")
BODY=$(tail -n +2 "$TMPFILE")
rm -f "$TMPFILE"
if [ "$SUBJECT" = "ZIR-5: Do the thing" ] && echo "$BODY" | grep -q "Some body text"; then
    echo "PASS: multi-line message preserves body"
    PASS=$((PASS + 1))
else
    echo "FAIL: multi-line message preserves body"
    echo "  Subject: $SUBJECT"
    echo "  Body:    $BODY"
    FAIL=$((FAIL + 1))
fi

echo ""
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
