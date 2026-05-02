#!/bin/bash
# Validates commit messages against the ZIR-[0-9]+: format.
# Usage:
#   check-commit-messages.sh                  # last 50 commits or origin/main..HEAD
#   check-commit-messages.sh <rev1> <rev2>    # explicit range
#   check-commit-messages.sh --fix-local      # print rebase -i instructions for non-compliant unpushed commits

set -euo pipefail

PATTERN="^ZIR-[0-9]+: .+"

# ── helpers ──────────────────────────────────────────────────────────────────

is_exempt() {
    local msg="$1"
    echo "$msg" | grep -qE "^(Merge|Revert) " && return 0
    [ -z "$msg" ] && return 0
    return 1
}

check_range() {
    local range="$1"
    local commits
    commits=$(git log --format="%H %s" "$range" 2>/dev/null) || {
        echo "ERROR: could not resolve range '$range'" >&2
        exit 1
    }

    if [ -z "$commits" ]; then
        echo "No commits in range '$range'."
        exit 0
    fi

    local pass=0 fail=0
    local failed_hashes=()

    while IFS= read -r line; do
        local hash subject
        hash="${line%% *}"
        subject="${line#* }"

        if is_exempt "$subject"; then
            echo "  SKIP  $(git rev-parse --short "$hash")  $subject"
            continue
        fi

        if echo "$subject" | grep -qE "$PATTERN"; then
            echo "  PASS  $(git rev-parse --short "$hash")  $subject"
            ((pass++)) || true
        else
            echo "  FAIL  $(git rev-parse --short "$hash")  $subject"
            ((fail++)) || true
            failed_hashes+=("$hash")
        fi
    done <<< "$commits"

    echo ""
    echo "Results: $pass passed, $fail failed."

    if [ "$fail" -gt 0 ]; then
        exit 1
    fi
}

fix_local() {
    local upstream
    if git rev-parse --verify origin/main >/dev/null 2>&1; then
        upstream="origin/main"
    elif git rev-parse --verify origin/master >/dev/null 2>&1; then
        upstream="origin/master"
    else
        echo "ERROR: cannot find origin/main or origin/master" >&2
        exit 1
    fi

    local commits
    commits=$(git log --format="%H %s" "${upstream}..HEAD" 2>/dev/null)

    if [ -z "$commits" ]; then
        echo "No unpushed commits."
        exit 0
    fi

    local non_compliant=()
    while IFS= read -r line; do
        local hash subject
        hash="${line%% *}"
        subject="${line#* }"
        is_exempt "$subject" && continue
        echo "$subject" | grep -qE "$PATTERN" && continue
        non_compliant+=("$hash")
    done <<< "$commits"

    if [ "${#non_compliant[@]}" -eq 0 ]; then
        echo "All unpushed commits are compliant. Nothing to fix."
        exit 0
    fi

    echo "# Run the following to fix non-compliant commit messages:"
    echo "#   git rebase -i ${upstream}"
    echo "# Then change 'pick' to 'reword' for the commits listed below,"
    echo "# and prefix the message with 'ZIR-000: ' when the editor opens."
    echo ""
    for h in "${non_compliant[@]}"; do
        local short msg
        short=$(git rev-parse --short "$h")
        msg=$(git log -1 --format="%s" "$h")
        echo "  reword $short  $msg"
    done
    echo ""
    echo "# Suggested amended messages (prepend ZIR-000: to each):"
    for h in "${non_compliant[@]}"; do
        local msg
        msg=$(git log -1 --format="%s" "$h")
        echo "  ZIR-000: $msg"
    done
}

# ── main ─────────────────────────────────────────────────────────────────────

if [ "${1:-}" = "--fix-local" ]; then
    fix_local
    exit 0
fi

if [ $# -eq 2 ]; then
    RANGE="${1}..${2}"
elif [ $# -eq 1 ] && [ "$1" != "--fix-local" ]; then
    RANGE="$1"
else
    # Default: origin/main..HEAD, or last 50 commits if no remote
    if git rev-parse --verify origin/main >/dev/null 2>&1; then
        RANGE="origin/main..HEAD"
    elif git rev-parse --verify origin/master >/dev/null 2>&1; then
        RANGE="origin/master..HEAD"
    else
        RANGE="HEAD~50..HEAD"
    fi
fi

echo "Checking commit messages in range: $RANGE"
echo ""
check_range "$RANGE"
