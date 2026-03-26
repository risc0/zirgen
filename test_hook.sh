#!/bin/bash
# Test suite for commit-msg hook
# This script validates that the commit message hook works correctly

set -e

HOOK_SCRIPT="./hooks/commit-msg"
TEMP_MSG_FILE=$(mktemp)

cleanup() {
    rm -f "$TEMP_MSG_FILE"
}

trap cleanup EXIT

test_valid_message() {
    local msg="$1"
    echo "$msg" > "$TEMP_MSG_FILE"
    if "$HOOK_SCRIPT" "$TEMP_MSG_FILE"; then
        echo "✓ Test passed: '$msg'"
        return 0
    else
        echo "✗ Test failed: '$msg' should be valid"
        return 1
    fi
}

test_invalid_message() {
    local msg="$1"
    echo "$msg" > "$TEMP_MSG_FILE"
    if "$HOOK_SCRIPT" "$TEMP_MSG_FILE"; then
        echo "✗ Test failed: '$msg' should be invalid"
        return 1
    else
        echo "✓ Test passed: '$msg' correctly rejected"
        return 0
    fi
}

echo "Running commit-msg hook tests..."
echo ""

echo "Test 1: Valid commit message with single digit issue number"
test_valid_message "ZIR-1: Add new feature"

echo "Test 2: Valid commit message with multiple digit issue number"
test_valid_message "ZIR-123: Fix memory leak in connection pool"

echo "Test 3: Invalid - multi-line message with valid format on second line"
cat > "$TEMP_MSG_FILE" << 'EOF'
This is an invalid first line

ZIR-123: Valid format appears later
EOF
if "$HOOK_SCRIPT" "$TEMP_MSG_FILE"; then
    echo "✗ Test 3 failed: Multi-line message with invalid first line should be rejected"
    exit 1
else
    echo "✓ Test 3 passed: Multi-line message with invalid first line correctly rejected"
fi

echo "Test 4: Invalid - missing ZIR prefix"
test_invalid_message "123: Add new feature"

echo "Test 5: Invalid - missing colon"
test_invalid_message "ZIR-123 Add new feature"

echo "Test 6: Invalid - missing issue number"
test_invalid_message "ZIR-: Add new feature"

echo "Test 7: Valid - merge commit"
test_valid_message "Merge branch 'feature/xyz' into main"

echo "Test 8: Valid - revert commit"
test_valid_message "Revert \"ZIR-123: Add new feature\""

echo ""
echo "All tests passed! ✓"
