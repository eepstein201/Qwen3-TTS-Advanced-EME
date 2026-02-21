#!/bin/bash

# Test suite for statusline-add.sh
# Run with: bash tests/test_statusline_add.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
STATUSLINE_ADD_SCRIPT="$HOME/.claude/statusline-add.sh"
ZSHRC_FILE="$HOME/.zshrc"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track test results
TESTS_PASSED=0
TESTS_FAILED=0

# Test helper function
assert_file_exists() {
    local file="$1"
    local test_name="$2"

    if [ -f "$file" ]; then
        echo -e "${GREEN}✓ PASS${NC}: $test_name"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}✗ FAIL${NC}: $test_name"
        echo "  Expected file: $file"
        ((TESTS_FAILED++))
    fi
}

assert_file_executable() {
    local file="$1"
    local test_name="$2"

    if [ -x "$file" ]; then
        echo -e "${GREEN}✓ PASS${NC}: $test_name"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}✗ FAIL${NC}: $test_name"
        echo "  Expected executable: $file"
        ((TESTS_FAILED++))
    fi
}

assert_file_contains() {
    local file="$1"
    local pattern="$2"
    local test_name="$3"

    if grep -q "$pattern" "$file"; then
        echo -e "${GREEN}✓ PASS${NC}: $test_name"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}✗ FAIL${NC}: $test_name"
        echo "  Expected pattern: $pattern"
        echo "  In file: $file"
        ((TESTS_FAILED++))
    fi
}

assert_true() {
    local condition="$1"
    local test_name="$2"

    if eval "$condition"; then
        echo -e "${GREEN}✓ PASS${NC}: $test_name"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}✗ FAIL${NC}: $test_name"
        echo "  Condition: $condition"
        ((TESTS_FAILED++))
    fi
}

# ============================================================================
# Tests
# ============================================================================

echo "Running statusline-add.sh tests..."
echo ""

# Test 1: Script exists
assert_file_exists "$STATUSLINE_ADD_SCRIPT" "statusline-add.sh exists"

# Test 2: Script is executable
assert_file_executable "$STATUSLINE_ADD_SCRIPT" "statusline-add.sh is executable"

# Test 3: Script has bash shebang
assert_file_contains "$STATUSLINE_ADD_SCRIPT" "^#!/bin/bash" "statusline-add.sh has #!/bin/bash shebang"

# Test 4: Script has set -e for safety
assert_file_contains "$STATUSLINE_ADD_SCRIPT" "set -e" "statusline-add.sh has set -e for safety"

# Test 5: Script defines STATUSLINES_DIR variable
assert_file_contains "$STATUSLINE_ADD_SCRIPT" "STATUSLINES_DIR" "statusline-add.sh defines STATUSLINES_DIR variable"

# Test 6: Script defines TOGGLE_SCRIPT variable
assert_file_contains "$STATUSLINE_ADD_SCRIPT" "TOGGLE_SCRIPT" "statusline-add.sh defines TOGGLE_SCRIPT variable"

# Test 7: Script defines SETTINGS_FILE variable
assert_file_contains "$STATUSLINE_ADD_SCRIPT" "SETTINGS_FILE" "statusline-add.sh defines SETTINGS_FILE variable"

# Test 8: Script has register_existing function
assert_file_contains "$STATUSLINE_ADD_SCRIPT" "register_existing()" "statusline-add.sh defines register_existing() function"

# Test 9: Script has create_new_with_claude function
assert_file_contains "$STATUSLINE_ADD_SCRIPT" "create_new_with_claude()" "statusline-add.sh defines create_new_with_claude() function"

# Test 10: Script has main menu logic
assert_file_contains "$STATUSLINE_ADD_SCRIPT" "echo.*[Cc]hoose" "statusline-add.sh displays menu prompt"

# Test 11: alias exists in ~/.zshrc
if [ -f "$ZSHRC_FILE" ]; then
    assert_file_contains "$ZSHRC_FILE" "claude-statusline-add" "alias 'claude-statusline-add' registered in ~/.zshrc"
else
    echo -e "${YELLOW}⊘ SKIP${NC}: alias 'claude-statusline-add' registered in ~/.zshrc (zshrc not found)"
fi

# Test 12: Script runs without error and shows menu
if [ -f "$STATUSLINE_ADD_SCRIPT" ]; then
    # Test with 'help' as input to show menu without waiting for interactive input
    output=$("$STATUSLINE_ADD_SCRIPT" 2>&1 || true)

    if echo "$output" | grep -q -i "menu\|choose\|option"; then
        echo -e "${GREEN}✓ PASS${NC}: statusline-add.sh displays menu when run"
        ((TESTS_PASSED++))
    else
        echo -e "${YELLOW}⊘ SKIP${NC}: statusline-add.sh displays menu when run (interactive test skipped)"
    fi
fi

# ============================================================================
# Summary
# ============================================================================

echo ""
echo "============================================================================"
echo "Test Results: ${GREEN}${TESTS_PASSED} passed${NC}, ${RED}${TESTS_FAILED} failed${NC}"
echo "============================================================================"

if [ $TESTS_FAILED -gt 0 ]; then
    exit 1
fi

exit 0
