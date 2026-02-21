#!/bin/bash

# Integration test for the full statusline management workflow
# Tests: statusline-add.sh + statusline-toggle.sh working together
# Run with: bash tests/test_statusline_integration.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

TESTS_PASSED=0
TESTS_FAILED=0

pass() {
    echo -e "${GREEN}✓ PASS${NC}: $1"
    TESTS_PASSED=$((TESTS_PASSED + 1))
}

fail() {
    echo -e "${RED}✗ FAIL${NC}: $1"
    TESTS_FAILED=$((TESTS_FAILED + 1))
}

echo "========================================================================"
echo "Integration Tests: Statusline Management System"
echo "========================================================================"
echo ""

# ── Group 1: File existence and permissions ──
echo "Group 1: Core files exist and are executable (6 tests)"
echo "------------------------------------------------------"

[ -f "$HOME/.claude/statusline-add.sh" ] && pass "statusline-add.sh exists" || fail "statusline-add.sh exists"
[ -x "$HOME/.claude/statusline-add.sh" ] && pass "statusline-add.sh is executable" || fail "statusline-add.sh is executable"
[ -f "$HOME/.claude/statusline-toggle.sh" ] && pass "statusline-toggle.sh exists" || fail "statusline-toggle.sh exists"
[ -x "$HOME/.claude/statusline-toggle.sh" ] && pass "statusline-toggle.sh is executable" || fail "statusline-toggle.sh is executable"
[ -d "$HOME/.claude/statuslines" ] && pass "statuslines directory exists" || fail "statuslines directory exists"
[ -f "$HOME/.claude/statuslines/original.sh" ] && pass "original.sh migrated to statuslines dir" || fail "original.sh migrated to statuslines dir"

echo ""

# ── Group 2: Script syntax validation ──
echo "Group 2: All scripts have valid syntax (3 tests)"
echo "-------------------------------------------------"

bash -n "$HOME/.claude/statusline-add.sh" 2>/dev/null && pass "statusline-add.sh valid bash" || fail "statusline-add.sh valid bash"
bash -n "$HOME/.claude/statusline-toggle.sh" 2>/dev/null && pass "statusline-toggle.sh valid bash" || fail "statusline-toggle.sh valid bash"
bash -n "$HOME/.claude/statuslines/original.sh" 2>/dev/null && pass "original.sh valid bash" || fail "original.sh valid bash"

echo ""

# ── Group 3: Toggle script has correct STATUSLINES format ──
echo "Group 3: Toggle script format (3 tests)"
echo "----------------------------------------"

grep -q 'STATUSLINES=' "$HOME/.claude/statusline-toggle.sh" && pass "Toggle defines STATUSLINES array" || fail "Toggle defines STATUSLINES array"
grep -q '"original|' "$HOME/.claude/statusline-toggle.sh" && pass "Toggle has original entry" || fail "Toggle has original entry"
grep -q '"starship|' "$HOME/.claude/statusline-toggle.sh" && pass "Toggle has starship entry" || fail "Toggle has starship entry"

echo ""

# ── Group 4: Settings.json integration ──
echo "Group 4: Settings.json structure (2 tests)"
echo "-------------------------------------------"

[ -f "$HOME/.claude/settings.json" ] && pass "settings.json exists" || fail "settings.json exists"

if command -v jq &> /dev/null; then
    jq -e '.statusLine.command' "$HOME/.claude/settings.json" > /dev/null 2>&1 && \
        pass "settings.json has statusLine.command" || fail "settings.json has statusLine.command"
else
    grep -q '"command"' "$HOME/.claude/settings.json" && \
        pass "settings.json has statusLine.command" || fail "settings.json has statusLine.command"
fi

echo ""

# ── Group 5: Statusline scripts produce output ──
echo "Group 5: Statusline scripts produce output (2 tests)"
echo "----------------------------------------------------"

SAMPLE='{"model":{"display_name":"Test Model"},"workspace":{"current_dir":"/tmp"},"session_id":"int-test","context_window":{"used_percentage":50,"total_input_tokens":1000,"total_output_tokens":500}}'

output=$(echo "$SAMPLE" | "$HOME/.claude/statuslines/original.sh" 2>/dev/null)
[ -n "$output" ] && pass "original.sh produces output from JSON" || fail "original.sh produces output from JSON"

# Test generated script capability
TEST_DIR="/tmp/integration_test_$$"
mkdir -p "$TEST_DIR"
trap 'rm -rf "$TEST_DIR"' EXIT

source "$HOME/.claude/statusline-add.sh" 2>/dev/null
set +euo pipefail 2>/dev/null || true

_generate_statusline_script "$TEST_DIR/test-gen.sh" "MDPCT" "two-line"
chmod +x "$TEST_DIR/test-gen.sh"
gen_output=$(echo "$SAMPLE" | "$TEST_DIR/test-gen.sh" 2>/dev/null)
[ -n "$gen_output" ] && pass "Generated statusline produces output" || fail "Generated statusline produces output"

echo ""

# ── Group 6: Shell aliases configured ──
echo "Group 6: Shell aliases (2 tests)"
echo "---------------------------------"

grep -q 'claude-statusline-toggle' "$HOME/.zshrc" && pass "Toggle alias in .zshrc" || fail "Toggle alias in .zshrc"
grep -q 'claude-statusline-add' "$HOME/.zshrc" && pass "Add alias in .zshrc" || fail "Add alias in .zshrc"

echo ""

# ── Summary ──
echo "========================================================================"
echo "Integration Results: ${GREEN}${TESTS_PASSED} passed${NC}, ${RED}${TESTS_FAILED} failed${NC}"
echo "========================================================================"

[ "$TESTS_FAILED" -gt 0 ] && exit 1
exit 0
