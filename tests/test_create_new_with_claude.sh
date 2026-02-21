#!/bin/bash

# Test suite for create_new_with_claude() and _generate_statusline_script()
# Run with: bash tests/test_create_new_with_claude.sh

# Guard against re-execution when sourced
[[ "${_TEST_CREATE_RUNNING:-}" == "1" ]] && return 0 2>/dev/null || true
export _TEST_CREATE_RUNNING=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATUSLINE_ADD_SCRIPT="$HOME/.claude/statusline-add.sh"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

TESTS_PASSED=0
TESTS_FAILED=0

TEST_TEMP_DIR="/tmp/statusline_create_test_$$"
trap 'rm -rf "$TEST_TEMP_DIR" 2>/dev/null || true; unset _TEST_CREATE_RUNNING' EXIT

pass() {
    echo -e "${GREEN}✓ PASS${NC}: $1"
    TESTS_PASSED=$((TESTS_PASSED + 1))
}

fail() {
    echo -e "${RED}✗ FAIL${NC}: $1"
    TESTS_FAILED=$((TESTS_FAILED + 1))
}

# Source the script functions (then disable set -euo so test assertions work)
source "$STATUSLINE_ADD_SCRIPT" 2>/dev/null
set +euo pipefail 2>/dev/null || true

SAMPLE_JSON='{"model":{"display_name":"Claude Haiku 4.5"},"workspace":{"current_dir":"/tmp/demo"},"session_id":"test1","context_window":{"used_percentage":42.5,"total_input_tokens":50000,"total_output_tokens":8000}}'

echo "========================================================================"
echo "Testing create_new_with_claude() & _generate_statusline_script()"
echo "========================================================================"
echo ""

mkdir -p "$TEST_TEMP_DIR"

# ── Group 1: Script Generation (two-line, all components) ──
echo "Group 1: Two-line layout with all components (4 tests)"
echo "------------------------------------------------------"

GEN1="$TEST_TEMP_DIR/gen1.sh"
_generate_statusline_script "$GEN1" "MDGPCT" "two-line"

[ -f "$GEN1" ] && pass "Generator creates output file" || fail "Generator creates output file"
grep -q '#!/usr/bin/env bash' "$GEN1" && pass "Generated script has shebang" || fail "Generated script has shebang"
grep -q 'jq -r' "$GEN1" && pass "Generated script uses jq for JSON parsing" || fail "Generated script uses jq for JSON parsing"
bash -n "$GEN1" 2>/dev/null && pass "Generated script has valid bash syntax" || fail "Generated script has valid bash syntax"

echo ""

# ── Group 2: Script Generation (single-line) ──
echo "Group 2: Single-line layout (3 tests)"
echo "--------------------------------------"

GEN2="$TEST_TEMP_DIR/gen2.sh"
_generate_statusline_script "$GEN2" "MDP" "single-line"
chmod +x "$GEN2"

output2=$(echo "$SAMPLE_JSON" | "$GEN2" 2>/dev/null)
line_count=$(echo "$output2" | wc -l | tr -d ' ')

[ "$line_count" -eq 1 ] && pass "Single-line layout produces 1 line" || fail "Single-line layout produces 1 line (got $line_count)"
echo "$output2" | grep -q "Haiku" && pass "Single-line includes model name" || fail "Single-line includes model name"
echo "$output2" | grep -q "42%" && pass "Single-line includes progress percentage" || fail "Single-line includes progress percentage"

echo ""

# ── Group 3: Two-line output verification ──
echo "Group 3: Two-line output verification (3 tests)"
echo "------------------------------------------------"

chmod +x "$GEN1"
output1=$(echo "$SAMPLE_JSON" | "$GEN1" 2>/dev/null)
line_count1=$(echo "$output1" | wc -l | tr -d ' ')

[ "$line_count1" -eq 2 ] && pass "Two-line layout produces 2 lines" || fail "Two-line layout produces 2 lines (got $line_count1)"
echo "$output1" | head -1 | grep -q "Haiku" && pass "Line 1 includes model name" || fail "Line 1 includes model name"
echo "$output1" | tail -1 | grep -q "42%" && pass "Line 2 includes progress bar" || fail "Line 2 includes progress bar"

echo ""

# ── Group 4: Component selection ──
echo "Group 4: Component selection (4 tests)"
echo "---------------------------------------"

# Model only
GEN_M="$TEST_TEMP_DIR/gen_m.sh"
_generate_statusline_script "$GEN_M" "M" "single-line"
chmod +x "$GEN_M"
! grep -q 'git -C' "$GEN_M" && pass "M-only script excludes git code" || fail "M-only script excludes git code"
! grep -q 'cost=' "$GEN_M" && pass "M-only script excludes cost code" || fail "M-only script excludes cost code"

# Git only
GEN_G="$TEST_TEMP_DIR/gen_g.sh"
_generate_statusline_script "$GEN_G" "G" "single-line"
grep -q 'git -C' "$GEN_G" && pass "G component includes git branch code" || fail "G component includes git branch code"

# Progress only
GEN_P="$TEST_TEMP_DIR/gen_p.sh"
_generate_statusline_script "$GEN_P" "P" "single-line"
grep -q 'pct_int' "$GEN_P" && pass "P component includes progress bar code" || fail "P component includes progress bar code"

echo ""

# ── Group 5: Edge cases ──
echo "Group 5: Edge cases (3 tests)"
echo "------------------------------"

# Empty JSON
GEN_EMPTY="$TEST_TEMP_DIR/gen_empty.sh"
_generate_statusline_script "$GEN_EMPTY" "MDGPCT" "two-line"
chmod +x "$GEN_EMPTY"
empty_output=$(echo '{}' | "$GEN_EMPTY" 2>/dev/null)
[ -n "$empty_output" ] && pass "Script handles empty JSON input" || fail "Script handles empty JSON input"

# Missing fields
partial_json='{"model":{"display_name":"Test"}}'
partial_output=$(echo "$partial_json" | "$GEN_EMPTY" 2>/dev/null)
[ -n "$partial_output" ] && pass "Script handles partial JSON input" || fail "Script handles partial JSON input"

# Very long model name
long_json='{"model":{"display_name":"Claude Very Long Model Name 4.5 Extended Preview"},"workspace":{"current_dir":"/tmp"},"session_id":"t","context_window":{"used_percentage":50}}'
long_output=$(echo "$long_json" | "$GEN_EMPTY" 2>/dev/null)
[ -n "$long_output" ] && pass "Script handles long model names" || fail "Script handles long model names"

echo ""

# ── Summary ──
echo "========================================================================"
echo "Test Results: ${GREEN}${TESTS_PASSED} passed${NC}, ${RED}${TESTS_FAILED} failed${NC}"
echo "========================================================================"

[ "$TESTS_FAILED" -gt 0 ] && exit 1
exit 0
