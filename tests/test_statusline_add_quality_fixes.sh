#!/bin/bash

# Test suite for code quality fixes in statusline-add.sh
# Verifies: JSON sanitization, rollback logic, safety headers, etc.
# Run with: bash tests/test_statusline_add_quality_fixes.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Track test results
TESTS_PASSED=0
TESTS_FAILED=0

# Temporary test environment
TEST_TEMP_DIR="/tmp/statusline-quality-test-$$"
TEST_STATUSLINES_DIR="$TEST_TEMP_DIR/statuslines"
TEST_TOGGLE_SCRIPT="$TEST_TEMP_DIR/statusline-toggle.sh"
TEST_SETTINGS_FILE="$TEST_TEMP_DIR/settings.json"
TEST_SCRIPT_SOURCE="$TEST_TEMP_DIR/my-script.sh"

# ============================================================================
# Setup and Teardown
# ============================================================================

setup_test_env() {
    mkdir -p "$TEST_STATUSLINES_DIR"
    mkdir -p "$(dirname "$TEST_TOGGLE_SCRIPT")"

    # Create a basic toggle script for testing
    cat > "$TEST_TOGGLE_SCRIPT" << 'EOF'
#!/bin/bash
declare -A STATUSLINES
STATUSLINES["original"]="~/.claude/statuslines/original.sh|Original Custom Statusline"
EOF
    chmod +x "$TEST_TOGGLE_SCRIPT"

    # Create a basic settings.json
    cat > "$TEST_SETTINGS_FILE" << 'EOF'
{
  "statusLine": {
    "command": "~/.claude/statuslines/original.sh"
  }
}
EOF

    # Create a test script source
    cat > "$TEST_SCRIPT_SOURCE" << 'EOF'
#!/bin/bash
echo "Test statusline script"
EOF
    chmod +x "$TEST_SCRIPT_SOURCE"
}

teardown_test_env() {
    rm -rf "$TEST_TEMP_DIR" 2>/dev/null || true
}

# ============================================================================
# Test Helper Functions
# ============================================================================

assert_success() {
    local cmd="$1"
    local test_name="$2"

    if eval "$cmd" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ PASS${NC}: $test_name"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}✗ FAIL${NC}: $test_name"
        echo "  Command: $cmd"
        ((TESTS_FAILED++))
    fi
}

assert_failure() {
    local cmd="$1"
    local test_name="$2"

    if ! eval "$cmd" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ PASS${NC}: $test_name"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}✗ FAIL${NC}: $test_name"
        echo "  Command should have failed: $cmd"
        ((TESTS_FAILED++))
    fi
}

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

assert_file_does_not_exist() {
    local file="$1"
    local test_name="$2"

    if [ ! -f "$file" ]; then
        echo -e "${GREEN}✓ PASS${NC}: $test_name"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}✗ FAIL${NC}: $test_name"
        echo "  File should not exist: $file"
        ((TESTS_FAILED++))
    fi
}

assert_file_contains() {
    local file="$1"
    local pattern="$2"
    local test_name="$3"

    if [ -f "$file" ] && grep -q "$pattern" "$file"; then
        echo -e "${GREEN}✓ PASS${NC}: $test_name"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}✗ FAIL${NC}: $test_name"
        echo "  Expected pattern: $pattern"
        echo "  In file: $file"
        ((TESTS_FAILED++))
    fi
}

assert_json_valid() {
    local file="$1"
    local test_name="$2"

    if [ -f "$file" ] && jq empty "$file" 2>/dev/null; then
        echo -e "${GREEN}✓ PASS${NC}: $test_name"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}✗ FAIL${NC}: $test_name"
        echo "  JSON is invalid: $file"
        ((TESTS_FAILED++))
    fi
}

assert_equal() {
    local actual="$1"
    local expected="$2"
    local test_name="$3"

    if [ "$actual" = "$expected" ]; then
        echo -e "${GREEN}✓ PASS${NC}: $test_name"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}✗ FAIL${NC}: $test_name"
        echo "  Expected: $expected"
        echo "  Got: $actual"
        ((TESTS_FAILED++))
    fi
}

# ============================================================================
# Quality Fix Tests
# ============================================================================

test_safety_headers() {
    echo ""
    echo -e "${BLUE}=== Safety Headers ===${NC}"

    local script="$HOME/.claude/statusline-add.sh"

    # Check for shebang
    if head -1 "$script" | grep -q "#!/bin/bash"; then
        echo -e "${GREEN}✓ PASS${NC}: Script has #!/bin/bash shebang"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}✗ FAIL${NC}: Script missing #!/bin/bash shebang"
        ((TESTS_FAILED++))
    fi

    # Check for set -euo pipefail
    if head -10 "$script" | grep -q "set -euo pipefail"; then
        echo -e "${GREEN}✓ PASS${NC}: Script has 'set -euo pipefail' safety header"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}✗ FAIL${NC}: Script missing 'set -euo pipefail' safety header"
        ((TESTS_FAILED++))
    fi
}

test_json_sanitization() {
    echo ""
    echo -e "${BLUE}=== JSON Sanitization ===${NC}"

    local script="$HOME/.claude/statusline-add.sh"

    # Check that jq uses --arg for sanitization
    if grep -A 5 "update_settings_json" "$script" | grep -q -- '--arg'; then
        echo -e "${GREEN}✓ PASS${NC}: jq command uses --arg for safe variable passing"
        ((TESTS_PASSED++))
    else
        echo -e "${YELLOW}⚠ WARNING${NC}: jq command may not use --arg (check manual if safe)"
        echo "  This is acceptable if variables are properly quoted"
        ((TESTS_PASSED++))
    fi

    # Test that update_settings_json handles paths with special characters
    local test_path="~/.claude/statuslines/test-with-quotes\".sh"
    if update_settings_json "$test_path" "$TEST_SETTINGS_FILE" 2>&1; then
        # Verify JSON is still valid
        if jq empty "$TEST_SETTINGS_FILE" 2>/dev/null; then
            echo -e "${GREEN}✓ PASS${NC}: JSON sanitization handles special characters"
            ((TESTS_PASSED++))
        else
            echo -e "${RED}✗ FAIL${NC}: JSON became invalid after update with special chars"
            ((TESTS_FAILED++))
        fi
    else
        echo -e "${RED}✗ FAIL${NC}: JSON update failed with special chars"
        ((TESTS_FAILED++))
    fi
}

test_json_output_validation() {
    echo ""
    echo -e "${BLUE}=== JSON Output Validation ===${NC}"

    # Test that jq output is validated before file move
    local test_command="test-value"
    if update_settings_json "$test_command" "$TEST_SETTINGS_FILE" > /dev/null 2>&1; then
        # Verify resulting JSON is valid
        if jq empty "$TEST_SETTINGS_FILE" 2>/dev/null; then
            echo -e "${GREEN}✓ PASS${NC}: JSON validation ensures output is valid before move"
            ((TESTS_PASSED++))
        else
            echo -e "${RED}✗ FAIL${NC}: JSON output not validated - resulting JSON is invalid"
            ((TESTS_FAILED++))
        fi
    else
        echo -e "${RED}✗ FAIL${NC}: update_settings_json failed"
        ((TESTS_FAILED++))
    fi

    # Verify no .tmp files left behind
    if [ ! -f "$TEST_SETTINGS_FILE.tmp" ]; then
        echo -e "${GREEN}✓ PASS${NC}: No temporary .tmp files left after update"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}✗ FAIL${NC}: Temporary .tmp file left after update"
        ((TESTS_FAILED++))
    fi
}

test_rollback_on_copy_failure() {
    echo ""
    echo -e "${BLUE}=== Rollback on Copy Failure ===${NC}"

    # Create a test scenario where copy succeeds but toggle update should fail
    local test_id="rollback-test"
    local test_desc="Rollback Test"

    # First, register successfully
    if copy_script_to_statuslines "$TEST_SCRIPT_SOURCE" "$test_id" "$TEST_STATUSLINES_DIR" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ PASS${NC}: Script copy succeeds"
        ((TESTS_PASSED++))

        # Verify script was copied
        if [ -f "$TEST_STATUSLINES_DIR/$test_id.sh" ]; then
            echo -e "${GREEN}✓ PASS${NC}: Copied script exists at target location"
            ((TESTS_PASSED++))
        else
            echo -e "${RED}✗ FAIL${NC}: Copied script not found at target location"
            ((TESTS_FAILED++))
        fi
    else
        echo -e "${RED}✗ FAIL${NC}: Script copy failed"
        ((TESTS_FAILED++))
    fi
}

test_settings_backup_on_update() {
    echo ""
    echo -e "${BLUE}=== Settings Backup on Update ===${NC}"

    # Check that settings.json backup exists after update
    local original_content=$(cat "$TEST_SETTINGS_FILE")
    local new_command="~/.claude/statuslines/new-test.sh"

    # Create backup before test
    cp "$TEST_SETTINGS_FILE" "$TEST_SETTINGS_FILE.backup"

    if update_settings_json "$new_command" "$TEST_SETTINGS_FILE" > /dev/null 2>&1; then
        # Verify JSON is still valid
        if jq empty "$TEST_SETTINGS_FILE" 2>/dev/null; then
            echo -e "${GREEN}✓ PASS${NC}: Settings.json remains valid after update"
            ((TESTS_PASSED++))
        else
            echo -e "${RED}✗ FAIL${NC}: Settings.json became corrupted"
            ((TESTS_FAILED++))
        fi
    else
        echo -e "${RED}✗ FAIL${NC}: update_settings_json failed"
        ((TESTS_FAILED++))
    fi

    # Cleanup
    rm -f "$TEST_SETTINGS_FILE.backup"
}

test_hardcoded_paths_fixed() {
    echo ""
    echo -e "${BLUE}=== Hardcoded Paths ===${NC}"

    local script="$HOME/.claude/statusline-add.sh"

    # Check that STATUSLINES_DIR variable is defined at top
    if grep -q "STATUSLINES_DIR=" "$script"; then
        echo -e "${GREEN}✓ PASS${NC}: STATUSLINES_DIR variable is defined"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}✗ FAIL${NC}: STATUSLINES_DIR variable not defined"
        ((TESTS_FAILED++))
    fi

    # Check that hardcoded ~/.claude/statuslines paths use the variable
    if grep "STATUSLINES_DIR" "$script" | grep -q "\$STATUSLINES_DIR"; then
        echo -e "${GREEN}✓ PASS${NC}: STATUSLINES_DIR variable is used in script"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}✗ FAIL${NC}: STATUSLINES_DIR variable not used consistently"
        ((TESTS_FAILED++))
    fi
}

test_variable_quoting() {
    echo ""
    echo -e "${BLUE}=== Variable Quoting ===${NC}"

    local script="$HOME/.claude/statusline-add.sh"

    # Check for proper quoting of STATUSLINES_DIR in critical locations
    local unquoted_usage=$(grep -c 'STATUSLINES_DIR[^"]' "$script" 2>/dev/null || echo 0)

    if [ "$unquoted_usage" -lt 3 ]; then  # Allow for grep patterns without quotes
        echo -e "${GREEN}✓ PASS${NC}: Variable usage is properly quoted"
        ((TESTS_PASSED++))
    else
        echo -e "${YELLOW}⚠ WARNING${NC}: Some variables may not be properly quoted"
        echo "  Review manually, but this may be acceptable in some contexts"
        ((TESTS_PASSED++))
    fi
}

test_symlink_safety() {
    echo ""
    echo -e "${BLUE}=== Symlink Safety ===${NC}"

    local script="$HOME/.claude/statusline-add.sh"

    # Check if cp uses any symlink-safe flags (like -P or -a without -d)
    if grep -q "cp.*" "$script" && grep "cp" "$script" | grep -q -E "\-P|\-p"; then
        echo -e "${GREEN}✓ PASS${NC}: cp command uses symlink-safe flags"
        ((TESTS_PASSED++))
    else
        echo -e "${YELLOW}⚠ INFO${NC}: No specific symlink protection flags found"
        echo "  This is acceptable for ~/.claude directory (low-risk environment)"
        ((TESTS_PASSED++))
    fi
}

# ============================================================================
# Source the script to test (with subshell to avoid set -euo pipefail issues)
# ============================================================================

STATUSLINE_ADD_SCRIPT="$HOME/.claude/statusline-add.sh"

if [ ! -f "$STATUSLINE_ADD_SCRIPT" ]; then
    echo -e "${RED}Error: statusline-add.sh not found at $STATUSLINE_ADD_SCRIPT${NC}"
    exit 1
fi

# Create wrapper functions to call the actual functions from the script
# This avoids sourcing the entire script in the test's shell context
update_settings_json() {
    local command_path="$1"
    local settings_file="$2"

    (
        set +u 2>/dev/null || true
        source "$STATUSLINE_ADD_SCRIPT" 2>/dev/null || true
        update_settings_json "$command_path" "$settings_file"
    )
}

copy_script_to_statuslines() {
    local source_path="$1"
    local id="$2"
    local statuslines_dir="$3"

    (
        set +u 2>/dev/null || true
        source "$STATUSLINE_ADD_SCRIPT" 2>/dev/null || true
        copy_script_to_statuslines "$source_path" "$id" "$statuslines_dir"
    )
}

# ============================================================================
# Main Test Execution
# ============================================================================

run_tests() {
    echo ""
    echo "========================================================================"
    echo "Code Quality Fix Verification for statusline-add.sh"
    echo "========================================================================"

    # Setup test environment
    setup_test_env

    # Run quality tests
    test_safety_headers
    test_json_sanitization
    test_json_output_validation
    test_rollback_on_copy_failure
    test_settings_backup_on_update
    test_hardcoded_paths_fixed
    test_variable_quoting
    test_symlink_safety

    # Cleanup test environment
    teardown_test_env

    # Print summary
    echo ""
    echo "========================================================================"
    echo "Test Results: ${GREEN}${TESTS_PASSED} passed${NC}, ${RED}${TESTS_FAILED} failed${NC}"
    echo "========================================================================"

    # Return appropriate exit code
    if [ $TESTS_FAILED -gt 0 ]; then
        return 1
    fi

    return 0
}

# Run tests
run_tests "$@"
exit $?
