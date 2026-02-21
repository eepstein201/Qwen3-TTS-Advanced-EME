#!/bin/bash

# Test suite for register_existing function in statusline-add.sh
# Run with: bash tests/test_register_existing.sh
#
# This test file follows TDD principles:
# - Test first, watch fail
# - Implement minimal code to pass
# - All tests comprehensive and isolated

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
TEST_TEMP_DIR="/tmp/statusline-test-$$"
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
STATUSLINES["starship"]="~/.local/bin/starship-claude|Starship-Claude"
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
# Validation Tests
# ============================================================================

test_id_format_valid() {
    echo ""
    echo -e "${BLUE}=== ID Format Validation ===${NC}"

    # Test valid IDs
    if validate_id_format "my-statusline" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ PASS${NC}: ID format validation: accepts 'my-statusline'"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}✗ FAIL${NC}: ID format validation: accepts 'my-statusline'"
        ((TESTS_FAILED++))
    fi

    if validate_id_format "mystatus123" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ PASS${NC}: ID format validation: accepts 'mystatus123'"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}✗ FAIL${NC}: ID format validation: accepts 'mystatus123'"
        ((TESTS_FAILED++))
    fi

    if validate_id_format "a" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ PASS${NC}: ID format validation: accepts single character 'a'"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}✗ FAIL${NC}: ID format validation: accepts single character 'a'"
        ((TESTS_FAILED++))
    fi

    if validate_id_format "test-script-123" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ PASS${NC}: ID format validation: accepts 'test-script-123'"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}✗ FAIL${NC}: ID format validation: accepts 'test-script-123'"
        ((TESTS_FAILED++))
    fi
}

test_id_format_invalid() {
    echo ""
    echo -e "${BLUE}=== ID Format Rejection ===${NC}"

    # Test invalid IDs - use function call directly instead of eval strings
    if ! validate_id_format 'my statusline' > /dev/null 2>&1; then
        echo -e "${GREEN}✓ PASS${NC}: ID format validation: rejects spaces"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}✗ FAIL${NC}: ID format validation: rejects spaces"
        ((TESTS_FAILED++))
    fi

    if ! validate_id_format 'my_statusline' > /dev/null 2>&1; then
        echo -e "${GREEN}✓ PASS${NC}: ID format validation: rejects underscores"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}✗ FAIL${NC}: ID format validation: rejects underscores"
        ((TESTS_FAILED++))
    fi

    if ! validate_id_format 'my.statusline' > /dev/null 2>&1; then
        echo -e "${GREEN}✓ PASS${NC}: ID format validation: rejects dots"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}✗ FAIL${NC}: ID format validation: rejects dots"
        ((TESTS_FAILED++))
    fi

    if ! validate_id_format 'my@statusline' > /dev/null 2>&1; then
        echo -e "${GREEN}✓ PASS${NC}: ID format validation: rejects special chars"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}✗ FAIL${NC}: ID format validation: rejects special chars"
        ((TESTS_FAILED++))
    fi

    if ! validate_id_format '' > /dev/null 2>&1; then
        echo -e "${GREEN}✓ PASS${NC}: ID format validation: rejects empty string"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}✗ FAIL${NC}: ID format validation: rejects empty string"
        ((TESTS_FAILED++))
    fi
}

test_id_uniqueness() {
    echo ""
    echo -e "${BLUE}=== ID Uniqueness Validation ===${NC}"

    # Test that existing IDs are rejected
    if ! validate_id_uniqueness 'original' "$TEST_TOGGLE_SCRIPT" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ PASS${NC}: ID uniqueness: rejects existing 'original'"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}✗ FAIL${NC}: ID uniqueness: rejects existing 'original'"
        ((TESTS_FAILED++))
    fi

    if ! validate_id_uniqueness 'starship' "$TEST_TOGGLE_SCRIPT" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ PASS${NC}: ID uniqueness: rejects existing 'starship'"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}✗ FAIL${NC}: ID uniqueness: rejects existing 'starship'"
        ((TESTS_FAILED++))
    fi

    if validate_id_uniqueness 'new-script' "$TEST_TOGGLE_SCRIPT" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ PASS${NC}: ID uniqueness: accepts new ID 'new-script'"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}✗ FAIL${NC}: ID uniqueness: accepts new ID 'new-script'"
        ((TESTS_FAILED++))
    fi
}

test_script_path_validation() {
    echo ""
    echo -e "${BLUE}=== Script Path Validation ===${NC}"

    # Test path existence
    if ! validate_script_path '/nonexistent/path/script.sh' > /dev/null 2>&1; then
        echo -e "${GREEN}✓ PASS${NC}: Script path: rejects non-existent file"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}✗ FAIL${NC}: Script path: rejects non-existent file"
        ((TESTS_FAILED++))
    fi

    if validate_script_path "$TEST_SCRIPT_SOURCE" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ PASS${NC}: Script path: accepts existing file"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}✗ FAIL${NC}: Script path: accepts existing file"
        ((TESTS_FAILED++))
    fi
}

test_script_executable_validation() {
    echo ""
    echo -e "${BLUE}=== Script Executable Validation ===${NC}"

    # Create non-executable script
    local non_exec_script="$TEST_TEMP_DIR/non-exec.sh"
    touch "$non_exec_script"
    chmod -x "$non_exec_script"

    if ! validate_script_executable "$non_exec_script" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ PASS${NC}: Script executable: rejects non-executable file"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}✗ FAIL${NC}: Script executable: rejects non-executable file"
        ((TESTS_FAILED++))
    fi

    if validate_script_executable "$TEST_SCRIPT_SOURCE" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ PASS${NC}: Script executable: accepts executable file"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}✗ FAIL${NC}: Script executable: accepts executable file"
        ((TESTS_FAILED++))
    fi

    rm -f "$non_exec_script"
}

test_script_readable_validation() {
    echo ""
    echo -e "${BLUE}=== Script Readable Validation ===${NC}"

    # Test readable script
    if validate_script_readable "$TEST_SCRIPT_SOURCE" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ PASS${NC}: Script readable: accepts readable file"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}✗ FAIL${NC}: Script readable: accepts readable file"
        ((TESTS_FAILED++))
    fi
}

# ============================================================================
# File Operation Tests
# ============================================================================

test_script_copy() {
    echo ""
    echo -e "${BLUE}=== Script Copy ===${NC}"

    local target_script="$TEST_STATUSLINES_DIR/my-test.sh"

    # Copy script
    if copy_script_to_statuslines "$TEST_SCRIPT_SOURCE" 'my-test' "$TEST_STATUSLINES_DIR" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ PASS${NC}: Script copy: copies script to statuslines directory"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}✗ FAIL${NC}: Script copy: copies script to statuslines directory"
        ((TESTS_FAILED++))
    fi

    # Verify target file exists
    if [ -f "$target_script" ]; then
        echo -e "${GREEN}✓ PASS${NC}: Script copy: target file exists"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}✗ FAIL${NC}: Script copy: target file exists"
        ((TESTS_FAILED++))
    fi

    # Verify target is executable
    if [ -x "$target_script" ]; then
        echo -e "${GREEN}✓ PASS${NC}: Script copy: target file is executable"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}✗ FAIL${NC}: Script copy: target file is executable"
        ((TESTS_FAILED++))
    fi
}

test_toggle_script_update() {
    echo ""
    echo -e "${BLUE}=== Toggle Script Update ===${NC}"

    local new_id="new-statusline"
    local new_desc="New Custom Statusline"
    local new_path="~/.claude/statuslines/new-statusline.sh"

    # Update toggle script
    if update_toggle_script "$new_id" "$new_path" "$new_desc" "$TEST_TOGGLE_SCRIPT" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ PASS${NC}: Toggle script: adds new entry"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}✗ FAIL${NC}: Toggle script: adds new entry"
        ((TESTS_FAILED++))
    fi

    # Verify entry was added
    if [ -f "$TEST_TOGGLE_SCRIPT" ] && grep -q "STATUSLINES\[\"$new_id\"\]" "$TEST_TOGGLE_SCRIPT"; then
        echo -e "${GREEN}✓ PASS${NC}: Toggle script: new ID found in script"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}✗ FAIL${NC}: Toggle script: new ID found in script"
        ((TESTS_FAILED++))
    fi

    if [ -f "$TEST_TOGGLE_SCRIPT" ] && grep -q "$new_desc" "$TEST_TOGGLE_SCRIPT"; then
        echo -e "${GREEN}✓ PASS${NC}: Toggle script: description found in script"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}✗ FAIL${NC}: Toggle script: description found in script"
        ((TESTS_FAILED++))
    fi
}

test_toggle_script_backup() {
    echo ""
    echo -e "${BLUE}=== Toggle Script Backup ===${NC}"

    local backup_file="${TEST_TOGGLE_SCRIPT}.bak"

    # Create backup before modification
    if create_backup "$TEST_TOGGLE_SCRIPT" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ PASS${NC}: Backup: creates backup file"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}✗ FAIL${NC}: Backup: creates backup file"
        ((TESTS_FAILED++))
    fi

    # Verify backup exists
    if [ -f "$backup_file" ]; then
        echo -e "${GREEN}✓ PASS${NC}: Backup: backup file exists"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}✗ FAIL${NC}: Backup: backup file exists"
        ((TESTS_FAILED++))
    fi
}

test_settings_update() {
    echo ""
    echo -e "${BLUE}=== Settings JSON Update ===${NC}"

    local new_command="~/.claude/statuslines/new-statusline.sh"

    # Update settings.json
    if update_settings_json "$new_command" "$TEST_SETTINGS_FILE" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ PASS${NC}: Settings update: updates statusLine.command in settings.json"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}✗ FAIL${NC}: Settings update: updates statusLine.command in settings.json"
        ((TESTS_FAILED++))
    fi

    # Verify command was updated
    if [ -f "$TEST_SETTINGS_FILE" ] && grep -q "$new_command" "$TEST_SETTINGS_FILE"; then
        echo -e "${GREEN}✓ PASS${NC}: Settings update: new command found in settings.json"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}✗ FAIL${NC}: Settings update: new command found in settings.json"
        ((TESTS_FAILED++))
    fi
}

# ============================================================================
# Integration Tests
# ============================================================================

test_register_existing_full_flow() {
    echo ""
    echo -e "${BLUE}=== Full Registration Flow ===${NC}"

    # This test simulates the full flow with mocked user input
    # We'll test that all components work together

    local test_id="integration-test"
    local test_desc="Integration Test Statusline"
    local test_path="$TEST_SCRIPT_SOURCE"
    local target_path="$TEST_STATUSLINES_DIR/$test_id.sh"

    # 1. Validate ID format
    if validate_id_format "$test_id" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ PASS${NC}: Full flow: ID format validation passes"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}✗ FAIL${NC}: Full flow: ID format validation passes"
        ((TESTS_FAILED++))
    fi

    # 2. Validate ID uniqueness
    if validate_id_uniqueness "$test_id" "$TEST_TOGGLE_SCRIPT" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ PASS${NC}: Full flow: ID uniqueness validation passes"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}✗ FAIL${NC}: Full flow: ID uniqueness validation passes"
        ((TESTS_FAILED++))
    fi

    # 3. Validate script path
    if validate_script_path "$test_path" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ PASS${NC}: Full flow: script path validation passes"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}✗ FAIL${NC}: Full flow: script path validation passes"
        ((TESTS_FAILED++))
    fi

    # 4. Validate script is executable
    if validate_script_executable "$test_path" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ PASS${NC}: Full flow: script executable validation passes"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}✗ FAIL${NC}: Full flow: script executable validation passes"
        ((TESTS_FAILED++))
    fi

    # 5. Copy script
    if copy_script_to_statuslines "$test_path" "$test_id" "$TEST_STATUSLINES_DIR" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ PASS${NC}: Full flow: script copy succeeds"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}✗ FAIL${NC}: Full flow: script copy succeeds"
        ((TESTS_FAILED++))
    fi

    # 6. Update toggle script
    if update_toggle_script "$test_id" "~/.claude/statuslines/$test_id.sh" "$test_desc" "$TEST_TOGGLE_SCRIPT" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ PASS${NC}: Full flow: toggle script update succeeds"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}✗ FAIL${NC}: Full flow: toggle script update succeeds"
        ((TESTS_FAILED++))
    fi

    # 7. Verify all changes
    if [ -f "$target_path" ]; then
        echo -e "${GREEN}✓ PASS${NC}: Full flow: registered script exists"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}✗ FAIL${NC}: Full flow: registered script exists"
        ((TESTS_FAILED++))
    fi

    if [ -x "$target_path" ]; then
        echo -e "${GREEN}✓ PASS${NC}: Full flow: registered script is executable"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}✗ FAIL${NC}: Full flow: registered script is executable"
        ((TESTS_FAILED++))
    fi

    if [ -f "$TEST_TOGGLE_SCRIPT" ] && grep -q "$test_id" "$TEST_TOGGLE_SCRIPT"; then
        echo -e "${GREEN}✓ PASS${NC}: Full flow: ID in toggle script"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}✗ FAIL${NC}: Full flow: ID in toggle script"
        ((TESTS_FAILED++))
    fi
}

test_register_existing_error_handling() {
    echo ""
    echo -e "${BLUE}=== Error Handling ===${NC}"

    # Test invalid ID format
    if ! validate_id_format 'invalid id' > /dev/null 2>&1; then
        echo -e "${GREEN}✓ PASS${NC}: Error handling: rejects invalid ID format"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}✗ FAIL${NC}: Error handling: rejects invalid ID format"
        ((TESTS_FAILED++))
    fi

    # Test duplicate ID
    if ! validate_id_uniqueness 'original' "$TEST_TOGGLE_SCRIPT" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ PASS${NC}: Error handling: rejects duplicate ID"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}✗ FAIL${NC}: Error handling: rejects duplicate ID"
        ((TESTS_FAILED++))
    fi

    # Test non-existent script
    if ! validate_script_path '/fake/path/script.sh' > /dev/null 2>&1; then
        echo -e "${GREEN}✓ PASS${NC}: Error handling: rejects non-existent script path"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}✗ FAIL${NC}: Error handling: rejects non-existent script path"
        ((TESTS_FAILED++))
    fi
}

# ============================================================================
# Source the script to test
# ============================================================================

# Source the statusline-add.sh script to test its functions
STATUSLINE_ADD_SCRIPT="$HOME/.claude/statusline-add.sh"

if [ -f "$STATUSLINE_ADD_SCRIPT" ]; then
    # Source without executing the main logic
    source "$STATUSLINE_ADD_SCRIPT" 2>/dev/null || true
else
    echo -e "${RED}Error: statusline-add.sh not found at $STATUSLINE_ADD_SCRIPT${NC}"
    echo "This test requires the statusline-add.sh script to be present."
    exit 1
fi

# ============================================================================
# Main Test Execution
# ============================================================================

run_tests() {
    echo ""
    echo "========================================================================"
    echo "Testing register_existing() Function in statusline-add.sh"
    echo "========================================================================"

    # Setup test environment
    setup_test_env

    # Run test groups
    test_id_format_valid
    test_id_format_invalid
    test_id_uniqueness
    test_script_path_validation
    test_script_executable_validation
    test_script_readable_validation
    test_script_copy
    test_toggle_script_update
    test_toggle_script_backup
    test_settings_update
    test_register_existing_full_flow
    test_register_existing_error_handling

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
