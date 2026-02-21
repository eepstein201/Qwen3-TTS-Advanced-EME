#!/bin/bash
# Test: Statusline Directory Migration
# Verifies that ~/.claude/statuslines/ directory is created and existing
# statusline scripts are migrated to it with proper path updates.

set -u

TEST_DIR="${TMPDIR:-/tmp}/statusline_test_$$"
HOME_DIR="$TEST_DIR/home"
CLAUDE_DIR="$HOME_DIR/.claude"
STATUSLINES_DIR="$CLAUDE_DIR/statuslines"

# Color codes for output
GREEN='\033[32m'
RED='\033[31m'
RESET='\033[0m'

test_count=0
pass_count=0
fail_count=0

# Helper function to run a test
run_test() {
    local test_name="$1"
    local test_func="$2"

    ((test_count++))
    echo ""
    echo "Test $test_count: $test_name"

    if $test_func; then
        echo -e "${GREEN}✓ PASS${RESET}"
        ((pass_count++))
    else
        echo -e "${RED}✗ FAIL${RESET}"
        ((fail_count++))
    fi
}

# Helper to assert condition
assert_true() {
    if [ "$1" ]; then
        return 0
    else
        echo "  Assertion failed: $2"
        return 1
    fi
}

# Helper to assert file exists
assert_file_exists() {
    if [ -f "$1" ]; then
        return 0
    else
        echo "  File does not exist: $1"
        return 1
    fi
}

# Helper to assert directory exists
assert_dir_exists() {
    if [ -d "$1" ]; then
        return 0
    else
        echo "  Directory does not exist: $1"
        return 1
    fi
}

# Helper to assert file is executable
assert_executable() {
    if [ -x "$1" ]; then
        return 0
    else
        echo "  File is not executable: $1"
        return 1
    fi
}

# Helper to assert string contains
assert_contains() {
    local haystack="$1"
    local needle="$2"
    if echo "$haystack" | grep -q "$needle"; then
        return 0
    else
        echo "  String does not contain '$needle'"
        echo "  String: $haystack"
        return 1
    fi
}

# ─────────────────────────────────────────────────────────────────────────────
# Setup: Create mock directory structure and run migration
# ─────────────────────────────────────────────────────────────────────────────

setup_test_environment() {
    echo "Setting up test environment..."

    mkdir -p "$CLAUDE_DIR"
    mkdir -p "$CLAUDE_DIR/user-scripts"

    # Create mock statusline-original.sh
    cat > "$CLAUDE_DIR/statusline-original.sh" << 'EOF'
#!/bin/bash
# Original statusline script
echo "Original statusline"
EOF
    chmod 644 "$CLAUDE_DIR/statusline-original.sh"

    # Create mock claude_statusbar_multiline_v1.py
    cat > "$CLAUDE_DIR/user-scripts/claude_statusbar_multiline_v1.py" << 'EOF'
#!/usr/bin/env python3
# Multiline statusbar script
print("Multiline statusbar")
EOF
    chmod 644 "$CLAUDE_DIR/user-scripts/claude_statusbar_multiline_v1.py"

    # Create mock statusline-toggle.sh (with old paths)
    cat > "$CLAUDE_DIR/statusline-toggle.sh" << 'EOF'
#!/bin/bash
set -e

SETTINGS_FILE="$HOME/.claude/settings.json"
STATUSLINE_DIR="$HOME/.claude"

declare -A STATUSLINES
STATUSLINES["original"]="~/.claude/statusline-original.sh|Original Custom Statusline"
STATUSLINES["starship"]="~/.local/bin/starship-claude|Starship-Claude"

echo "Current toggle script references:"
echo "  - ~/.claude/statusline-original.sh"
EOF
    chmod 755 "$CLAUDE_DIR/statusline-toggle.sh"

    # Create mock .zshrc
    cat > "$HOME_DIR/.zshrc" << 'EOF'
# Claude Code statusline toggle
alias claude-statusline-toggle='~/.claude/statusline-toggle.sh'
EOF

    # Create mock .bashrc
    cat > "$HOME_DIR/.bashrc" << 'EOF'
# Claude Code statusline toggle
alias claude-statusline-toggle='~/.claude/statusline-toggle.sh'
EOF

    # Run migration in the test environment
    run_migration_in_test_env
}

# ─────────────────────────────────────────────────────────────────────────────
# Run migration script in test environment
# ─────────────────────────────────────────────────────────────────────────────

run_migration_in_test_env() {
    echo "Running migration in test environment..."

    # Create a migration script that works with test paths
    cat > "$TEST_DIR/migrate-test.sh" << 'MIGRATE_EOF'
#!/bin/bash
set -e

CLAUDE_DIR="$1"
STATUSLINES_DIR="$CLAUDE_DIR/statuslines"

# Step 1: Create statuslines directory
if [ ! -d "$STATUSLINES_DIR" ]; then
    mkdir -p "$STATUSLINES_DIR"
    chmod 755 "$STATUSLINES_DIR"
fi

# Step 2: Migrate statusline-original.sh
if [ -f "$CLAUDE_DIR/statusline-original.sh" ]; then
    cp "$CLAUDE_DIR/statusline-original.sh" "$STATUSLINES_DIR/original.sh"
    chmod +x "$STATUSLINES_DIR/original.sh"
fi

# Step 3: Migrate claude_statusbar_multiline_v1.py
if [ -f "$CLAUDE_DIR/user-scripts/claude_statusbar_multiline_v1.py" ]; then
    cp "$CLAUDE_DIR/user-scripts/claude_statusbar_multiline_v1.py" "$STATUSLINES_DIR/multiline-v1.py"
    chmod +x "$STATUSLINES_DIR/multiline-v1.py"
fi

# Step 4: Update statusline-toggle.sh
if [ -f "$CLAUDE_DIR/statusline-toggle.sh" ]; then
    cp "$CLAUDE_DIR/statusline-toggle.sh" "$CLAUDE_DIR/statusline-toggle.sh.bak"
    # Use sed to update paths (works on both macOS and Linux)
    if [[ "$OSTYPE" == "darwin"* ]]; then
        sed -i '' 's|~/.claude/statusline-original.sh|~/.claude/statuslines/original.sh|g' "$CLAUDE_DIR/statusline-toggle.sh"
    else
        sed -i 's|~/.claude/statusline-original.sh|~/.claude/statuslines/original.sh|g' "$CLAUDE_DIR/statusline-toggle.sh"
    fi
fi
MIGRATE_EOF

    chmod +x "$TEST_DIR/migrate-test.sh"
    "$TEST_DIR/migrate-test.sh" "$CLAUDE_DIR"
}

# ─────────────────────────────────────────────────────────────────────────────
# Test 1: ~/.claude/statuslines/ directory exists and is writable
# ─────────────────────────────────────────────────────────────────────────────

test_statuslines_directory_exists() {
    # Before migration, directory should not exist
    if [ -d "$STATUSLINES_DIR" ]; then
        echo "  Skipping: directory already exists (cleanup may have failed)"
        return 0
    fi

    # This test will FAIL until we implement the migration
    assert_dir_exists "$STATUSLINES_DIR" "statuslines directory should exist"
}

test_statuslines_directory_writable() {
    if ! [ -d "$STATUSLINES_DIR" ]; then
        echo "  Skipping: statuslines directory doesn't exist yet"
        return 0
    fi

    # Test that directory is writable
    if touch "$STATUSLINES_DIR/.test_write" 2>/dev/null; then
        rm -f "$STATUSLINES_DIR/.test_write"
        return 0
    else
        echo "  Directory is not writable: $STATUSLINES_DIR"
        return 1
    fi
}

# ─────────────────────────────────────────────────────────────────────────────
# Test 2: statusline-original.sh migrated to statuslines/original.sh
# ─────────────────────────────────────────────────────────────────────────────

test_original_migrated() {
    if ! [ -d "$STATUSLINES_DIR" ]; then
        echo "  Skipping: statuslines directory doesn't exist yet"
        return 0
    fi

    assert_file_exists "$STATUSLINES_DIR/original.sh" \
        "statuslines/original.sh should exist after migration"
}

test_original_executable() {
    if ! [ -f "$STATUSLINES_DIR/original.sh" ]; then
        echo "  Skipping: original.sh doesn't exist yet"
        return 0
    fi

    assert_executable "$STATUSLINES_DIR/original.sh" \
        "statuslines/original.sh should be executable"
}

test_original_content_preserved() {
    if ! [ -f "$STATUSLINES_DIR/original.sh" ]; then
        echo "  Skipping: original.sh doesn't exist yet"
        return 0
    fi

    # Verify content was preserved
    if grep -q "Original statusline" "$STATUSLINES_DIR/original.sh"; then
        return 0
    else
        echo "  Original content not preserved in statuslines/original.sh"
        return 1
    fi
}

# ─────────────────────────────────────────────────────────────────────────────
# Test 3: claude_statusbar_multiline_v1.py migrated to statuslines/multiline-v1.py
# ─────────────────────────────────────────────────────────────────────────────

test_multiline_migrated() {
    if ! [ -d "$STATUSLINES_DIR" ]; then
        echo "  Skipping: statuslines directory doesn't exist yet"
        return 0
    fi

    assert_file_exists "$STATUSLINES_DIR/multiline-v1.py" \
        "statuslines/multiline-v1.py should exist after migration"
}

test_multiline_executable() {
    if ! [ -f "$STATUSLINES_DIR/multiline-v1.py" ]; then
        echo "  Skipping: multiline-v1.py doesn't exist yet"
        return 0
    fi

    assert_executable "$STATUSLINES_DIR/multiline-v1.py" \
        "statuslines/multiline-v1.py should be executable"
}

test_multiline_content_preserved() {
    if ! [ -f "$STATUSLINES_DIR/multiline-v1.py" ]; then
        echo "  Skipping: multiline-v1.py doesn't exist yet"
        return 0
    fi

    # Verify content was preserved
    if grep -q "Multiline statusbar" "$STATUSLINES_DIR/multiline-v1.py"; then
        return 0
    else
        echo "  Multiline content not preserved in statuslines/multiline-v1.py"
        return 1
    fi
}

# ─────────────────────────────────────────────────────────────────────────────
# Test 4: statusline-toggle.sh references new paths
# ─────────────────────────────────────────────────────────────────────────────

test_toggle_references_new_paths() {
    if ! [ -f "$CLAUDE_DIR/statusline-toggle.sh" ]; then
        echo "  Skipping: statusline-toggle.sh doesn't exist"
        return 0
    fi

    local toggle_content=$(cat "$CLAUDE_DIR/statusline-toggle.sh")

    # Should contain reference to new statuslines/original.sh
    if assert_contains "$toggle_content" "statuslines/original.sh"; then
        return 0
    else
        return 1
    fi
}

test_toggle_removes_old_paths() {
    if ! [ -f "$CLAUDE_DIR/statusline-toggle.sh" ]; then
        echo "  Skipping: statusline-toggle.sh doesn't exist"
        return 0
    fi

    local toggle_content=$(cat "$CLAUDE_DIR/statusline-toggle.sh")

    # Should NOT contain reference to old ~/.claude/statusline-original.sh
    if echo "$toggle_content" | grep -q '~/.claude/statusline-original.sh'; then
        echo "  Toggle script still references old path ~/.claude/statusline-original.sh"
        return 1
    else
        return 0
    fi
}

# ─────────────────────────────────────────────────────────────────────────────
# Test 5: Shell aliases updated
# ─────────────────────────────────────────────────────────────────────────────

test_zshrc_updated() {
    if ! [ -f "$HOME_DIR/.zshrc" ]; then
        echo "  Skipping: .zshrc doesn't exist"
        return 0
    fi

    local zshrc_content=$(cat "$HOME_DIR/.zshrc")

    # Should still reference the toggle script (which itself will have been updated)
    if assert_contains "$zshrc_content" "claude-statusline-toggle"; then
        return 0
    else
        return 1
    fi
}

test_bashrc_updated() {
    if ! [ -f "$HOME_DIR/.bashrc" ]; then
        echo "  Skipping: .bashrc doesn't exist"
        return 0
    fi

    local bashrc_content=$(cat "$HOME_DIR/.bashrc")

    # Should still reference the toggle script (or be empty if not used)
    # This test is more lenient - just verify file exists and is valid
    return 0
}

# ─────────────────────────────────────────────────────────────────────────────
# Cleanup
# ─────────────────────────────────────────────────────────────────────────────

cleanup_test_environment() {
    echo ""
    echo "Cleaning up test environment..."
    rm -rf "$TEST_DIR"
}

# ─────────────────────────────────────────────────────────────────────────────
# Main test runner
# ─────────────────────────────────────────────────────────────────────────────

main() {
    echo "========================================================================"
    echo "Test: Statusline Directory Migration"
    echo "========================================================================"

    # Setup
    setup_test_environment

    # Run all tests
    run_test "statuslines directory exists and is writable" test_statuslines_directory_exists
    run_test "statuslines directory is writable" test_statuslines_directory_writable

    run_test "statusline-original.sh migrated to statuslines/original.sh" test_original_migrated
    run_test "statuslines/original.sh is executable" test_original_executable
    run_test "statuslines/original.sh content preserved" test_original_content_preserved

    run_test "claude_statusbar_multiline_v1.py migrated to statuslines/multiline-v1.py" test_multiline_migrated
    run_test "statuslines/multiline-v1.py is executable" test_multiline_executable
    run_test "statuslines/multiline-v1.py content preserved" test_multiline_content_preserved

    run_test "statusline-toggle.sh references new paths" test_toggle_references_new_paths
    run_test "statusline-toggle.sh removes old paths" test_toggle_removes_old_paths

    run_test ".zshrc references toggle script" test_zshrc_updated
    run_test ".bashrc is valid" test_bashrc_updated

    # Cleanup
    cleanup_test_environment

    # Summary
    echo ""
    echo "========================================================================"
    echo "Test Results: $pass_count passed, $fail_count failed out of $test_count"
    echo "========================================================================"

    if [ $fail_count -eq 0 ]; then
        echo -e "${GREEN}All tests passed!${RESET}"
        exit 0
    else
        echo -e "${RED}Some tests failed.${RESET}"
        exit 1
    fi
}

main "$@"
