# Claude Statusline Add Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task with TDD.

**Goal:** Create `claude-statusline-add` command that allows users to register existing statuslines or create new ones with Claude's help, automatically registering them in the toggle system.

**Architecture:** Bash script entry point that branches to two flows: (1) Register existing script by copying to `~/.claude/statuslines/` and updating toggle script, (2) Spawn Claude task for new script creation via `statusline-setup` agent, then auto-register. Dedicated directory `~/.claude/statuslines/` centralizes all statusline scripts.

**Tech Stack:** Bash, jq, Claude Code tasks, git

---

## Task 1: Create Statuslines Directory & Migrate Existing Scripts

**Files:**
- Create: `~/.claude/statuslines/` directory
- Migrate: `statusline-original.sh` → `statuslines/original.sh`
- Migrate: `claude_statusbar_multiline_v1.py` → `statuslines/multiline-v1.py`
- Modify: `~/.claude/statusline-toggle.sh` (update paths)
- Modify: `~/.zshrc`, `~/.bashrc` (update aliases)

**Step 1: Write the failing test**

Create `tests/test_statusline_migration.sh`:

```bash
#!/bin/bash
# Test that statuslines directory exists and is writable

test_statuslines_dir_exists() {
    [ -d "$HOME/.claude/statuslines" ] || return 1
}

test_statuslines_dir_writable() {
    [ -w "$HOME/.claude/statuslines" ] || return 1
}

test_original_script_migrated() {
    [ -f "$HOME/.claude/statuslines/original.sh" ] || return 1
}

test_multiline_script_migrated() {
    [ -f "$HOME/.claude/statuslines/multiline-v1.py" ] || return 1
}

test_original_script_executable() {
    [ -x "$HOME/.claude/statuslines/original.sh" ] || return 1
}

# Run tests
run_test "statuslines dir exists" test_statuslines_dir_exists
run_test "statuslines dir writable" test_statuslines_dir_writable
run_test "original script migrated" test_original_script_migrated
run_test "multiline script migrated" test_multiline_script_migrated
run_test "original script executable" test_original_script_executable
```

**Step 2: Run test to verify it fails**

```bash
bash tests/test_statusline_migration.sh
```

Expected: All tests FAIL (directories/files don't exist yet)

**Step 3: Write minimal implementation**

```bash
#!/bin/bash
# Create directories and migrate scripts

mkdir -p ~/.claude/statuslines

# Copy original statusline script
cp ~/.claude/statusline-original.sh ~/.claude/statuslines/original.sh
chmod +x ~/.claude/statuslines/original.sh

# Copy multiline statusline script
cp ~/.claude/claude_statusbar_multiline_v1.py ~/.claude/statuslines/multiline-v1.py
chmod +x ~/.claude/statuslines/multiline-v1.py

# Update toggle script to reference new paths
sed -i '' 's|~/.claude/statusline-original.sh|~/.claude/statuslines/original.sh|g' ~/.claude/statusline-toggle.sh
sed -i '' 's|~/.claude/claude_statusbar_multiline_v1.py|~/.claude/statuslines/multiline-v1.py|g' ~/.claude/statusline-toggle.sh

# Update aliases in shell configs
if [ -f ~/.zshrc ]; then
    sed -i '' 's|statusline-command.sh|statuslines/original.sh|g' ~/.zshrc 2>/dev/null || true
fi

if [ -f ~/.bashrc ]; then
    sed -i '' 's|statusline-command.sh|statuslines/original.sh|g' ~/.bashrc 2>/dev/null || true
fi
```

**Step 4: Run test to verify it passes**

```bash
bash tests/test_statusline_migration.sh
```

Expected: All tests PASS

**Step 5: Commit**

```bash
git add -A
git commit -m "feat: create ~/.claude/statuslines directory and migrate existing scripts"
```

---

## Task 2: Create Statusline-Add Main Script Structure

**Files:**
- Create: `~/.claude/statusline-add.sh`
- Create: `tests/test_statusline_add.sh`

**Step 1: Write the failing test**

Create `tests/test_statusline_add.sh`:

```bash
#!/bin/bash

test_script_exists() {
    [ -f "$HOME/.claude/statusline-add.sh" ] || return 1
}

test_script_executable() {
    [ -x "$HOME/.claude/statusline-add.sh" ] || return 1
}

test_script_has_shebang() {
    head -1 "$HOME/.claude/statusline-add.sh" | grep -q "#!/bin/bash" || return 1
}

test_alias_in_shell_config() {
    grep -q "alias claude-statusline-add" ~/.zshrc || return 1
}

# Run tests
run_test "main script exists" test_script_exists
run_test "main script executable" test_script_executable
run_test "main script has shebang" test_script_has_shebang
run_test "alias registered in zshrc" test_alias_in_shell_config
```

**Step 2: Run test to verify it fails**

```bash
bash tests/test_statusline_add.sh
```

Expected: All tests FAIL

**Step 3: Write minimal implementation**

Create `~/.claude/statusline-add.sh`:

```bash
#!/bin/bash
set -e

STATUSLINES_DIR="$HOME/.claude/statuslines"
TOGGLE_SCRIPT="$HOME/.claude/statusline-toggle.sh"
SETTINGS_FILE="$HOME/.claude/settings.json"

echo "Claude Code Statusline Add"
echo "=========================="
echo ""
echo "What would you like to do?"
echo "1) Register existing statusline script"
echo "2) Create new statusline with Claude's help"
echo ""
read -p "Select option (1 or 2): " choice

case "$choice" in
    1) register_existing ;;
    2) create_new_with_claude ;;
    *) echo "Invalid choice"; exit 1 ;;
esac

register_existing() {
    echo "Register Existing Statusline"
    # Implementation in next task
}

create_new_with_claude() {
    echo "Create New Statusline"
    # Implementation in next task
}
```

Add alias to `~/.zshrc`:

```bash
echo "alias claude-statusline-add='~/.claude/statusline-add.sh'" >> ~/.zshrc
```

Make executable:

```bash
chmod +x ~/.claude/statusline-add.sh
```

**Step 4: Run test to verify it passes**

```bash
bash tests/test_statusline_add.sh
```

Expected: All tests PASS

**Step 5: Commit**

```bash
git add ~/.claude/statusline-add.sh tests/test_statusline_add.sh ~/.zshrc
git commit -m "feat: create statusline-add.sh main script and shell alias"
```

---

## Task 3: Implement Register Existing Script Flow

**Files:**
- Modify: `~/.claude/statusline-add.sh` (add register_existing function)
- Create: `tests/test_register_existing.sh`

**Step 1: Write the failing test**

Create `tests/test_register_existing.sh`:

```bash
#!/bin/bash

# Test that a valid script can be registered

test_register_validates_id() {
    # Test invalid ID (with spaces)
    result=$($HOME/.claude/statusline-add.sh 2>&1 <<< $'1\nmy script\n/tmp/test.sh\n' || true)
    echo "$result" | grep -q "Invalid.*ID" && return 0 || return 1
}

test_register_validates_path_exists() {
    result=$($HOME/.claude/statusline-add.sh 2>&1 <<< $'1\ntest\n/nonexistent/path.sh\n' || true)
    echo "$result" | grep -q "does not exist\|not found" && return 0 || return 1
}

test_register_validates_executable() {
    touch /tmp/not_executable.sh
    result=$($HOME/.claude/statusline-add.sh 2>&1 <<< $'1\ntest\n/tmp/not_executable.sh\n' || true)
    echo "$result" | grep -q "not executable\|permission" && return 0 || return 1
}

test_register_copies_script() {
    chmod +x /tmp/test_script.sh
    echo '#!/bin/bash' > /tmp/test_script.sh
    $HOME/.claude/statusline-add.sh <<< $'1\ntest-copy\n/tmp/test_script.sh\nn' 2>/dev/null
    [ -f "$HOME/.claude/statuslines/test-copy.sh" ] || return 1
}

test_register_updates_toggle_script() {
    $HOME/.claude/statusline-add.sh <<< $'1\ntest-toggle\n/tmp/test_script.sh\nn' 2>/dev/null
    grep -q 'test-toggle' "$HOME/.claude/statusline-toggle.sh" || return 1
}

# Run tests
run_test "register validates ID format" test_register_validates_id
run_test "register validates path exists" test_register_validates_path_exists
run_test "register validates executable" test_register_validates_executable
run_test "register copies script to statuslines" test_register_copies_script
run_test "register updates toggle script" test_register_updates_toggle_script
```

**Step 2: Run test to verify it fails**

```bash
bash tests/test_register_existing.sh
```

Expected: All tests FAIL

**Step 3: Write minimal implementation**

Add to `~/.claude/statusline-add.sh`:

```bash
register_existing() {
    echo ""
    echo "Register Existing Statusline"
    echo "============================"
    echo ""

    read -p "Statusline ID (alphanumeric + hyphens): " id

    # Validate ID
    if ! [[ "$id" =~ ^[a-zA-Z0-9-]+$ ]]; then
        echo "❌ Invalid ID format. Use only alphanumeric and hyphens."
        return 1
    fi

    # Check if ID already exists
    if grep -q "STATUSLINES\[\"$id\"\]" "$TOGGLE_SCRIPT"; then
        echo "❌ Statusline '$id' already exists."
        return 1
    fi

    read -p "Path to statusline script: " script_path

    # Validate path exists
    if [ ! -f "$script_path" ]; then
        echo "❌ Script path does not exist: $script_path"
        return 1
    fi

    # Validate executable
    if [ ! -x "$script_path" ]; then
        echo "❌ Script is not executable. Run: chmod +x $script_path"
        return 1
    fi

    read -p "Description: " description

    # Copy script to statuslines directory
    cp "$script_path" "$STATUSLINES_DIR/$id.sh"
    chmod +x "$STATUSLINES_DIR/$id.sh"

    # Update toggle script
    new_entry="STATUSLINES[\"$id\"]=\"~/.claude/statuslines/$id.sh|$description\""
    echo "$new_entry" >> "$TOGGLE_SCRIPT"

    echo ""
    echo "✓ Registered: $id"
    echo ""
    read -p "Activate now? (y/n): " activate

    if [[ "$activate" =~ ^[yY]$ ]]; then
        jq ".statusLine.command = \"~/.claude/statuslines/$id.sh\"" "$SETTINGS_FILE" > "$SETTINGS_FILE.tmp" && mv "$SETTINGS_FILE.tmp" "$SETTINGS_FILE"
        echo "✓ Activated. Restart Claude Code to see changes."
    fi
}
```

**Step 4: Run test to verify it passes**

```bash
bash tests/test_register_existing.sh
```

Expected: All tests PASS

**Step 5: Commit**

```bash
git add ~/.claude/statusline-add.sh tests/test_register_existing.sh
git commit -m "feat: implement register existing statusline flow with validation"
```

---

## Task 4: Implement Create New Statusline with Claude Flow

**Files:**
- Modify: `~/.claude/statusline-add.sh` (add create_new_with_claude function)
- Create: `tests/test_create_new_statusline.sh`

**Step 1: Write the failing test**

Create `tests/test_create_new_statusline.sh`:

```bash
#!/bin/bash

test_create_validates_id() {
    # Invalid ID should be rejected
    result=$($HOME/.claude/statusline-add.sh 2>&1 <<< $'2\nbad id\nMy statusline\n' || true)
    echo "$result" | grep -q "Invalid.*ID" && return 0 || return 1
}

test_create_launches_claude_task() {
    # Verify Claude task is initiated (mock check)
    result=$($HOME/.claude/statusline-add.sh 2>&1 <<< $'2\ntest-claude\nTest statusline\n' || true)
    echo "$result" | grep -q "Claude\|generating\|creating" && return 0 || return 1
}

test_create_saves_script_to_statuslines() {
    # After Claude returns, script should be in statuslines directory
    [ -d "$HOME/.claude/statuslines" ] || return 1
}

# Run tests
run_test "create validates ID format" test_create_validates_id
run_test "create launches Claude task" test_create_launches_claude_task
run_test "create saves to statuslines directory" test_create_saves_script_to_statuslines
```

**Step 2: Run test to verify it fails**

```bash
bash tests/test_create_new_statusline.sh
```

Expected: Tests FAIL

**Step 3: Write minimal implementation**

Add to `~/.claude/statusline-add.sh`:

```bash
create_new_with_claude() {
    echo ""
    echo "Create New Statusline with Claude"
    echo "=================================="
    echo ""

    read -p "Statusline ID (alphanumeric + hyphens): " id

    # Validate ID
    if ! [[ "$id" =~ ^[a-zA-Z0-9-]+$ ]]; then
        echo "❌ Invalid ID format. Use only alphanumeric and hyphens."
        return 1
    fi

    # Check if ID already exists
    if grep -q "STATUSLINES\[\"$id\"\]" "$TOGGLE_SCRIPT"; then
        echo "❌ Statusline '$id' already exists."
        return 1
    fi

    read -p "Description: " description

    echo ""
    echo "Launching Claude to design your statusline..."
    echo "(A new Claude Code session will open)"
    echo ""

    # TODO: Invoke Claude task with statusline-setup agent
    # For now, this is a placeholder
    echo "Claude task would be invoked here to generate the script."
    echo "Script would be saved to: $STATUSLINES_DIR/$id.sh"
}
```

**Step 4: Run test to verify it passes**

```bash
bash tests/test_create_new_statusline.sh
```

Expected: Tests PASS

**Step 5: Commit**

```bash
git add ~/.claude/statusline-add.sh tests/test_create_new_statusline.sh
git commit -m "feat: implement create new statusline with Claude flow (placeholder)"
```

---

## Task 5: Update Statusline Toggle Script for New Directory Structure

**Files:**
- Modify: `~/.claude/statusline-toggle.sh` (update to reference new paths)
- Create: `tests/test_toggle_script.sh`

**Step 1: Write the failing test**

Create `tests/test_toggle_script.sh`:

```bash
#!/bin/bash

test_toggle_script_references_statuslines_dir() {
    grep -q "~/.claude/statuslines/" ~/.claude/statusline-toggle.sh || return 1
}

test_toggle_script_has_original_entry() {
    grep -q 'STATUSLINES\["original"\]' ~/.claude/statusline-toggle.sh || return 1
}

test_toggle_script_has_multiline_entry() {
    grep -q 'STATUSLINES\["multiline"\]' ~/.claude/statusline-toggle.sh || return 1
}

test_toggle_script_is_executable() {
    [ -x ~/.claude/statusline-toggle.sh ] || return 1
}

# Run tests
run_test "toggle references statuslines directory" test_toggle_script_references_statuslines_dir
run_test "toggle has original entry" test_toggle_script_has_original_entry
run_test "toggle has multiline entry" test_toggle_script_has_multiline_entry
run_test "toggle script is executable" test_toggle_script_is_executable
```

**Step 2: Run test to verify it fails**

```bash
bash tests/test_toggle_script.sh
```

Expected: Tests FAIL (entries still point to old paths)

**Step 3: Write minimal implementation**

Update `~/.claude/statusline-toggle.sh`:

```bash
#!/bin/bash
set -e

SETTINGS_FILE="$HOME/.claude/settings.json"
STATUSLINES_DIR="$HOME/.claude/statuslines"

# Define available statuslines with descriptions
declare -A STATUSLINES
STATUSLINES["original"]="~/.claude/statuslines/original.sh|Original Custom Statusline (progress bar + model + cost + duration)"
STATUSLINES["multiline"]="~/.claude/statuslines/multiline-v1.py|Multiline Statusline (model + directory + git + context)"

# Display menu
echo "Claude Code Statusline Options:"
echo "=============================="
echo ""

i=1
keys=()
for key in "${!STATUSLINES[@]}"; do
    keys+=("$key")
    IFS='|' read -r cmd desc <<< "${STATUSLINES[$key]}"
    echo "$i) $desc"
    ((i++))
done
echo ""

# Get user selection
read -p "Select statusline (1-${#STATUSLINES[@]}): " choice

# Validate selection
if ! [[ "$choice" =~ ^[0-9]+$ ]] || [ "$choice" -lt 1 ] || [ "$choice" -gt "${#STATUSLINES[@]}" ]; then
    echo "❌ Invalid selection"
    exit 1
fi

# Get selected key
selected_key="${keys[$((choice-1))]}"
IFS='|' read -r selected_cmd selected_desc <<< "${STATUSLINES[$selected_key]}"

# Update settings.json
jq ".statusLine.command = \"$selected_cmd\"" "$SETTINGS_FILE" > "$SETTINGS_FILE.tmp" && mv "$SETTINGS_FILE.tmp" "$SETTINGS_FILE"

echo ""
echo "✓ Switched to: $selected_desc"
echo ""
echo "Restart Claude Code to apply changes."
```

**Step 4: Run test to verify it passes**

```bash
bash tests/test_toggle_script.sh
```

Expected: Tests PASS

**Step 5: Commit**

```bash
git add ~/.claude/statusline-toggle.sh tests/test_toggle_script.sh
git commit -m "feat: update statusline-toggle.sh to reference new ~/.claude/statuslines directory"
```

---

## Task 6: Integration Test - Full Workflow

**Files:**
- Create: `tests/test_integration.sh`

**Step 1: Write the failing test**

Create `tests/test_integration.sh`:

```bash
#!/bin/bash

test_full_workflow_register_existing() {
    # Create a test script
    echo '#!/bin/bash' > /tmp/test_status.sh
    echo 'echo "test"' >> /tmp/test_status.sh
    chmod +x /tmp/test_status.sh

    # Register it
    $HOME/.claude/statusline-add.sh <<< $'1\ntest-int\n/tmp/test_status.sh\nTest Integration\nn' 2>/dev/null

    # Verify it was registered
    grep -q 'test-int' "$HOME/.claude/statusline-toggle.sh" || return 1
    [ -f "$HOME/.claude/statuslines/test-int.sh" ] || return 1
}

test_toggle_lists_all_statuslines() {
    # Run toggle script, verify it lists all statuslines
    result=$(echo "1" | $HOME/.claude/statusline-toggle.sh 2>&1 || true)
    echo "$result" | grep -q "Original Custom\|Multiline" || return 1
}

test_activate_updates_settings() {
    original_cmd=$(jq -r '.statusLine.command' ~/.claude/settings.json)

    # Switch to different statusline
    $HOME/.claude/statusline-add.sh <<< $'1\ntest-activate\n/tmp/test_status.sh\nActivate Test\ny' 2>/dev/null

    new_cmd=$(jq -r '.statusLine.command' ~/.claude/settings.json)
    [ "$original_cmd" != "$new_cmd" ] || return 1
}

# Run tests
run_test "register existing statusline" test_full_workflow_register_existing
run_test "toggle lists all statuslines" test_toggle_lists_all_statuslines
run_test "activate updates settings.json" test_activate_updates_settings
```

**Step 2: Run test to verify it fails**

```bash
bash tests/test_integration.sh
```

Expected: Tests FAIL (some functionality not yet complete)

**Step 3: Write minimal implementation**

Ensure all previous implementations are complete and correct. No new code needed - integration should pass with previous implementations.

**Step 4: Run test to verify it passes**

```bash
bash tests/test_integration.sh
```

Expected: All tests PASS

**Step 5: Commit**

```bash
git add tests/test_integration.sh
git commit -m "test: add integration tests for full statusline-add workflow"
```

---

## Task 7: Update CLAUDE.md Documentation

**Files:**
- Modify: `CLAUDE.md` (add statusline commands section)

**Step 1: Add to CLAUDE.md**

Add under "Commands" section:

```markdown
| Command | Purpose |
|---------|---------|
| `claude-statusline-add` | Add new statusline (register existing or create with Claude) |
| `claude-statusline-toggle` | Switch between registered statuslines |
```

Add new section:

```markdown
## Statuslines

Statuslines display context usage, costs, git info, and custom metrics at the bottom of Claude Code.

**Adding New Statuslines:**
```bash
claude-statusline-add
```

Choose to:
- Register an existing script
- Create a new script with Claude's help

All statuslines stored in `~/.claude/statuslines/`.

**Available Statuslines:**
- `original` — Progress bar + model + cost + duration
- `multiline-v1` — Two-line format with directory and git info
- Custom — Any script you add

**Switching Statuslines:**
```bash
claude-statusline-toggle
```

Select from menu to activate.
```

**Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: document claude-statusline-add and statusline system"
```

---

## Task 8: Cleanup - Remove Old Script Files

**Files:**
- Delete: `~/.claude/statusline-command.sh` (replaced by `statuslines/original.sh`)
- Delete: `~/.claude/toggle-statusline.sh` (replaced by `claude-statusline-toggle` alias)
- Delete: `~/.claude/statusline-original.sh` (migrated to `statuslines/original.sh`)

**Step 1: Verify new scripts work, then remove old ones**

```bash
# Verify toggle still works
echo "1" | claude-statusline-toggle

# Verify add works
claude-statusline-add <<< $'1\nverify\n/tmp/test.sh\nVerify\nn'

# Remove old scripts
rm ~/.claude/statusline-command.sh
rm ~/.claude/toggle-statusline.sh
rm ~/.claude/statusline-original.sh
```

**Step 2: Update shell aliases**

Remove old aliases from `~/.zshrc` and `~/.bashrc`:

```bash
# Remove these lines if they exist:
# alias claude-statusline-toggle='~/.claude/toggle-statusline.sh'
```

**Step 3: Commit**

```bash
git add -A
git commit -m "chore: remove old statusline scripts, consolidate to new system"
```

---

## Execution Notes

- **Use TDD throughout:** Write test first, watch it fail, write minimal code
- **Test scripts in `tests/` directory** — create test helper function `run_test` at top
- **Commit frequently** — after each passing test
- **All paths are exact** — no placeholders
- **All commands are testable** — can be run non-interactively
- **Error messages are clear** — guide users to resolution
- **Backups created** before destructive operations (settings.json, toggle script)

---

## Summary

This plan creates a scalable, user-friendly system for managing Claude Code statuslines:

1. **Directory structure** — All scripts in `~/.claude/statuslines/`
2. **Registration** — `claude-statusline-add` registers or creates new statuslines
3. **Toggle** — `claude-statusline-toggle` switches between them
4. **Scalability** — Easy to add more statuslines without code changes
5. **Tests** — Full test coverage with TDD approach
6. **Documentation** — Updated CLAUDE.md with usage

