#!/bin/bash
# Batch test runner for Qwen3-TTS
# Runs tests in isolated groups to prevent hangs from cascading
# Usage: ./tests/run_batches.sh [--timeout SECONDS] [--batch N] [--continue]

set -e

TIMEOUT=300
CONTINUE=0
SPECIFIC_BATCH=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --timeout)
      TIMEOUT="$2"
      shift 2
      ;;
    --batch)
      SPECIFIC_BATCH="$2"
      shift 2
      ;;
    --continue|-c)
      CONTINUE=1
      shift
      ;;
    --help|-h)
      echo "Usage: $0 [OPTIONS]"
      echo "Options:"
      echo "  --timeout SECONDS  Timeout per batch (default: 300)"
      echo "  --batch N          Run only batch N (1-5)"
      echo "  --continue, -c     Continue on failure (run all batches)"
      echo "  --help, -h         Show this message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Array of batches: "name|test_modules"
BATCHES=(
  "1:Core Utilities|tests.test_audio_utils tests.test_text_processing tests.test_package_metadata tests.test_deprecated_refs tests.test_config"
  "2:Voice & CLI|tests.test_voice tests.test_cli_daemonization tests.test_caching tests.test_server_helpers"
  "3:Server Infrastructure|tests.test_fastapi_server tests.test_fastapi_endpoints tests.test_client"
  "4:Engine & UI|tests.test_engine tests.test_generate_server_fallback tests.test_ui_headless"
  "5:Optional Tests|tests.test_flash_attn_install"
)

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

FAILED_BATCHES=()
PASSED_BATCHES=()
SKIPPED_BATCHES=()

run_batch() {
  local batch_num="$1"
  local batch_name="$2"
  local test_modules="$3"

  echo -e "${BLUE}========================================${NC}"
  echo -e "${BLUE}Batch $batch_num: $batch_name${NC}"
  echo -e "${BLUE}========================================${NC}"

  if command -v timeout &> /dev/null; then
    timeout "$TIMEOUT" python -m unittest $test_modules -v
  else
    python -m unittest $test_modules -v
  fi

  return $?
}

# Run batches
for batch in "${BATCHES[@]}"; do
  IFS='|' read -r num name modules <<< "$batch"

  if [[ -n "$SPECIFIC_BATCH" ]] && [[ "$num" != "$SPECIFIC_BATCH" ]]; then
    SKIPPED_BATCHES+=("$num: $name")
    continue
  fi

  if run_batch "$num" "$name" "$modules"; then
    echo -e "${GREEN}✓ Batch $num passed${NC}\n"
    PASSED_BATCHES+=("$num: $name")
  else
    echo -e "${RED}✗ Batch $num failed${NC}\n"
    FAILED_BATCHES+=("$num: $name")

    if [[ $CONTINUE -eq 0 ]]; then
      echo -e "${RED}Stopping due to failure. Use --continue to run all batches.${NC}"
      break
    fi
  fi
done

# Summary
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Summary${NC}"
echo -e "${BLUE}========================================${NC}"

if [[ ${#PASSED_BATCHES[@]} -gt 0 ]]; then
  echo -e "${GREEN}Passed batches:${NC}"
  for batch in "${PASSED_BATCHES[@]}"; do
    echo -e "  ${GREEN}✓${NC} $batch"
  done
fi

if [[ ${#FAILED_BATCHES[@]} -gt 0 ]]; then
  echo -e "${RED}Failed batches:${NC}"
  for batch in "${FAILED_BATCHES[@]}"; do
    echo -e "  ${RED}✗${NC} $batch"
  done
fi

if [[ ${#SKIPPED_BATCHES[@]} -gt 0 ]]; then
  echo -e "${YELLOW}Skipped batches:${NC}"
  for batch in "${SKIPPED_BATCHES[@]}"; do
    echo -e "  ${YELLOW}○${NC} $batch"
  done
fi

echo -e "${BLUE}========================================${NC}"

# Exit with error if any batch failed
if [[ ${#FAILED_BATCHES[@]} -gt 0 ]]; then
  exit 1
fi

exit 0
