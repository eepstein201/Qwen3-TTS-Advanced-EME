#!/usr/bin/env bash
# scripts/upstream_watch.sh
#
# Monthly upstream change-detector for the Qwen3-TTS project.
# Reads a set of public-API signals, diffs them against the last-known state stored in
# the body of a single tracking GitHub issue, and posts a comment ONLY when a signal
# changed. The current state is written back into that issue's body (the issue is the
# durable state store, so this needs NO repo commits and NO contents:write permission).
#
# Signals (those that gate roadmap items R-28 / FUTURE-* / HIGH-*):
#   ⚡ sageattention_native_hf   SageAttention native HF attn_implementation (blocker)
#   ⚡ pcg_has_code              PCG (arXiv:2511.13732) reference code released (R-28 blocker)
#   ⚡ vllm_mainline_tts         vLLM mainline TTS / Qwen3-TTS support (HIGH-* blocker)
#   qwen_tts_pypi / _sha / _tag qwen_tts PyPI version + latest commit/tag (new release)
#   mlx_audio_tag / mlx_tag     mlx-audio + MLX core latest release (cache / spec-decode / quant)
#   qwen3_tts_models            Qwen3-TTS HuggingFace model inventory (new sizes / revisions)
#   mlx_community_models        mlx-community Qwen3-TTS model inventory
#   vllm_omni_tag               vLLM-omni latest release (Qwen3-TTS perf upstream)
#
# Requires: curl, jq, gh (all preinstalled on ubuntu-latest). Auth via GH_TOKEN.
set -uo pipefail

TITLE="📡 Upstream Watch — Qwen3-TTS / MLX / vLLM"
LABEL="upstream-watch"
STATE_BEGIN="<!-- upstream-watch-state"
STATE_END="-->"
SENTINEL="ERR"

GH_TOKEN="${GH_TOKEN:-${GITHUB_TOKEN:-}}"
export GH_TOKEN

log() { printf '%s\n' "$*" >&2; }
die() { log "ERROR: $*"; exit 1; }

# fetch <url> [extra curl args...] — prints body, or empty string on failure
fetch() {
  local url="$1"; shift || true
  curl -fsSL --max-time 30 "$@" "$url" 2>/dev/null || printf ''
}
# Authenticated GitHub API GET (the workflow token raises rate limits and reads public repos)
ghget() {
  fetch "$1" -H "Authorization: Bearer ${GH_TOKEN}" -H "Accept: application/vnd.github+json"
}
# jq scalar with ERR fallback on empty/invalid input
jv() {
  local v
  v="$(printf '%s' "$1" | jq -r "$2 // \"$SENTINEL\"" 2>/dev/null)" || true
  printf '%s' "${v:-$SENTINEL}"
}

command -v curl >/dev/null || die "curl not found"
command -v jq   >/dev/null || die "jq not found"
command -v gh   >/dev/null || die "gh not found"
[ -n "$GH_TOKEN" ] || die "GH_TOKEN/GITHUB_TOKEN not set"

log "Collecting upstream signals..."

# --- qwen_tts (torch path library) ---
qwen_pyi="$(jv "$(fetch https://pypi.org/pypi/qwen-tts/json)" '.info.version')"
qwen_sha="$(jv "$(ghget https://api.github.com/repos/QwenLM/Qwen3-TTS/commits/main)" '.sha')"
qwen_tag="$(jv "$(ghget https://api.github.com/repos/QwenLM/Qwen3-TTS/tags)" '.[0].name')"

# --- SageAttention native HF attn_implementation? (heuristic: doc mentions it) ---
if fetch https://huggingface.co/docs/transformers/main/en/attention_interface | grep -qi 'sageattention'; then
  sage="true"; else sage="false"; fi

# --- MLX family releases ---
mlx_audio_tag="$(jv "$(ghget https://api.github.com/repos/Blaizzy/mlx-audio/releases/latest)" '.tag_name')"
mlx_tag="$(jv "$(ghget https://api.github.com/repos/ml-explore/mlx/releases/latest)" '.tag_name')"

# --- PCG (arXiv:2511.13732): revision count + any code link ---
pcg_feed="$(fetch "http://export.arxiv.org/api/query?id_list=2511.13732")"
pcg_versions="$(printf '%s' "$pcg_feed" | grep -oE '2511\.13732v[0-9]+' | sort -u | wc -l | tr -d ' ')"
[ -n "${pcg_versions:-}" ] || pcg_versions=0
pcg_abs="$(fetch https://arxiv.org/abs/2511.13732)"
if printf '%s' "$pcg_abs" | grep -qiE 'github\.com|gitlab\.com|bitbucket\.org|paperswithcode'; then
  pcg_code="true"; else pcg_code="false"; fi

# --- HuggingFace model inventories (sorted id arrays) ---
qwen_models="$(fetch "https://huggingface.co/api/models?search=Qwen3-TTS&limit=50" \
  | jq -c 'map(.id) | sort' 2>/dev/null || printf '[]')"
mlx_models="$(fetch "https://huggingface.co/api/models?author=mlx-community&search=Qwen3-TTS&limit=50" \
  | jq -c 'map(.id) | sort' 2>/dev/null || printf '[]')"

# --- vLLM ---
vllm_omni_tag="$(jv "$(ghget https://api.github.com/repos/vllm-project/vllm-omni/releases/latest)" '.tag_name')"
if fetch https://docs.vllm.ai/en/latest/models/supported_models.html | grep -qiE 'qwen3-tts|text-to-speech'; then
  vllm_tts="true"; else vllm_tts="false"; fi

checked_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

state="$(jq -nc \
  --arg a "$qwen_pyi" --arg b "$qwen_sha" --arg c "$qwen_tag" \
  --argjson d "$sage" --arg e "$mlx_audio_tag" --arg f "$mlx_tag" \
  --argjson g "$pcg_versions" --argjson h "$pcg_code" \
  --argjson i "$qwen_models" --argjson j "$mlx_models" \
  --arg k "$vllm_omni_tag" --argjson l "$vllm_tts" --arg m "$checked_at" \
  '{qwen_tts_pypi:$a, qwen_tts_sha:$b, qwen_tts_tag:$c,
    sageattention_native_hf:$d, mlx_audio_tag:$e, mlx_tag:$f,
    pcg_versions:$g, pcg_has_code:$h,
    qwen3_tts_models:$i, mlx_community_models:$j,
    vllm_omni_tag:$k, vllm_mainline_tts:$l, checked_at:$m}')" \
  || die "failed to assemble state"

log "State: $state"

# --- ensure the label exists (ignore error if it already does) ---
gh label create "$LABEL" --color 0a6fc5 \
  --description "Upstream change-detector (Qwen3-TTS / MLX / vLLM)" >/dev/null 2>&1 || true

# --- find the single tracking issue by label ---
issue_num="$(gh issue list --state open --label "$LABEL" --json number -q '.[0].number' 2>/dev/null || printf '')"

label_for() {  # internal key -> human label; ⚡ prefix marks blocker-clearing signals
  case "$1" in
    sageattention_native_hf) printf '⚡ SageAttention native HF attn_implementation (blocker)' ;;
    pcg_has_code)            printf '⚡ PCG reference code released — arXiv:2511.13732 (R-28 blocker)' ;;
    vllm_mainline_tts)       printf '⚡ vLLM mainline TTS/Qwen3-TTS support (HIGH-* blocker)' ;;
    qwen_tts_pyi)            printf 'qwen_tts PyPI version' ;;
    qwen_tts_sha)            printf 'qwen_tts latest commit (main)' ;;
    qwen_tts_tag)            printf 'qwen_tts latest tag' ;;
    mlx_audio_tag)           printf 'mlx-audio release' ;;
    mlx_tag)                 printf 'MLX core release' ;;
    pcg_versions)            printf 'PCG paper revisions (arXiv:2511.13732)' ;;
    qwen3_tts_models)        printf 'Qwen3-TTS HuggingFace models' ;;
    mlx_community_models)    printf 'mlx-community Qwen3-TTS models' ;;
    vllm_omni_tag)           printf 'vLLM-omni release' ;;
    *) printf '%s' "$1" ;;
  esac
}

render_body() {  # $1 = compact state json
  local s="$1"
  printf '%s\n' \
'**Living dashboard for the upstream change-detector** (`.github/workflows/upstream-watch.yml`, monthly).' \
'' \
'A new comment is posted only when a signal changes. No comment = no movement. Re-run the deep-research sweep (quarterly, or on demand) if a ⚡ blocker clears.' \
'' \
'---' \
'' \
"**Current state** (`checked_at` in JSON):" \
'' \
'```json'
  printf '%s\n' "$s"
  printf '%s\n' '```' '' "$STATE_BEGIN"
  printf '%s\n' "$s"
  printf '%s\n' "$STATE_END"
}

# --- first run: establish baseline (no change to report) ---
if [ -z "$issue_num" ]; then
  log "No tracking issue found — establishing baseline."
  gh issue create --title "$TITLE" --body "$(render_body "$state")" --label "$LABEL" >/dev/null \
    || die "failed to create tracking issue"
  log "Baseline issue created. Exiting (no change to report)."
  exit 0
fi

# --- extract previous state from the issue body (between the HTML-comment markers) ---
prev_raw="$(gh issue view "$issue_num" --json body -q .body 2>/dev/null || printf '')"
prev="$(printf '%s' "$prev_raw" | sed -n "/${STATE_BEGIN}/,/${STATE_END}/p" | sed '1d;$d' | jq -c . 2>/dev/null || printf '')"
if [ -z "$prev" ]; then
  log "Could not parse previous state from issue body — re-baselining in place."
  prev="$state"
fi

# compare ignoring checked_at (it changes every run)
state_cmp="$(printf '%s' "$state" | jq -c 'del(.checked_at)')"
prev_cmp="$(printf '%s' "$prev"  | jq -c 'del(.checked_at)')"
if [ "$prev_cmp" = "$state_cmp" ]; then
  log "No changes detected. Staying quiet."
  exit 0
fi

# --- compute a human-readable diff ---
diff_body=""
blocker_hit="false"
while IFS= read -r key; do
  [ "$key" = "checked_at" ] && continue
  old="$(printf '%s' "$prev"  | jq -rc --arg k "$key" '.[$k]')"
  new="$(printf '%s' "$state" | jq -rc --arg k "$key" '.[$k]')"
  [ "$old" = "$new" ] && continue
  lbl="$(label_for "$key")"
  case "$lbl" in ⚡*) blocker_hit="true";; esac
  if [ "$key" = "qwen3_tts_models" ] || [ "$key" = "mlx_community_models" ]; then
    added="$(jq -nr --argjson a "$old" --argjson b "$new" '($b - $a) | join(", ")' 2>/dev/null || printf '?')"
    removed="$(jq -nr --argjson a "$old" --argjson b "$new" '($a - $b) | join(", ")' 2>/dev/null || printf '?')"
    diff_body+="$(printf -- '- **%s** changed\n  - added: `%s`\n  - removed: `%s`\n' "$lbl" "$added" "$removed")"
  else
    diff_body+="$(printf -- '- **%s**: `%s` → `%s`\n' "$lbl" "$old" "$new")"
  fi
done < <(printf '%s' "$state" | jq -r 'keys[]')

# checked_at is the only diff → just refresh the body, stay quiet
if [ -z "$diff_body" ]; then
  log "Only checked_at changed — refreshing body, no comment."
  gh issue edit "$issue_num" --body "$(render_body "$state")" >/dev/null || true
  exit 0
fi

{
  printf '## Upstream movement detected\n\n'
  printf '%s' "$diff_body"
  printf '\n'
  if [ "$blocker_hit" = "true" ]; then
    printf -- '---\n⚠️ **A ⚡ blocker-clearing signal changed** — worth re-running the deep-research sweep to reassess R-28 / FUTURE-* / HIGH-*.\n'
  fi
  printf '\n_State updated._\n'
} | gh issue comment "$issue_num" --body-file - >/dev/null || die "failed to post change comment"

gh issue edit "$issue_num" --body "$(render_body "$state")" >/dev/null || die "failed to update issue body"
log "Posted change comment to #$issue_num and updated stored state."
