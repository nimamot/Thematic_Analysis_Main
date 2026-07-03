#!/bin/bash
# Load pipeline_config.env + .env.supabase. Used by run.sh (host) and launch_*.sh (container).
#
# Priority: shell overrides (sbatch) > pipeline_config.env > .env.supabase (secrets only) > defaults

# Keys that may be overridden by the shell before load_pipeline_env runs.
_PIPELINE_CONFIG_KEYS=(
    RESEARCH_QUESTION
    GT_DATA_CSV
    PIPELINE_SLUG
    GT_LAUNCHER
    GT_CODEBOOK_REVIEW
    GT_CODEBOOK_REVIEW_MODE
    GT_CODEBOOK_REVIEW_BACKEND
    GT_CODEBOOK_REVIEW_TIMEOUT_SEC
    GT_CODEBOOK_REVIEW_POLL_INTERVAL_SEC
    GT_VIEWER_EXPORT
    GT_VIEWER_AUTO_LAUNCH
    UPLOAD_TO_SUPABASE
    GT_QUALITATIVE_ENRICHMENT
    GT_ENRICH_WORKERS
    GT_OPEN_CODING_WORKERS
    GT_HIGH_LEVEL_STRATEGY
    GT_HL_N_SAMPLES
    GT_HL_SAMPLE_SIZE
    GT_HIGH_LEVEL_WORKERS
    GT_HIERARCHY_WORKERS
    GT_HIERARCHY_REFINE
    GT_USE_SKILLS
    GT_SGLANG_CONTEXT_LENGTH
    GT_OPENAI_BASE
    GT_LLM_MODEL
    GT_EMBED_BACKEND
    GT_EMBED_MODEL
)

_PIPELINE_SAVED_OVERRIDES=()

_save_shell_overrides() {
    _PIPELINE_SAVED_OVERRIDES=()
    local key
    for key in "${_PIPELINE_CONFIG_KEYS[@]}"; do
        if [ -n "${!key+x}" ]; then
            _PIPELINE_SAVED_OVERRIDES+=("${key}=${!key}")
        fi
    done
}

_restore_shell_overrides() {
    local item key value
    for item in "${_PIPELINE_SAVED_OVERRIDES[@]}"; do
        key="${item%%=*}"
        value="${item#*=}"
        export "${key}=${value}"
    done
}

_finalize_pipeline_paths() {
    local scripts_dir="$1"
    export AGENTS_ROOT="$(cd "$scripts_dir/.." && pwd)"
    export REPO_ROOT="$(cd "$AGENTS_ROOT/.." && pwd)"

    if [ -z "${GT_DATA_CSV:-}" ]; then
        export GT_DATA_CSV="$REPO_ROOT/data/train.csv"
    elif [[ "$GT_DATA_CSV" != /* ]]; then
        export GT_DATA_CSV="$REPO_ROOT/$GT_DATA_CSV"
    fi

    export RESEARCH_QUESTION="${RESEARCH_QUESTION:-What thematic patterns emerge across these reviews?}"
    export PIPELINE_SLUG="${PIPELINE_SLUG:-default}"
    export GT_LAUNCHER="${GT_LAUNCHER:-sgl}"
}

_load_supabase_secrets() {
    local secrets_file="$1"
    [ -f "$secrets_file" ] || return 0

    local line key val
    while IFS= read -r line || [ -n "$line" ]; do
        case "$line" in
            \#*|"") continue ;;
        esac
        line="${line#export }"
        key="${line%%=*}"
        key="${key// /}"
        val="${line#*=}"
        val="${val#\"}"
        val="${val%\"}"
        val="${val#\'}"
        val="${val%\'}"
        case "$key" in
            SUPABASE_URL|SUPABASE_SERVICE_ROLE_KEY)
                export "${key}=${val}"
                ;;
        esac
    done <"$secrets_file"
}

load_pipeline_env() {
    local scripts_dir="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
    local config="$scripts_dir/pipeline_config.env"
    local secrets="$scripts_dir/.env.supabase"

    _save_shell_overrides

    if [ -f "$config" ]; then
        set -a
        # shellcheck disable=SC1090
        source "$config"
        set +a
    else
        echo "Warning: $config not found — using code defaults only." >&2
    fi

    _load_supabase_secrets "$secrets"
    if [ ! -f "$secrets" ]; then
        echo "Note: $secrets not found — Supabase upload/review-backend=supabase unavailable." >&2
    fi

    _restore_shell_overrides
    _finalize_pipeline_paths "$scripts_dir"
}

print_pipeline_config() {
    local where="${1:-pipeline}"
    if [ -z "${REPO_ROOT:-}" ]; then
        echo "print_pipeline_config: REPO_ROOT unset — call load_pipeline_env first" >&2
        return 1
    fi
    export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
    export PIPELINE_CONFIG_WHERE="$where"
    python - <<'PY' || true
import os
import sys
from pathlib import Path

repo = Path(os.environ["REPO_ROOT"])
sys.path.insert(0, str(repo))
from agents.core.pipeline_config import format_config_report, write_config_snapshot, effective_config

where = os.environ.get("PIPELINE_CONFIG_WHERE", "pipeline")
cfg = effective_config(where=where)
snapshot = write_config_snapshot(where=where)
print(format_config_report(cfg))
print(f"\nSaved: {snapshot}")
PY
}

# Backward-compatible alias
print_pipeline_env_flags() {
    print_pipeline_config "${1:-pipeline}"
}

require_supabase_credentials() {
    if [ -n "${SUPABASE_URL:-}" ] && [ -n "${SUPABASE_SERVICE_ROLE_KEY:-}" ]; then
        return 0
    fi
    echo "Error: Supabase credentials missing. Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in agents/scripts/.env.supabase" >&2
    exit 1
}

truthy_env() {
    case "${1:-0}" in
        1|true|yes|on|TRUE|YES|ON) return 0 ;;
        *) return 1 ;;
    esac
}

codebook_review_enabled() {
    truthy_env "${GT_CODEBOOK_REVIEW:-0}"
}

viewer_export_enabled() {
    truthy_env "${GT_VIEWER_EXPORT:-1}"
}

qualitative_enrichment_enabled() {
    truthy_env "${GT_QUALITATIVE_ENRICHMENT:-1}"
}

upload_to_supabase_enabled() {
    truthy_env "${UPLOAD_TO_SUPABASE:-0}"
}

codebook_review_uses_supabase() {
    case "${GT_CODEBOOK_REVIEW_BACKEND:-}" in
        supabase) return 0 ;;
        local) return 1 ;;
    esac
    if [ -n "${SUPABASE_URL:-}" ] && [ -n "${SUPABASE_SERVICE_ROLE_KEY:-}" ]; then
        return 0
    fi
    return 1
}
