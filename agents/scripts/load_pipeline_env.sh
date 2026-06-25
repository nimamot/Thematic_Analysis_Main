#!/bin/bash
# Source pipeline_config.env + .env.supabase. Used by run.sh (host) and launch_*.sh (container).

load_pipeline_env() {
    local scripts_dir="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
    local config="$scripts_dir/pipeline_config.env"
    local secrets="$scripts_dir/.env.supabase"

    if [ -f "$config" ]; then
        set -a
        # shellcheck disable=SC1090
        source "$config"
        set +a
    else
        echo "Warning: $config not found — GT_CODEBOOK_REVIEW and UPLOAD_TO_SUPABASE default off." >&2
    fi

    if [ -f "$secrets" ]; then
        set -a
        # shellcheck disable=SC1090
        source "$secrets"
        set +a
    else
        echo "Note: $secrets not found — set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY for review/upload." >&2
    fi

    if [ -n "${SUPABASE_SERVICE_ROLE_KEY:-}" ] && [ -n "${SUPABASE_URL:-}" ]; then
        export UPLOAD_TO_SUPABASE="${UPLOAD_TO_SUPABASE:-1}"
        export GT_CODEBOOK_REVIEW="${GT_CODEBOOK_REVIEW:-1}"
    fi
}

print_pipeline_env_flags() {
    local where="${1:-pipeline}"
    local supabase_ok=no
    if [ -n "${SUPABASE_URL:-}" ] && [ -n "${SUPABASE_SERVICE_ROLE_KEY:-}" ]; then
        supabase_ok=yes
    fi
    echo "${where} flags: GT_CODEBOOK_REVIEW=${GT_CODEBOOK_REVIEW:-0} GT_CODEBOOK_REVIEW_BACKEND=${GT_CODEBOOK_REVIEW_BACKEND:-auto} GT_VIEWER_EXPORT=${GT_VIEWER_EXPORT:-1} GT_QUALITATIVE_ENRICHMENT=${GT_QUALITATIVE_ENRICHMENT:-1} UPLOAD_TO_SUPABASE=${UPLOAD_TO_SUPABASE:-0} PIPELINE_SLUG=${PIPELINE_SLUG:-default} SUPABASE_CREDENTIALS=${supabase_ok}"
}

require_supabase_credentials() {
    if [ -n "${SUPABASE_URL:-}" ] && [ -n "${SUPABASE_SERVICE_ROLE_KEY:-}" ]; then
        return 0
    fi
    echo "Error: Supabase credentials missing. Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in agents/scripts/.env.supabase" >&2
    exit 1
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
