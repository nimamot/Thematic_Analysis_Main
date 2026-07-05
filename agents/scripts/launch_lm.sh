#!/bin/bash
# LM Studio via Apptainer: Qwen LLM + Qwen3-Embedding on one server (no SGLang kill/restart).
# Requires: lmstudio-llmster-preview.sif, pytorch-langgraph-sgl.sif, GGUF weights (download_lm_models.sh).
# Use via: GT_LAUNCHER=lm sbatch run.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
AGENTS_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$AGENTS_ROOT/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"
export PYTHONUNBUFFERED=1

# shellcheck disable=SC1091
source "$SCRIPT_DIR/load_pipeline_env.sh"
load_pipeline_env "$SCRIPT_DIR"
print_pipeline_config "Container"

export RESEARCH_QUESTION
export GT_DATA_CSV

SERVER_LOG="${LM_SERVER_LOG:-$AGENTS_ROOT/lm_server.log}"
PORT="${LM_PORT:-1234}"

LM_SIF="${LM_SIF_PATH:-$AGENTS_ROOT/lmstudio-llmster-preview.sif}"
PYTORCH_SIF="${PYTORCH_SIF_PATH:-${SIF_PATH:-$AGENTS_ROOT/pytorch-langgraph-sgl.sif}}"
LM_HOME="${LM_APPTAINER_HOME:-/scratch/nimamot/lmstudio_apptainer_home}"
PIPELINE_HOME="${APPTAINER_HOME:-/scratch/nimamot/vllm_env_home}"

LM_WEIGHTS_DIR="${LM_WEIGHTS_DIR:-$AGENTS_ROOT/weights/lmstudio}"
LM_LLM_REPO="${LM_LLM_REPO:-lmstudio-community/Qwen3-30B-A3B-Instruct-2507-GGUF}"
LM_LLM_QUANT="${LM_LLM_QUANT:-Q4_K_M}"
LM_EMBED_REPO="${LM_EMBED_REPO:-Qwen/Qwen3-Embedding-0.6B-GGUF}"
LM_EMBED_QUANT="${LM_EMBED_QUANT:-Q8_0}"
LM_LLM_GGUF="${LM_LLM_GGUF:-$LM_WEIGHTS_DIR/llm/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf}"
LM_EMBED_GGUF="${LM_EMBED_GGUF:-$LM_WEIGHTS_DIR/embed/Qwen3-Embedding-0.6B-Q8_0.gguf}"
# lms load after symlink-import uses repo/file.gguf (not repo@quant from lms get).
LM_LLM_LOAD="${LM_LLM_LOAD:-${LM_LLM_REPO}/$(basename "$LM_LLM_GGUF")}"
LM_EMBED_LOAD="${LM_EMBED_LOAD:-${LM_EMBED_REPO}/$(basename "$LM_EMBED_GGUF")}"
LLM_CONTEXT_LENGTH="${LLM_CONTEXT_LENGTH:-8192}"
EMBED_GPU_FRACTION="${EMBED_GPU_FRACTION:-max}"
MAX_RETRIES=2400

mkdir -p "$LM_HOME" "$PIPELINE_HOME"

module load apptainer

APPTAINER_BINDS=(-B /project -B /scratch -B "$REPO_ROOT:$REPO_ROOT")

ensure_lm_sif() {
    if [ -f "$LM_SIF" ]; then
        return 0
    fi
    echo "LM Studio SIF not found at $LM_SIF" >&2
    echo "Run: bash $SCRIPT_DIR/pull_lmstudio_sif.sh" >&2
    exit 1
}

ensure_pytorch_sif() {
    if [ -f "$PYTORCH_SIF" ]; then
        return 0
    fi
    echo "PyTorch pipeline SIF not found at $PYTORCH_SIF" >&2
    exit 1
}

lm_exec() {
    apptainer exec --home "$LM_HOME" -W "$LM_HOME" \
        "${APPTAINER_BINDS[@]}" --pwd /app "$LM_SIF" lms "$@"
}

pipeline_exec() {
    apptainer exec -C --nv --home "$PIPELINE_HOME" -W "$PIPELINE_HOME" \
        "${APPTAINER_BINDS[@]}" \
        --env PYTHONPATH="$REPO_ROOT" \
        --env PYTHONUNBUFFERED=1 \
        --env GT_OPENAI_BASE="${GT_OPENAI_BASE:-}" \
        --env GT_LLM_MODEL="${GT_LLM_MODEL:-}" \
        --env GT_EMBED_BACKEND="${GT_EMBED_BACKEND:-}" \
        --env GT_EMBED_MODEL="${GT_EMBED_MODEL:-}" \
        --env RESEARCH_QUESTION="$RESEARCH_QUESTION" \
        --env HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-}" \
        --env TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-}" \
        --env HF_HOME="${HF_HOME:-}" \
        --env UPLOAD_TO_SUPABASE="${UPLOAD_TO_SUPABASE:-0}" \
        --env PIPELINE_SLUG="${PIPELINE_SLUG:-default}" \
        --env SUPABASE_URL="${SUPABASE_URL:-}" \
        --env SUPABASE_SERVICE_ROLE_KEY="${SUPABASE_SERVICE_ROLE_KEY:-}" \
        "$PYTORCH_SIF" "$@"
}

ensure_pipeline_python_deps() {
    if pipeline_exec python -c "import langchain_core.messages; import langchain_openai; import openai" 2>/dev/null; then
        return 0
    fi
    echo "Installing pipeline Python deps from $AGENTS_ROOT/requirements-pipeline.txt (Apptainer user site)..."
    if ! pipeline_exec python -m pip install --user -r "$AGENTS_ROOT/requirements-pipeline.txt"; then
        echo "Error: pip could not install LangChain deps inside $PYTORCH_SIF." >&2
        exit 1
    fi
}

lm_cuda_preflight() {
    echo "=== LM Studio CUDA preflight (via $PYTORCH_SIF) ==="
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
    pipeline_exec python -c "
import sys
try:
    import torch
except Exception as e:
    print('Error: import torch failed:', e, file=sys.stderr)
    sys.exit(1)
ok = torch.cuda.is_available()
n = torch.cuda.device_count()
print('torch.cuda.is_available():', ok)
print('torch.cuda.device_count():', n)
if not ok or n < 1:
    print('Error: no CUDA device visible to PyTorch.', file=sys.stderr)
    sys.exit(1)
print('device 0:', torch.cuda.get_device_name(0))
" || exit 1
    echo "=== end preflight ==="
}

_import_marker_path() {
    local repo="$1"
    local gguf="$2"
    echo "$LM_HOME/.lmstudio/models/$repo/$(basename "$gguf")"
}

ensure_gguf_imported() {
    local gguf="$1"
    local repo="$2"
    local label="$3"

    if [ ! -f "$gguf" ]; then
        echo "Error: $label GGUF missing at $gguf" >&2
        echo "Run: bash $SCRIPT_DIR/download_lm_models.sh --hf-only" >&2
        exit 1
    fi

    local marker
    marker="$(_import_marker_path "$repo" "$gguf")"
    if [ -e "$marker" ]; then
        echo "$label already imported: $marker"
        return 0
    fi

    echo "Importing $label (symlink): $gguf"
    lm_exec import -l -y "$gguf"
}

_models_endpoint_ready() {
    local port="$1"
    curl -sS --connect-timeout 2 --max-time 10 "http://127.0.0.1:${port}/v1/models" 2>/dev/null | grep -qE '"object"[[:space:]]*:[[:space:]]*"list"'
}

wait_for_openai_ready() {
    local port="$1"
    local log_file="$2"
    local pid="$3"
    local phase_label="${4:-LM Studio}"
    local counter=0
    echo "Waiting for $phase_label to become ready on port $port..."
    while true; do
        if _models_endpoint_ready "$port"; then
            echo "$phase_label is READY!"
            return 0
        fi
        if [ $counter -ge $MAX_RETRIES ]; then
            echo "Timeout waiting for $phase_label. Last 50 lines of $log_file:"
            tail -n 50 "$log_file" 2>/dev/null || true
            stop_lmstudio_server "$pid"
            return 1
        fi
        if [ -n "$pid" ] && ! ps -p "$pid" > /dev/null 2>&1; then
            echo "$phase_label process died unexpectedly. Check $log_file."
            tail -n 50 "$log_file" 2>/dev/null || true
            return 1
        fi
        echo "Waiting for $phase_label... ($counter/$MAX_RETRIES)"
        sleep 5
        ((counter++))
    done
}

stop_lmstudio_server() {
    local pid="${1:-}"
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        echo "Stopping LM Studio Apptainer server (PID $pid)..."
        kill -TERM "$pid" 2>/dev/null || true
        local i=0
        while [ "$i" -lt 30 ] && kill -0 "$pid" 2>/dev/null; do
            sleep 1
            i=$((i + 1))
        done
        kill -KILL "$pid" 2>/dev/null || true
    fi
    if [ -f "$LM_SIF" ]; then
        lm_exec unload --all 2>/dev/null || true
        lm_exec server stop 2>/dev/null || true
    fi
    echo "nvidia-smi (after LM Studio stop):"
    nvidia-smi 2>/dev/null || true
}

cleanup_on_exit() {
    stop_lmstudio_server "$SERVER_PID"
}
trap cleanup_on_exit EXIT

ensure_lm_sif
ensure_pytorch_sif
ensure_pipeline_python_deps
lm_cuda_preflight

ensure_gguf_imported "$LM_LLM_GGUF" "$LM_LLM_REPO" "LLM"
ensure_gguf_imported "$LM_EMBED_GGUF" "$LM_EMBED_REPO" "Embedding"

echo "Starting LM Studio Apptainer server ($LM_SIF) on port $PORT..."
export APPTAINERENV_EXTS=cuda
apptainer run --nv --home "$LM_HOME" -W "$LM_HOME" \
    "${APPTAINER_BINDS[@]}" --pwd /app "$LM_SIF" >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "LM Studio Apptainer PID: $SERVER_PID (log: $SERVER_LOG)"

if ! wait_for_openai_ready "$PORT" "$SERVER_LOG" "$SERVER_PID" "LM Studio"; then
    exit 1
fi

echo "Loading chat LLM: $LM_LLM_LOAD"
if ! lm_exec load "$LM_LLM_LOAD" --identifier llm --gpu max --context-length "$LLM_CONTEXT_LENGTH" --yes; then
    echo "Error: failed to load LLM." >&2
    exit 1
fi

echo "Loading embedding model: $LM_EMBED_LOAD"
if ! lm_exec load "$LM_EMBED_LOAD" --identifier embed --gpu "$EMBED_GPU_FRACTION" --yes; then
    echo "Error: failed to load embedding model." >&2
    exit 1
fi

if ! _models_endpoint_ready "$PORT"; then
    echo "Error: server not responding after model load." >&2
    exit 1
fi

export GT_OPENAI_BASE="http://127.0.0.1:${PORT}/v1"
export GT_LLM_MODEL=llm
export GT_EMBED_BACKEND=lmstudio
export GT_EMBED_MODEL=embed

echo "Inference env: GT_OPENAI_BASE=$GT_OPENAI_BASE GT_EMBED_BACKEND=$GT_EMBED_BACKEND"

run_stage() {
    local label="$1"
    shift
    echo "=== $label ==="
    pipeline_exec "$@"
    local rc=$?
    echo "$label finished with exit code $rc"
    if [ $rc -ne 0 ]; then
        exit $rc
    fi
}

run_stage "Open coding" \
    python -m agents.cli --open-coding-only --research-question "$RESEARCH_QUESTION" --data "$GT_DATA_CSV"

run_stage "Axial coding" \
    python -m agents.cli --axial-only --research-question "$RESEARCH_QUESTION"

run_stage "High-level code generation" \
    python -m agents.cli --high-level-only --research-question "$RESEARCH_QUESTION"

if codebook_review_enabled; then
    echo "Codebook review gate enabled (GT_CODEBOOK_REVIEW=${GT_CODEBOOK_REVIEW})..."
    export PIPELINE_SLUG="${PIPELINE_SLUG:-default}"
    export RESEARCH_QUESTION
    if codebook_review_uses_supabase; then
        require_supabase_credentials
    else
        echo "Codebook review backend: local (viewer-data/)."
    fi
    if [ "${GT_CODEBOOK_REVIEW_MODE:-manual}" != "interrupt" ]; then
        pipeline_exec python "$AGENTS_ROOT/scripts/upload_codebook_for_review.py" || exit 1
    fi
    pipeline_exec python -m agents.cli --wait-codebook-review --research-question "$RESEARCH_QUESTION" || exit 1
else
    echo "Codebook review gate skipped (GT_CODEBOOK_REVIEW=${GT_CODEBOOK_REVIEW:-0}). Set GT_CODEBOOK_REVIEW=1 in agents/scripts/pipeline_config.env."
fi

if codebook_review_enabled && [ "${GT_CODEBOOK_REVIEW_MODE:-manual}" = "interrupt" ]; then
    run_stage "Refine cluster assignments (resume)" \
        python -m agents.cli --resume-codebook-review --research-question "$RESEARCH_QUESTION"
else
    run_stage "Refine cluster assignments" \
        python -m agents.cli --refine-only --research-question "$RESEARCH_QUESTION"
fi

if qualitative_enrichment_enabled; then
    run_stage "Cluster qualitative enrichment" \
        python -m agents.cli --enrich-codebook-only --research-question "$RESEARCH_QUESTION"
else
    echo "Skipping cluster qualitative enrichment (GT_QUALITATIVE_ENRICHMENT=0)."
fi

run_stage "Hierarchy construction" \
    python -m agents.cli --hierarchy-only --research-question "$RESEARCH_QUESTION"

run_stage "Meta-theme grouping" \
    python -m agents.cli --meta-themes-only --research-question "$RESEARCH_QUESTION"

if qualitative_enrichment_enabled; then
    run_stage "Dimension qualitative enrichment" \
        python -m agents.cli --enrich-dimensions-only --research-question "$RESEARCH_QUESTION"
fi

run_stage "Tree assembly" \
    python -m agents.cli --global-graph-only --research-question "$RESEARCH_QUESTION"

run_stage "Research report" \
    python -m agents.cli --report-only --research-question "$RESEARCH_QUESTION"

run_stage "Co-occurrence network" \
    python -m agents.cli --cooccurrence-only --research-question "$RESEARCH_QUESTION"

if upload_to_supabase_enabled; then
    echo "Uploading pipeline artifacts to Supabase (UPLOAD_TO_SUPABASE=1)..."
    export PIPELINE_SLUG="${PIPELINE_SLUG:-default}"
    if ! pipeline_exec python "$AGENTS_ROOT/scripts/upload_pipeline_to_supabase.py"; then
        echo "Error: Supabase upload failed."
        exit 1
    fi
else
    echo "Supabase upload skipped (UPLOAD_TO_SUPABASE=${UPLOAD_TO_SUPABASE:-0}). Set UPLOAD_TO_SUPABASE=1 in agents/scripts/pipeline_config.env to enable."
fi

if viewer_export_enabled; then
    echo "Exporting pipeline artifacts to viewer-data/ (GT_VIEWER_EXPORT=1)..."
    export PIPELINE_SLUG="${PIPELINE_SLUG:-default}"
    if ! pipeline_exec python "$AGENTS_ROOT/scripts/export_viewer_data.py" --mode project; then
        echo "Error: viewer-data export failed."
        exit 1
    fi
else
    echo "Viewer export skipped (GT_VIEWER_EXPORT=0)."
fi

trap - EXIT
stop_lmstudio_server "$SERVER_PID"
echo "LM Studio pipeline completed successfully."
exit 0
