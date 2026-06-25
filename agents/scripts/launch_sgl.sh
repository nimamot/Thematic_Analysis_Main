#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
AGENTS_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$AGENTS_ROOT/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"
# When stdout/stderr are redirected to server.log, Python block-buffers; logs look "empty" for a long time while the model loads.
export PYTHONUNBUFFERED=1

# Apptainer -C strips host env; load toggles + Supabase credentials from bind-mounted repo files.
# shellcheck disable=SC1091
source "$SCRIPT_DIR/load_pipeline_env.sh"
load_pipeline_env "$SCRIPT_DIR"
print_pipeline_env_flags "Container"

# LangChain stack: bake into the .sif when you rebuild the image, or install once into the Apptainer
ensure_pipeline_python_deps() {
    if python -c "import langchain_core.messages; import langchain_openai" 2>/dev/null; then
        return 0
    fi
    echo "Installing pipeline Python deps from $AGENTS_ROOT/requirements-pipeline.txt (user site)..."
    if ! python -m pip install --user -r "$AGENTS_ROOT/requirements-pipeline.txt"; then
        echo "Error: pip could not install LangChain deps. Add them to the container image or fix pip/network." >&2
        exit 1
    fi
}
ensure_pipeline_python_deps


# Override examples:
#   export GT_DATA_CSV="$REPO_ROOT/data/reddit_comment_text_1000.csv"
# GT_DATA_CSV="${GT_DATA_CSV:-$REPO_ROOT/data/school_burnout_text_review.csv}"
# GT_DATA_CSV="$REPO_ROOT/data/gameplayreview_text_english_3k.csv"
GT_DATA_CSV="$REPO_ROOT/data/train.csv"
# GT_DATA_CSV="$REPO_ROOT/data/school_burnout_text_review.csv"


MODEL_PATH="$AGENTS_ROOT/weights/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit"
SERVER_LOG="$AGENTS_ROOT/server.log"
PORT=8000
# Research question — keep it broad so open coding stays inductive (avoid naming expected themes).
# Override: RESEARCH_QUESTION="..." sbatch run.sh
RESEARCH_QUESTION="What thematic patterns emerge across these reviews?"
# RESEARCH_QUESTION="What do players dislike about games and software in these reviews?"

export RESEARCH_QUESTION

# LLM axial clustering: read USE_LLM_CLUSTERING from agents/core/llm_clustering.py.
USE_LLM_CLUSTERING="$(python -c "from agents.core.llm_clustering import USE_LLM_CLUSTERING; print(1 if USE_LLM_CLUSTERING else 0)")"

_use_llm_clustering() {
    [ "${USE_LLM_CLUSTERING}" = "1" ]
}

# Total context window for SGLang. LLM clustering sends large label batches and needs
# room for input + JSON completion (HICode uses max_completion_tokens=8192).
if _use_llm_clustering; then
    SGLANG_CONTEXT_LENGTH="${GT_SGLANG_CONTEXT_LENGTH:-16384}"
else
    SGLANG_CONTEXT_LENGTH="${GT_SGLANG_CONTEXT_LENGTH:-8192}"
fi
export GT_SGLANG_CONTEXT_LENGTH="$SGLANG_CONTEXT_LENGTH"
echo "SGLang context-length=${SGLANG_CONTEXT_LENGTH} (USE_LLM_CLUSTERING=${USE_LLM_CLUSTERING})"

ensure_embed_weights() {
    EMBED_WEIGHTS_DIR="$AGENTS_ROOT/weights/Qwen3-Embedding-0.6B"
    EMBED_HF_ID="Qwen/Qwen3-Embedding-0.6B"
    if [ -d "$EMBED_WEIGHTS_DIR" ]; then
        export GT_EMBED_MODEL="$EMBED_WEIGHTS_DIR"
        return 0
    fi
    echo "Embed model not in weights/; downloading once to $EMBED_WEIGHTS_DIR ..."
    if command -v huggingface-cli &>/dev/null; then
        (cd "$AGENTS_ROOT" && huggingface-cli download "$EMBED_HF_ID" --local-dir "weights/Qwen3-Embedding-0.6B")
    else
        (cd "$AGENTS_ROOT" && python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='$EMBED_HF_ID', local_dir='weights/Qwen3-Embedding-0.6B', local_dir_use_symlinks=False)
")
    fi
    export GT_EMBED_MODEL="$EMBED_WEIGHTS_DIR"
}

# Stop SGLang reliably so GPU VRAM is freed before axial (embedding) and other steps.
stop_sglang_server() {
    local pid="${1:-}"
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        echo "Stopping SGLang main process (PID $pid)..."
        kill -TERM "$pid" 2>/dev/null || true
        local i=0
        while [ "$i" -lt 45 ] && kill -0 "$pid" 2>/dev/null; do
            sleep 1
            i=$((i + 1))
        done
        if kill -0 "$pid" 2>/dev/null; then
            echo "SGLang PID $pid still running; sending SIGKILL..."
            kill -KILL "$pid" 2>/dev/null || true
            sleep 2
        fi
    fi
    # Catch worker/orphan processes that still hold VRAM (same user only).
    local u="${USER:-$(whoami)}"
    if pkill -TERM -u "$u" -f "sglang.launch_server" 2>/dev/null; then
        sleep 3
    fi
    pkill -KILL -u "$u" -f "sglang.launch_server" 2>/dev/null || true
    sleep 2
    echo "nvidia-smi (after SGLang stop):"
    nvidia-smi 2>/dev/null || true
}

# Fail fast if this Python cannot see a GPU (common when MIG / Slurm GPU binding and container disagree).
sglang_cuda_preflight() {
    local model_path="$1"
    echo "=== SGLang CUDA preflight ==="
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
    echo "MODEL_PATH=$model_path"
    if [ ! -d "$model_path" ]; then
        echo "Error: model directory missing (wrong path or weights not on this filesystem)."
        exit 1
    fi
    python - <<'PY' || exit 1
import os, sys
try:
    import torch
except Exception as e:
    print("Error: import torch failed:", e, file=sys.stderr)
    sys.exit(1)
ok = torch.cuda.is_available()
n = torch.cuda.device_count()
print("torch.cuda.is_available():", ok)
print("torch.cuda.device_count():", n)
if not ok or n < 1:
    print(
        "Error: no CUDA device visible to PyTorch inside this environment.\n"
        "  Typical causes: MIG slice vs full GPU mismatch, wrong Slurm --gres, or Apptainer --nv not passing the GPU.\n"
        "  Ask your site how to run GPU jobs on this partition (CUDA_VISIBLE_DEVICES / MIG instance ID).",
        file=sys.stderr,
    )
    sys.exit(1)
print("device 0:", torch.cuda.get_device_name(0))
PY
    echo "=== end preflight ==="
}

_models_endpoint_ready() {
    local port="$1"
    curl -sS --connect-timeout 2 --max-time 10 "http://127.0.0.1:${port}/v1/models" 2>/dev/null | grep -qE '"object"[[:space:]]*:[[:space:]]*"list"'
}

# Wait until OpenAI-compatible /v1/models responds (shared by Qwen and Mistral phases).
# Args: port, log_file, pid, label
wait_for_openai_ready() {
    local port="$1"
    local log_file="$2"
    local pid="$3"
    local phase_label="${4:-SGLang}"
    local counter=0
    local diagf=""
    echo "Waiting for $phase_label to become ready on port $port..."
    while true; do
        if _models_endpoint_ready "$port"; then
            echo "$phase_label is READY!"
            return 0
        fi
        if [ $counter -ge $MAX_RETRIES ]; then
            echo "Timeout waiting for $phase_label. Last 50 lines of $log_file:"
            tail -n 50 "$log_file"
            stop_sglang_server "$pid"
            return 1
        fi
        if ! ps -p "$pid" > /dev/null 2>&1; then
            echo "$phase_label process died unexpectedly. Check $log_file."
            tail -n 50 "$log_file"
            return 1
        fi
        # Every ~5 min: show HTTP status, response snippet, log size (helps debug hang vs wrong JSON vs slow load).
        if [ "$counter" -gt 0 ] && [ $((counter % 60)) -eq 0 ]; then
            diagf=$(mktemp /tmp/sgl_diag.XXXXXX 2>/dev/null || echo "/tmp/sgl_diag.$$")
            # Do not use "|| echo 000" — curl already prints 000 on failure and would concatenate to 000000.
            http_code=$(curl -sS -o "$diagf" --connect-timeout 2 --max-time 15 -w "%{http_code}" "http://127.0.0.1:${port}/v1/models" 2>/dev/null) || true
            http_code="${http_code:-000}"
            log_bytes=$(wc -c <"$log_file" 2>/dev/null || echo 0)
            echo "  [diag @${counter}] GET /v1/models http=${http_code} server.log_bytes=${log_bytes}"
            if [ -s "$diagf" ]; then
                echo -n "  [diag] body head: "
                head -c 240 "$diagf" | tr '\n' ' '
                echo
            fi
            rm -f "$diagf" 2>/dev/null || true
        fi
        echo "Waiting for $phase_label... ($counter/$MAX_RETRIES)"
        sleep 5
        ((counter++))
    done
}

# --- 2. Launch Server in Background ---
sglang_cuda_preflight "$MODEL_PATH"

echo "Starting SGLang server..."
# Run python directly (no subshell) so $! is the real server PID.
python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --port $PORT \
  --host 0.0.0.0 \
  --served-model-name llm \
  --mem-fraction-static 0.90 \
  --context-length "$SGLANG_CONTEXT_LENGTH" \
  >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

MAX_RETRIES=2400
# --- 3. The 'Wait-For-Ready' Loop ---
if ! wait_for_openai_ready "$PORT" "$SERVER_LOG" "$SERVER_PID" "SGLang (Qwen)"; then
    exit 1
fi

# --- 4. Open coding ---
echo "Starting Open Coding (agents.cli --open-coding-only)..."
echo "GT_DATA_CSV=$GT_DATA_CSV"
python -m agents.cli --open-coding-only --research-question "$RESEARCH_QUESTION" --data "$GT_DATA_CSV"
OPEN_EXIT=$?
if [ $OPEN_EXIT -ne 0 ]; then
    stop_sglang_server "$SERVER_PID"
    exit $OPEN_EXIT
fi

if _use_llm_clustering; then
    echo "Open coding done. Keeping SGLang up for LLM axial clustering (USE_LLM_CLUSTERING in llm_clustering.py)."
else
    echo "Open coding finished. Stopping SGLang so axial can use GPU for embedding model..."
    stop_sglang_server "$SERVER_PID"
    # --- 5. Embed weights (embedding axial path only) ---
    ensure_embed_weights
fi

# --- 6. Axial step ---
export HF_HUB_OFFLINE=1
echo "Starting Axial step (agents.cli --axial-only)..."
python -m agents.cli --axial-only --research-question "$RESEARCH_QUESTION"
AXIAL_EXIT=$?
if [ $AXIAL_EXIT -ne 0 ]; then
    exit $AXIAL_EXIT
fi

# --- 7. SGLang for high-level (restart only if we stopped it after open coding) ---
if _use_llm_clustering; then
    echo "SGLang still running; proceeding to high-level / refine."
else
    sglang_cuda_preflight "$MODEL_PATH"
    echo "Restarting SGLang for high-level code generation..."
    python -m sglang.launch_server \
      --model-path "$MODEL_PATH" \
      --port $PORT \
      --host 0.0.0.0 \
      --served-model-name llm \
      --mem-fraction-static 0.90 \
      --context-length "$SGLANG_CONTEXT_LENGTH" \
      >"$SERVER_LOG" 2>&1 &
    SERVER_PID=$!
    if ! wait_for_openai_ready "$PORT" "$SERVER_LOG" "$SERVER_PID" "SGLang (Qwen, phase 2)"; then
        exit 1
    fi
fi

# Refine uses embedding model on CPU for similar cluster labels
if _use_llm_clustering; then
    ensure_embed_weights
fi

# --- 8. High-level code generation ---
echo "Starting high-level code generation (agents.cli --high-level-only)..."
python -m agents.cli --high-level-only --research-question "$RESEARCH_QUESTION"
HL_EXIT=$?
echo "High-level step finished with exit code $HL_EXIT."
if [ $HL_EXIT -ne 0 ]; then
    stop_sglang_server "$SERVER_PID"
    exit $HL_EXIT
fi

# Release GPU while waiting for human codebook review (optional gate).
stop_sglang_server "$SERVER_PID"
SERVER_PID=""

if [ "${GT_CODEBOOK_REVIEW:-0}" = "1" ]; then
    echo "Codebook review gate enabled (GT_CODEBOOK_REVIEW=1)..."
    export PIPELINE_SLUG="${PIPELINE_SLUG:-default}"
    export RESEARCH_QUESTION
    if codebook_review_uses_supabase; then
        require_supabase_credentials
    else
        echo "Codebook review backend: local (viewer-data/)."
    fi
    if [ "${GT_CODEBOOK_REVIEW_MODE:-manual}" != "interrupt" ]; then
        PYTHONPATH="$REPO_ROOT" python "$AGENTS_ROOT/scripts/upload_codebook_for_review.py" || exit 1
    fi
    PYTHONPATH="$REPO_ROOT" python -m agents.cli --wait-codebook-review --research-question "$RESEARCH_QUESTION" || exit 1
else
    echo "Codebook review gate skipped (GT_CODEBOOK_REVIEW=${GT_CODEBOOK_REVIEW:-0}). Set GT_CODEBOOK_REVIEW=1 for human review."
fi

# --- 9. Refine cluster assignments (restart SGLang) ---
sglang_cuda_preflight "$MODEL_PATH"
echo "Restarting SGLang for refine cluster assignments..."
python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --port $PORT \
  --host 0.0.0.0 \
  --served-model-name llm \
  --mem-fraction-static 0.90 \
  --context-length 8000 \
  >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!
if ! wait_for_openai_ready "$PORT" "$SERVER_LOG" "$SERVER_PID" "SGLang (Qwen, refine)"; then
    exit 1
fi

if [ "${GT_CODEBOOK_REVIEW:-0}" = "1" ] && [ "${GT_CODEBOOK_REVIEW_MODE:-manual}" = "interrupt" ]; then
    echo "Starting refine via LangGraph resume (GT_CODEBOOK_REVIEW_MODE=interrupt)..."
    python -m agents.cli --resume-codebook-review --research-question "$RESEARCH_QUESTION"
else
    echo "Starting refine cluster assignments (agents.cli --refine-only)..."
    python -m agents.cli --refine-only --research-question "$RESEARCH_QUESTION"
fi
REFINE_EXIT=$?
echo "Refine step finished with exit code $REFINE_EXIT."
if [ $REFINE_EXIT -ne 0 ]; then
    stop_sglang_server "$SERVER_PID"
    exit $REFINE_EXIT
fi

# --- 9b. Cluster qualitative enrichment (optional) ---
if [ "${GT_QUALITATIVE_ENRICHMENT:-1}" = "1" ]; then
    echo "Starting cluster qualitative enrichment (agents.cli --enrich-codebook-only)..."
    python -m agents.cli --enrich-codebook-only --research-question "$RESEARCH_QUESTION"
    ENRICH_CB_EXIT=$?
    echo "Cluster enrichment finished with exit code $ENRICH_CB_EXIT."
    if [ $ENRICH_CB_EXIT -ne 0 ]; then
        stop_sglang_server "$SERVER_PID"
        exit $ENRICH_CB_EXIT
    fi
else
    echo "Skipping cluster qualitative enrichment (GT_QUALITATIVE_ENRICHMENT=0)."
fi

# --- 10. Hierarchy construction ---
echo "Starting hierarchy construction (agents.cli --hierarchy-only)..."
python -m agents.cli --hierarchy-only --research-question "$RESEARCH_QUESTION"
HIER_EXIT=$?
echo "Hierarchy step finished with exit code $HIER_EXIT."
if [ $HIER_EXIT -ne 0 ]; then
    stop_sglang_server "$SERVER_PID"
    exit $HIER_EXIT
fi

# --- 11. Meta-theme grouping (LLM: writes gt_meta_themes.json) ---
echo "Starting meta-theme grouping (agents.cli --meta-themes-only)..."
python -m agents.cli --meta-themes-only --research-question "$RESEARCH_QUESTION"
META_EXIT=$?
echo "Meta-themes step finished with exit code $META_EXIT."
if [ $META_EXIT -ne 0 ]; then
    stop_sglang_server "$SERVER_PID"
    exit $META_EXIT
fi

# --- 11b. Meta-theme qualitative enrichment (optional) ---
if [ "${GT_QUALITATIVE_ENRICHMENT:-1}" = "1" ]; then
    echo "Starting dimension qualitative enrichment (agents.cli --enrich-dimensions-only)..."
    python -m agents.cli --enrich-dimensions-only --research-question "$RESEARCH_QUESTION"
    ENRICH_DIM_EXIT=$?
    echo "Dimension enrichment finished with exit code $ENRICH_DIM_EXIT."
    if [ $ENRICH_DIM_EXIT -ne 0 ]; then
        stop_sglang_server "$SERVER_PID"
        exit $ENRICH_DIM_EXIT
    fi
fi

# --- 12. Tree assembly (no LLM: merges meta-themes + hierarchy into gt_global_graph.json) ---
echo "Starting tree assembly (agents.cli --global-graph-only)..."
python -m agents.cli --global-graph-only --research-question "$RESEARCH_QUESTION"
GLOBAL_EXIT=$?
echo "Tree / global graph step finished with exit code $GLOBAL_EXIT."
if [ $GLOBAL_EXIT -ne 0 ]; then
    echo "Stopping SGLang after tree assembly failure..."
    stop_sglang_server "$SERVER_PID"
    exit $GLOBAL_EXIT
fi

echo "Stopping Qwen SGLang to free GPU for Mistral research-report server..."
stop_sglang_server "$SERVER_PID"

REPORT_MODEL_PATH="$AGENTS_ROOT/weights/Mistral-7B-Instruct-v0.3"
if [ ! -d "$REPORT_MODEL_PATH" ]; then
    echo "Error: Mistral weights not found at $REPORT_MODEL_PATH"
    echo "Run: bash \"$AGENTS_ROOT/scripts/download_mistral_7b_v03.sh\""
    exit 1
fi

REPORT_SERVER_LOG="$AGENTS_ROOT/report_server.log"
sglang_cuda_preflight "$REPORT_MODEL_PATH"

echo "Starting Mistral SGLang for research report..."
python -m sglang.launch_server \
  --model-path "$REPORT_MODEL_PATH" \
  --port $PORT \
  --host 0.0.0.0 \
  --served-model-name llm \
  --mem-fraction-static 0.85 \
  --context-length 8192 \
  >"$REPORT_SERVER_LOG" 2>&1 &
REPORT_PID=$!
echo "Mistral report server PID: $REPORT_PID"

if ! wait_for_openai_ready "$PORT" "$REPORT_SERVER_LOG" "$REPORT_PID" "SGLang (Mistral report)"; then
    exit 1
fi

echo "Starting research report (agents.cli --report-only)..."
python -m agents.cli --report-only --research-question "$RESEARCH_QUESTION"
REPORT_STEP_EXIT=$?
echo "Research report step finished with exit code $REPORT_STEP_EXIT. Stopping Mistral SGLang..."
stop_sglang_server "$REPORT_PID"
if [ "$REPORT_STEP_EXIT" -ne 0 ]; then
    exit "$REPORT_STEP_EXIT"
fi

echo "Starting co-occurrence network (agents.cli --cooccurrence-only)..."
python -m agents.cli --cooccurrence-only --research-question "$RESEARCH_QUESTION"
COOCCURRENCE_EXIT=$?
echo "Co-occurrence step finished with exit code $COOCCURRENCE_EXIT."
if [ "$COOCCURRENCE_EXIT" -ne 0 ]; then
    exit "$COOCCURRENCE_EXIT"
fi

if [ "${UPLOAD_TO_SUPABASE:-0}" = "1" ]; then
    require_supabase_credentials
    echo "Uploading pipeline artifacts to Supabase (UPLOAD_TO_SUPABASE=1)..."
    export PIPELINE_SLUG="${PIPELINE_SLUG:-default}"
    if ! PYTHONPATH="$REPO_ROOT" python "$AGENTS_ROOT/scripts/upload_pipeline_to_supabase.py"; then
        echo "Error: Supabase upload failed. Fix credentials/table or set UPLOAD_TO_SUPABASE=0 to skip."
        exit 1
    fi
else
    echo "Supabase upload skipped (UPLOAD_TO_SUPABASE is not 1). With sbatch run.sh, export UPLOAD_TO_SUPABASE=1 in .env.supabase or rely on run.sh defaulting it to 1 when SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are set."
fi

if [ "${GT_VIEWER_EXPORT:-1}" = "1" ]; then
    echo "Exporting pipeline artifacts to viewer-data/ (GT_VIEWER_EXPORT=1)..."
    export PIPELINE_SLUG="${PIPELINE_SLUG:-default}"
    if ! PYTHONPATH="$REPO_ROOT" python "$AGENTS_ROOT/scripts/export_viewer_data.py" --mode project; then
        echo "Error: viewer-data export failed."
        exit 1
    fi
else
    echo "Viewer export skipped (GT_VIEWER_EXPORT=0)."
fi
exit 0
