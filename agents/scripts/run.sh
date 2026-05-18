#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=60G
#SBATCH --time=0-12:00:00
#SBATCH --gpus=h100:1
#SBATCH --account=def-lingjzhu

#
# Fir (Alliance): full H100 via --gpus=h100:1; GPU RAC is rrg-lingjzhu.
# Binds /project and /scratch are standard on Fir.
#
# Paths use SLURM_SUBMIT_DIR under Slurm (see below); otherwise this script's directory.
# Override any variable before calling sbatch, e.g.:
#   SIF_PATH=/project/6102159/shared/pytorch-langgraph-sgl.sif sbatch run.sh

# Slurm runs a *copy* of this script from the job spool; dirname(BASH_SOURCE) is then
# something like .../slurmd/job12345/, so paths to the SIF and launch_sgl.sh break.
# SLURM_SUBMIT_DIR is the cwd where sbatch was invoked — use it when set.
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  _submit="$(cd "$SLURM_SUBMIT_DIR" && pwd)"
  if [ -f "$_submit/scripts/run.sh" ]; then
    AGENTS_SCRIPTS="$(cd "$_submit/scripts" && pwd)"
  elif [ -f "$_submit/run.sh" ]; then
    AGENTS_SCRIPTS="$(cd "$_submit" && pwd)"
  else
    AGENTS_SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  fi
else
  AGENTS_SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
AGENTS_ROOT="$(cd "$AGENTS_SCRIPTS/.." && pwd)"
REPO_ROOT="$(cd "$AGENTS_ROOT/.." && pwd)"
SIF_PATH="${SIF_PATH:-$AGENTS_ROOT/pytorch-langgraph-sgl.sif}"

module load apptainer

nvidia-smi

HF_CACHE="${HF_CACHE:-$REPO_ROOT/cache/huggingface}"
export HF_DATASETS_CACHE="$HF_CACHE"
export TRANSFORMERS_CACHE="$HF_CACHE"
export HF_HOME="$HF_CACHE"

# Override path: SECRETS_FILE=/path/to/file sbatch run.sh
SECRETS_FILE="${SECRETS_FILE:-$AGENTS_SCRIPTS/.env.supabase}"
if [ -f "$SECRETS_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$SECRETS_FILE"
  set +a
else
  echo "Note: $SECRETS_FILE not found — set SUPABASE_* and UPLOAD_TO_SUPABASE there if you want uploads." >&2
fi

# TODO: upload should happen automatically
# If Supabase keys are set but UPLOAD_TO_SUPABASE was omitted, default to upload (set UPLOAD_TO_SUPABASE=0 to skip).
if [ -n "${SUPABASE_SERVICE_ROLE_KEY:-}" ] && [ -n "${SUPABASE_URL:-}" ]; then
  export UPLOAD_TO_SUPABASE="${UPLOAD_TO_SUPABASE:-1}"
fi

# Default CSV is train.csv (set in launch_sgl.sh). Override example:
# export GT_DATA_CSV="$REPO_ROOT/data/reddit_comment_text_1000.csv"

# --- Apptainer launch options (uncomment the one that matches your setup) ---

# Option A — shared project storage:
# apptainer exec -C --nv --home /scratch/lingjzhu/vllm_env_home -W /scratch/lingjzhu/vllm_env_home -B /project -B /scratch pytorch-langgraph-sgl.sif bash /lustre09/project/6102159/lingjzhu/agents/launch_sgl.sh

# Option B — portable (all paths derived from repo location; works for any account after cloning):
# APPTAINER_HOME="${APPTAINER_HOME:-$REPO_ROOT/vllm_env_home}"
# mkdir -p "$APPTAINER_HOME"
# apptainer exec -C --nv --home "$APPTAINER_HOME" -W "$APPTAINER_HOME" -B /project -B /scratch "$SIF_PATH" bash "$AGENTS_SCRIPTS/launch_sgl.sh"

# # Option C — nimamot's local paths (currently active):
apptainer exec -C --nv --home /scratch/nimamot/vllm_env_home -W /scratch/nimamot/vllm_env_home -B /project -B /scratch "$SIF_PATH" bash "$AGENTS_SCRIPTS/launch_sgl.sh"
