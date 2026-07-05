"""Effective pipeline configuration: read env, snapshot for logs and audit."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .codebook_review import human_review_enabled, review_backend, review_mode
from .inference_config import (
    embed_backend,
    embed_model_id,
    llm_model_name,
    openai_base,
    qualitative_enrichment_enabled,
    sentence_transformers_model_path,
)
from .llm_clustering import USE_LLM_CLUSTERING
from .paths import LOGS_DIR, REPO_ROOT, ensure_output_dirs

CONFIG_FILE = REPO_ROOT / "agents" / "scripts" / "pipeline_config.env"
SECRETS_FILE = REPO_ROOT / "agents" / "scripts" / ".env.supabase"
SNAPSHOT_PATH = LOGS_DIR / "pipeline_config.json"
EFFECTIVE_ENV_PATH = LOGS_DIR / "pipeline_config.effective.env"

# Keys owned by pipeline_config.env (secrets file is not listed here).
CONFIG_KEYS: tuple[str, ...] = (
    "RESEARCH_QUESTION",
    "GT_DATA_CSV",
    "PIPELINE_SLUG",
    "GT_LAUNCHER",
    "GT_CODEBOOK_REVIEW",
    "GT_CODEBOOK_REVIEW_MODE",
    "GT_CODEBOOK_REVIEW_BACKEND",
    "GT_CODEBOOK_REVIEW_TIMEOUT_SEC",
    "GT_CODEBOOK_REVIEW_POLL_INTERVAL_SEC",
    "GT_VIEWER_EXPORT",
    "GT_VIEWER_AUTO_LAUNCH",
    "UPLOAD_TO_SUPABASE",
    "GT_QUALITATIVE_ENRICHMENT",
    "GT_ENRICH_WORKERS",
    "GT_OPEN_CODING_WORKERS",
    "GT_HIGH_LEVEL_STRATEGY",
    "GT_HL_N_SAMPLES",
    "GT_HL_SAMPLE_SIZE",
    "GT_HIGH_LEVEL_WORKERS",
    "GT_HIERARCHY_WORKERS",
    "GT_HIERARCHY_REFINE",
    "GT_SGLANG_CONTEXT_LENGTH",
    "GT_USE_SKILLS",
)


def _truthy(raw: str | None, *, default: bool = False) -> bool:
    if raw is None or not str(raw).strip():
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()


def _resolve_data_csv() -> str:
    raw = _env("GT_DATA_CSV")
    if not raw:
        return str(REPO_ROOT / "data" / "train.csv")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return str(path)


def effective_config(*, where: str = "pipeline") -> dict[str, Any]:
    """Resolved configuration as seen by Python after load_pipeline_env."""
    supabase_configured = bool(_env("SUPABASE_URL") and _env("SUPABASE_SERVICE_ROLE_KEY"))
    backend = review_backend()
    return {
        "_meta": {
            "where": where,
            "config_file": str(CONFIG_FILE),
            "secrets_file": str(SECRETS_FILE),
            "priority": [
                "1. Shell exports before sbatch (one-off overrides)",
                "2. agents/scripts/pipeline_config.env",
                "3. agents/scripts/.env.supabase (SUPABASE_* only)",
                "4. Code defaults",
            ],
        },
        "study": {
            "RESEARCH_QUESTION": _env(
                "RESEARCH_QUESTION", "What thematic patterns emerge across these reviews?"
            ),
            "GT_DATA_CSV": _resolve_data_csv(),
            "PIPELINE_SLUG": _env("PIPELINE_SLUG", "default") or "default",
        },
        "launcher": {
            "GT_LAUNCHER": _env("GT_LAUNCHER", "sgl") or "sgl",
            "SLURM_JOB_ID": _env("SLURM_JOB_ID"),
        },
        "codebook_review": {
            "GT_CODEBOOK_REVIEW": _env("GT_CODEBOOK_REVIEW", "0"),
            "enabled": human_review_enabled(),
            "GT_CODEBOOK_REVIEW_MODE": review_mode(),
            "GT_CODEBOOK_REVIEW_BACKEND": _env("GT_CODEBOOK_REVIEW_BACKEND") or "auto",
            "resolved_backend": backend,
            "GT_CODEBOOK_REVIEW_TIMEOUT_SEC": int(
                _env("GT_CODEBOOK_REVIEW_TIMEOUT_SEC", "86400") or "86400"
            ),
            "GT_CODEBOOK_REVIEW_POLL_INTERVAL_SEC": int(
                _env("GT_CODEBOOK_REVIEW_POLL_INTERVAL_SEC", "30") or "30"
            ),
            "GT_VIEWER_AUTO_LAUNCH": _truthy(_env("GT_VIEWER_AUTO_LAUNCH", "1"), default=True),
            "SUPABASE_CREDENTIALS": supabase_configured,
        },
        "export_upload": {
            "GT_VIEWER_EXPORT": _truthy(_env("GT_VIEWER_EXPORT", "1"), default=True),
            "UPLOAD_TO_SUPABASE": _truthy(_env("UPLOAD_TO_SUPABASE", "0")),
        },
        "qualitative_enrichment": {
            "GT_QUALITATIVE_ENRICHMENT": _env("GT_QUALITATIVE_ENRICHMENT", "1"),
            "enabled": qualitative_enrichment_enabled(),
            "GT_ENRICH_WORKERS": int(_env("GT_ENRICH_WORKERS", "4") or "4"),
        },
        "inference": {
            "GT_OPENAI_BASE": openai_base(),
            "GT_LLM_MODEL": llm_model_name(),
            "GT_EMBED_BACKEND": embed_backend(),
            "GT_EMBED_MODEL": _env("GT_EMBED_MODEL") or sentence_transformers_model_path(),
            "embed_model_id_lmstudio": embed_model_id(),
            "GT_SGLANG_CONTEXT_LENGTH": _env("GT_SGLANG_CONTEXT_LENGTH"),
        },
        "workers_and_strategy": {
            "GT_OPEN_CODING_WORKERS": int(_env("GT_OPEN_CODING_WORKERS", "8") or "8"),
            "GT_HIGH_LEVEL_STRATEGY": _env("GT_HIGH_LEVEL_STRATEGY", "nsampling"),
            "GT_HL_N_SAMPLES": int(_env("GT_HL_N_SAMPLES", "5") or "5"),
            "GT_HL_SAMPLE_SIZE": int(_env("GT_HL_SAMPLE_SIZE", "15") or "15"),
            "GT_HIGH_LEVEL_WORKERS": int(_env("GT_HIGH_LEVEL_WORKERS", "8") or "8"),
            "GT_HIERARCHY_WORKERS": int(_env("GT_HIERARCHY_WORKERS", "8") or "8"),
            "GT_HIERARCHY_REFINE": _truthy(_env("GT_HIERARCHY_REFINE", "1"), default=True),
            "GT_USE_SKILLS": _truthy(_env("GT_USE_SKILLS", "1"), default=True),
        },
        "clustering": {
            "USE_LLM_CLUSTERING": bool(USE_LLM_CLUSTERING),
            "GT_LLM_CLUSTER_MAX_ITER": _env("GT_LLM_CLUSTER_MAX_ITER", "3"),
            "GT_LLM_CLUSTER_BATCH_SIZE": _env("GT_LLM_CLUSTER_BATCH_SIZE", "100"),
        },
    }


def _format_block(title: str, data: dict[str, Any], indent: int = 0) -> list[str]:
    pad = "  " * indent
    lines = [f"{pad}[{title}]"]
    for key, value in data.items():
        if isinstance(value, dict):
            lines.extend(_format_block(key, value, indent + 1))
        else:
            lines.append(f"{pad}  {key}={value!r}")
    return lines


def format_config_report(cfg: dict[str, Any]) -> str:
    lines: list[str] = []
    meta = cfg.get("_meta", {})
    lines.append("Pipeline configuration (effective values)")
    lines.append(f"  where: {meta.get('where', 'pipeline')}")
    lines.append(f"  config_file: {meta.get('config_file')}")
    lines.append(f"  secrets_file: {meta.get('secrets_file')} (SUPABASE_* only)")
    lines.append("  priority:")
    for item in meta.get("priority", []):
        lines.append(f"    {item}")
    lines.append("")
    for section, payload in cfg.items():
        if section == "_meta":
            continue
        lines.extend(_format_block(section, payload))
        lines.append("")
    return "\n".join(lines).rstrip()


def write_config_snapshot(*, where: str = "pipeline") -> Path:
    ensure_output_dirs()
    cfg = effective_config(where=where)
    SNAPSHOT_PATH.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")

    env_lines = [
        "# Effective pipeline env (auto-generated; edit pipeline_config.env instead)",
        f"# where={where}",
        "",
    ]
    for section, payload in cfg.items():
        if section == "_meta":
            continue
        env_lines.append(f"# --- {section} ---")
        for key, value in payload.items():
            if isinstance(value, (dict, list)):
                continue
            env_lines.append(f"{key}={value}")
        env_lines.append("")
    EFFECTIVE_ENV_PATH.write_text("\n".join(env_lines), encoding="utf-8")
    return SNAPSHOT_PATH


def log_effective_config(*, where: str = "python") -> None:
    from .utils import log_step

    cfg = effective_config(where=where)
    write_config_snapshot(where=where)
    log_step("PIPELINE_CONFIG", format_config_report(cfg))


def log_effective_config_once(*, where: str = "python") -> None:
    """Log full config once per Slurm job (or local run)."""
    ensure_output_dirs()
    job = os.environ.get("SLURM_JOB_ID", "local")
    marker = LOGS_DIR / f".pipeline_config_logged_{job}"
    if marker.is_file():
        return
    log_effective_config(where=where)
    marker.touch()
