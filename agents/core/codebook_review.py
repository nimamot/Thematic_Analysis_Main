"""Orchestration for codebook human review gate (Supabase + local materialization)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from .codebook_edits import apply_codebook_review, materialize_clustered_output
from .codebook_schema import build_enriched_from_artifacts
from .paths import (
    CLUSTERED_CODES_PATH,
    CODEBOOK_CONFIDENCE_PATH,
    CODEBOOK_PATH,
    CODEBOOK_PROVENANCE_PATH,
    ensure_output_dirs,
)
from .supabase_http import (
    codebook_reviews_fetch_by_id,
    codebook_reviews_fetch_latest_approved,
    codebook_reviews_fetch_pending,
    codebook_reviews_insert_row,
    codebook_reviews_poll_until_approved,
)
from .utils import log_step


def review_backend() -> str:
    """Return ``local`` or ``supabase`` for the codebook review gate."""
    explicit = os.environ.get("GT_CODEBOOK_REVIEW_BACKEND", "").strip().lower()
    if explicit in ("local", "supabase"):
        return explicit
    url = os.environ.get("SUPABASE_URL", "").strip()
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    if url and key:
        return "supabase"
    return "local"


def viewer_export_enabled() -> bool:
    from .viewer_export import viewer_export_enabled as _enabled

    return _enabled()


def human_review_enabled() -> bool:
    return os.environ.get("GT_CODEBOOK_REVIEW", "0").strip() in ("1", "true", "yes")


def review_mode() -> str:
    return os.environ.get("GT_CODEBOOK_REVIEW_MODE", "manual").strip().lower()


def _supabase_credentials() -> tuple[str, str]:
    url = os.environ.get("SUPABASE_URL", "").strip()
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    if not url or not key:
        raise RuntimeError("missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY")
    return url, key


def _load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: dict) -> None:
    ensure_output_dirs()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def build_v1_payload(
    *,
    codebook_path: Path = CODEBOOK_PATH,
    cluster_path: Path = CLUSTERED_CODES_PATH,
    confidence_path: Path = CODEBOOK_CONFIDENCE_PATH,
) -> dict:
    cb_data = _load_json(codebook_path)
    codebook = cb_data.get("codebook", {})
    cluster_to_codes = cb_data.get("cluster_to_codes") or _load_json(cluster_path).get(
        "cluster_to_codes", {}
    )
    confidence: Dict[str, Any] = {}
    if confidence_path.is_file():
        try:
            confidence = _load_json(confidence_path)
        except (json.JSONDecodeError, OSError):
            confidence = {}
    return build_enriched_from_artifacts(codebook, cluster_to_codes, confidence, version=1)


def stage_v1_for_review(
    slug: str,
    research_question: str,
    *,
    meta: Optional[dict] = None,
) -> str:
    """Export or upload v1 codebook for human review; returns review id (or slug for local)."""
    if review_backend() == "local":
        from .viewer_export import export_codebook_review

        export_codebook_review(slug, research_question)
        log_step("CODEBOOK_REVIEW_STAGED_LOCAL", f"slug={slug!r}")
        return slug
    return upload_v1_for_review(slug, research_question, meta=meta)


def upload_v1_for_review(
    slug: str,
    research_question: str,
    *,
    meta: Optional[dict] = None,
) -> str:
    """Upload v1 codebook for human review; returns review id."""
    url, key = _supabase_credentials()
    v1 = build_v1_payload()
    clustered = _load_json(CLUSTERED_CODES_PATH)
    confidence = _load_json(CODEBOOK_CONFIDENCE_PATH) if CODEBOOK_CONFIDENCE_PATH.is_file() else {}

    row: dict = {
        "slug": slug,
        "research_question": research_question or None,
        "status": "pending_review",
        "codebook_v1": v1,
        "clustered_codes": clustered,
        "codebook_confidence": confidence or None,
    }
    if meta:
        row["meta"] = meta

    status, body = codebook_reviews_insert_row(url, key, row)
    if not (200 <= status < 300):
        raise RuntimeError(f"codebook review upload failed HTTP {status}: {body[:2000]}")

    rows = json.loads(body) if body.strip() else []
    if not rows or not rows[0].get("id"):
        pending = codebook_reviews_fetch_pending(url, key, slug)
        if not pending:
            raise RuntimeError("upload succeeded but could not resolve review id")
        review_id = str(pending["id"])
    else:
        review_id = str(rows[0]["id"])

    log_step("CODEBOOK_REVIEW_UPLOADED", f"review_id={review_id} slug={slug!r}")
    return review_id


def materialize_approved_row(row: dict) -> Path:
    """Apply approved review row to local artifacts; returns CODEBOOK_PATH."""
    v2 = row.get("codebook_v2")
    if not v2:
        raise RuntimeError("approved row missing codebook_v2")
    clustered = row.get("clustered_codes") or _load_json(CLUSTERED_CODES_PATH)
    review_id = str(row.get("id") or "")
    approved_at = row.get("approved_at")

    result = apply_codebook_review(
        v2,
        clustered,
        review_id=review_id or None,
        approved_at=str(approved_at) if approved_at else None,
    )
    clustered_out = materialize_clustered_output(result, clustered)

    _write_json(
        CODEBOOK_PATH, {"codebook": result.codebook, "cluster_to_codes": result.cluster_to_codes}
    )
    _write_json(CODEBOOK_CONFIDENCE_PATH, result.codebook_confidence)
    _write_json(CLUSTERED_CODES_PATH, clustered_out)
    _write_json(CODEBOOK_PROVENANCE_PATH, result.provenance)

    log_step(
        "CODEBOOK_REVIEW_MATERIALIZED",
        f"review_id={review_id} clusters={len(result.codebook)}",
    )
    return CODEBOOK_PATH


def fetch_and_materialize_approved(
    *,
    review_id: Optional[str] = None,
    slug: Optional[str] = None,
) -> Path:
    url, key = _supabase_credentials()
    slug = slug or os.environ.get("PIPELINE_SLUG", "default").strip() or "default"
    if review_id:
        row = codebook_reviews_fetch_by_id(url, key, review_id)
    else:
        row = codebook_reviews_fetch_latest_approved(url, key, slug)
    if not row:
        raise RuntimeError(f"no approved codebook review for slug={slug!r}")
    if row.get("status") != "approved":
        raise RuntimeError(f"review {row.get('id')} status is {row.get('status')!r}, not approved")
    return materialize_approved_row(row)


def wait_for_approval(
    slug: Optional[str] = None,
    *,
    review_id: Optional[str] = None,
    timeout_sec: Optional[int] = None,
    interval_sec: Optional[int] = None,
) -> dict:
    slug = slug or os.environ.get("PIPELINE_SLUG", "default").strip() or "default"
    if review_backend() == "local":
        from .viewer_export import wait_for_local_approval

        wait_for_local_approval(
            slug,
            timeout_sec=timeout_sec,
            interval_sec=interval_sec,
        )
        return {"id": review_id or slug, "slug": slug, "status": "approved"}
    url, key = _supabase_credentials()
    if timeout_sec is None:
        timeout_sec = int(os.environ.get("GT_CODEBOOK_REVIEW_TIMEOUT_SEC", "86400"))
    if interval_sec is None:
        interval_sec = int(os.environ.get("GT_CODEBOOK_REVIEW_POLL_INTERVAL_SEC", "30"))

    log_step("CODEBOOK_REVIEW_WAIT", f"slug={slug!r} review_id={review_id or 'latest'}")
    row = codebook_reviews_poll_until_approved(
        url,
        key,
        slug,
        timeout_sec=timeout_sec,
        interval_sec=interval_sec,
        review_id=review_id,
    )
    materialize_approved_row(row)
    return row


def review_meta_from_env() -> dict:
    meta: dict = {}
    if os.environ.get("SLURM_JOB_ID"):
        meta["slurm_job_id"] = os.environ["SLURM_JOB_ID"]
    if os.environ.get("GIT_COMMIT"):
        meta["git_commit"] = os.environ["GIT_COMMIT"]
    return meta
