"""LangGraph node: pause for human codebook review via interrupt()."""

from __future__ import annotations

import os
from typing import Any, Dict

from langgraph.types import interrupt

from .codebook_review import (
    fetch_and_materialize_approved,
    human_review_enabled,
    materialize_approved_row,
    review_backend,
    review_meta_from_env,
    stage_v1_for_review,
)
from .state import GTState
from .utils import log_step


def codebook_review_gate(state: GTState) -> Dict[str, Any]:
    """Upload v1 to Supabase, interrupt for human approval, materialize v2 on resume."""
    if not human_review_enabled():
        return {"codebook_review_status": "skipped"}

    if state.get("codebook_review_status") == "approved":
        return {}

    slug = os.environ.get("PIPELINE_SLUG", "default").strip() or "default"
    rq = state.get("research_question", "")
    review_id = state.get("codebook_review_id")

    if not review_id:
        review_id = stage_v1_for_review(slug, rq, meta=review_meta_from_env())

    backend = review_backend()
    wait_hint = (
        "Waiting for human codebook approval via local viewer-data/"
        if backend == "local"
        else "Waiting for human codebook approval in Supabase"
    )
    resume_payload = interrupt(
        {
            "kind": "codebook_review",
            "review_id": review_id,
            "slug": slug,
            "message": wait_hint,
        }
    )

    if isinstance(resume_payload, dict):
        rid = resume_payload.get("review_id") or review_id
        if resume_payload.get("codebook_v2"):
            materialize_approved_row(
                {
                    "id": rid,
                    "codebook_v2": resume_payload["codebook_v2"],
                    "clustered_codes": resume_payload.get("clustered_codes"),
                    "approved_at": resume_payload.get("approved_at"),
                }
            )
        else:
            fetch_and_materialize_approved(review_id=str(rid), slug=slug)
    else:
        fetch_and_materialize_approved(review_id=str(review_id), slug=slug)

    log_step("CODEBOOK_REVIEW_GATE", f"approved review_id={review_id}")
    return {
        "codebook_review_id": review_id,
        "codebook_review_status": "approved",
    }
