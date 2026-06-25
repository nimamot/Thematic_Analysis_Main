#!/usr/bin/env python3
"""Upload LLM codebook (v1) to Supabase for human review."""

from __future__ import annotations

import os
import sys

from agents.core.codebook_review import review_backend, review_meta_from_env, stage_v1_for_review
from agents.core.paths import CODEBOOK_PATH, CLUSTERED_CODES_PATH, display_path


def main() -> int:
    if not CODEBOOK_PATH.is_file():
        print(f"upload_codebook_for_review: missing {display_path(CODEBOOK_PATH)}", file=sys.stderr)
        return 1
    if not CLUSTERED_CODES_PATH.is_file():
        print(
            f"upload_codebook_for_review: missing {display_path(CLUSTERED_CODES_PATH)}",
            file=sys.stderr,
        )
        return 1

    slug = os.environ.get("PIPELINE_SLUG", "default").strip() or "default"
    rq = os.environ.get("RESEARCH_QUESTION", "").strip()
    try:
        review_id = stage_v1_for_review(slug, rq, meta=review_meta_from_env())
    except RuntimeError as e:
        print(f"upload_codebook_for_review: {e}", file=sys.stderr)
        return 1

    backend = review_backend()
    print(f"upload_codebook_for_review: review_id={review_id} slug={slug!r} backend={backend}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
