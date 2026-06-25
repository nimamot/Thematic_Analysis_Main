#!/usr/bin/env python3
"""Export pipeline artifacts into viewer-data/ for the local Graph Builder viewer."""

from __future__ import annotations

import argparse
import os
import sys

from agents.core.viewer_export import (
    ensure_viewer_scaffold,
    export_codebook_review,
    export_project,
    viewer_export_enabled,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export pipeline outputs to viewer-data/.")
    parser.add_argument(
        "--mode",
        choices=("codebook-review", "project", "all"),
        default="project",
        help="codebook-review: after high-level; project: finished run; all: both when files exist",
    )
    parser.add_argument("--slug", help="Study slug (default: PIPELINE_SLUG)")
    parser.add_argument(
        "--research-question",
        default="",
        help="Research question for meta.json (default: RESEARCH_QUESTION env or report)",
    )
    args = parser.parse_args()

    if not viewer_export_enabled():
        print("export_viewer_data: skipped (GT_VIEWER_EXPORT=0)", file=sys.stderr)
        return 0

    slug = (args.slug or os.environ.get("PIPELINE_SLUG", "default")).strip() or "default"
    root = ensure_viewer_scaffold()

    try:
        if args.mode in ("codebook-review", "all"):
            export_codebook_review(slug, args.research_question, root=root)
            print(f"export_viewer_data: codebook review exported for slug={slug!r}")
        if args.mode in ("project", "all"):
            export_project(slug, args.research_question, root=root)
            print(f"export_viewer_data: project exported for slug={slug!r}")
    except RuntimeError as e:
        print(f"export_viewer_data: {e}", file=sys.stderr)
        return 1

    print(f"export_viewer_data: viewer root {root}")
    print(f"  python tools/viewer_launcher.py --data-dir {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
