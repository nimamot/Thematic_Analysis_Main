"""Export pipeline artifacts into viewer-data/ for the local Graph Builder viewer."""

from __future__ import annotations

import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .codebook_edits import apply_codebook_review
from .paths import (
    CLUSTERED_CODES_PATH,
    CODEBOOK_CONFIDENCE_PATH,
    CODEBOOK_PATH,
    COOCCURRENCE_PATH,
    GLOBAL_GRAPH_PATH,
    OPEN_CODES_MARKDOWN_PATH,
    RESEARCH_REPORT_PATH,
    VIEWER_DATA_DIR,
    ensure_output_dirs,
)
from .utils import log_step


def viewer_export_enabled() -> bool:
    return os.environ.get("GT_VIEWER_EXPORT", "1").strip() not in ("0", "false", "no")


def _viewer_root() -> Path:
    override = os.environ.get("VIEWER_DATA_DIR", "").strip()
    return Path(override).expanduser().resolve() if override else VIEWER_DATA_DIR


def _load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def _iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _research_question_from_report_md(text: str) -> str:
    marker = "## Research question"
    if marker not in text:
        return ""
    after = text.split(marker, 1)[1]
    for line in after.splitlines():
        s = line.strip()
        if not s:
            continue
        if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
            return s[1:-1].strip()
        return s
    return ""


def _research_question_from_graph(graph: dict) -> str:
    tree = graph.get("tree")
    if not isinstance(tree, dict):
        return ""
    name = tree.get("name")
    return name.strip() if isinstance(name, str) and name.strip() else ""


def resolve_research_question(explicit: str = "") -> str:
    rq = (explicit or os.environ.get("RESEARCH_QUESTION", "")).strip()
    if rq:
        return rq
    if RESEARCH_REPORT_PATH.is_file():
        try:
            rq = _research_question_from_report_md(RESEARCH_REPORT_PATH.read_text(encoding="utf-8"))
            if rq:
                return rq
        except OSError:
            pass
    if GLOBAL_GRAPH_PATH.is_file():
        try:
            rq = _research_question_from_graph(_load_json(GLOBAL_GRAPH_PATH))
            if rq:
                return rq
        except (OSError, json.JSONDecodeError):
            pass
    return ""


def ensure_viewer_scaffold(root: Path | None = None) -> Path:
    """Create viewer-data layout with an empty manifest if missing."""
    root = root or _viewer_root()
    (root / "projects").mkdir(parents=True, exist_ok=True)
    (root / "codebook-reviews").mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        _write_json(manifest_path, {"projects": [], "codebook_reviews": []})
    return root


def _read_manifest(root: Path) -> dict:
    ensure_viewer_scaffold(root)
    manifest_path = root / "manifest.json"
    try:
        data = _load_json(manifest_path)
    except (OSError, json.JSONDecodeError):
        data = {}
    projects = data.get("projects")
    reviews = data.get("codebook_reviews")
    return {
        "projects": list(projects) if isinstance(projects, list) else [],
        "codebook_reviews": list(reviews) if isinstance(reviews, list) else [],
    }


def _write_manifest(root: Path, manifest: dict) -> None:
    _write_json(root / "manifest.json", manifest)


def _add_slug(manifest: dict, key: str, slug: str) -> None:
    slugs: list[str] = manifest.setdefault(key, [])
    if slug not in slugs:
        slugs.append(slug)


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.is_file():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _review_meta(slug: str, research_question: str) -> dict:
    meta: dict[str, Any] = {
        "id": slug,
        "slug": slug,
        "created_at": _iso_now(),
        "updated_at": _iso_now(),
    }
    if research_question:
        meta["research_question"] = research_question
    return meta


def _project_meta(slug: str, research_question: str) -> dict:
    return _review_meta(slug, research_question)


def codebook_review_dir(root: Path, slug: str) -> Path:
    return root / "codebook-reviews" / slug


def local_codebook_v2_path(slug: str, *, root: Path | None = None) -> Path:
    """Where researchers drop the approved codebook_v2 export from the viewer."""
    return codebook_review_dir(root or _viewer_root(), slug) / "codebook_v2.json"


def export_codebook_review(
    slug: str,
    research_question: str = "",
    *,
    root: Path | None = None,
) -> Path:
    """Copy v1 review artifacts into viewer-data/codebook-reviews/<slug>/."""
    root = ensure_viewer_scaffold(root or _viewer_root())
    review_dir = codebook_review_dir(root, slug)
    review_dir.mkdir(parents=True, exist_ok=True)

    missing: list[str] = []
    for label, src, name in (
        ("codebook", CODEBOOK_PATH, "codebook.json"),
        ("clustered codes", CLUSTERED_CODES_PATH, "gt_clustered_codes.json"),
    ):
        if not _copy_if_exists(src, review_dir / name):
            missing.append(label)

    _copy_if_exists(CODEBOOK_CONFIDENCE_PATH, review_dir / "codebook_confidence.json")

    if missing:
        raise RuntimeError(f"viewer export missing required artifacts: {', '.join(missing)}")

    rq = resolve_research_question(research_question)
    _write_json(review_dir / "meta.json", _review_meta(slug, rq))

    manifest = _read_manifest(root)
    _add_slug(manifest, "codebook_reviews", slug)
    _write_manifest(root, manifest)

    log_step("VIEWER_CODEBOOK_REVIEW_EXPORTED", f"slug={slug!r} dir={review_dir}")
    return review_dir


def export_project(
    slug: str,
    research_question: str = "",
    *,
    root: Path | None = None,
) -> Path:
    """Copy finished pipeline artifacts into viewer-data/projects/<slug>/."""
    root = ensure_viewer_scaffold(root or _viewer_root())
    project_dir = root / "projects" / slug
    project_dir.mkdir(parents=True, exist_ok=True)

    missing: list[str] = []
    for label, src, name in (
        ("global graph", GLOBAL_GRAPH_PATH, "gt_global_graph.json"),
        ("research report", RESEARCH_REPORT_PATH, "research_report.md"),
    ):
        if not _copy_if_exists(src, project_dir / name):
            missing.append(label)

    _copy_if_exists(OPEN_CODES_MARKDOWN_PATH, project_dir / "gt_open_codes_all_reviews.md")
    _copy_if_exists(COOCCURRENCE_PATH, project_dir / "cooccurrence.json")

    if missing:
        raise RuntimeError(f"viewer export missing required artifacts: {', '.join(missing)}")

    rq = resolve_research_question(research_question)
    _write_json(project_dir / "meta.json", _project_meta(slug, rq))

    manifest = _read_manifest(root)
    _add_slug(manifest, "projects", slug)
    _write_manifest(root, manifest)

    log_step("VIEWER_PROJECT_EXPORTED", f"slug={slug!r} dir={project_dir}")
    return project_dir


def _find_local_v2(slug: str, root: Path) -> Path | None:
    review_dir = codebook_review_dir(root, slug)
    canonical = review_dir / "codebook_v2.json"
    if canonical.is_file():
        return canonical
    if review_dir.is_dir():
        matches = sorted(review_dir.glob("*-codebook_v2.json"))
        if matches:
            return matches[-1]
    return None


def materialize_local_codebook_v2(v2_path: Path, *, review_id: str | None = None) -> Path:
    """Apply a viewer-exported codebook_v2.json to agents/outputs/data/."""
    v2 = _load_json(v2_path)
    clustered = _load_json(CLUSTERED_CODES_PATH)
    result = apply_codebook_review(v2, clustered, review_id=review_id)
    from .codebook_edits import materialize_clustered_output

    clustered_out = materialize_clustered_output(result, clustered)
    ensure_output_dirs()
    _write_json(CODEBOOK_PATH, {"codebook": result.codebook, "cluster_to_codes": result.cluster_to_codes})
    _write_json(CODEBOOK_CONFIDENCE_PATH, result.codebook_confidence)
    _write_json(CLUSTERED_CODES_PATH, clustered_out)
    from .paths import CODEBOOK_PROVENANCE_PATH

    _write_json(CODEBOOK_PROVENANCE_PATH, result.provenance)
    log_step("CODEBOOK_REVIEW_MATERIALIZED", f"from={v2_path} clusters={len(result.codebook)}")
    return CODEBOOK_PATH


def wait_for_local_approval(
    slug: str,
    *,
    root: Path | None = None,
    timeout_sec: int | None = None,
    interval_sec: int | None = None,
) -> Path:
    """Poll viewer-data until codebook_v2.json appears, then materialize pipeline artifacts."""
    root = ensure_viewer_scaffold(root or _viewer_root())
    if timeout_sec is None:
        timeout_sec = int(os.environ.get("GT_CODEBOOK_REVIEW_TIMEOUT_SEC", "86400"))
    if interval_sec is None:
        interval_sec = int(os.environ.get("GT_CODEBOOK_REVIEW_POLL_INTERVAL_SEC", "30"))

    review_dir = codebook_review_dir(root, slug)
    drop_path = local_codebook_v2_path(slug, root=root)
    log_step(
        "CODEBOOK_REVIEW_WAIT_LOCAL",
        f"slug={slug!r} drop={drop_path}",
    )
    print(
        f"\nLocal codebook review: open the viewer and approve the review for {slug!r}.\n"
        f"  python tools/viewer_launcher.py --data-dir {root}\n"
        f"Then save the downloaded export as:\n"
        f"  {drop_path}\n"
        f"(or any *-codebook_v2.json in {review_dir})\n",
        flush=True,
    )

    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        v2_path = _find_local_v2(slug, root)
        if v2_path is not None:
            return materialize_local_codebook_v2(v2_path)
        time.sleep(interval_sec)

    raise TimeoutError(
        f"codebook review for slug={slug!r} not approved within {timeout_sec}s "
        f"(expected {drop_path})"
    )
