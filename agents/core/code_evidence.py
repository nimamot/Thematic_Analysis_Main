"""Build code_evidence.json for the codebook review viewer (hover quotes)."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import paths
from .source_memory import Snippet, SourceMemory


def _iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _snippet_payload(snippet: Snippet) -> dict[str, Any]:
    return {
        "snippet_id": snippet.snippet_id,
        "review_id": snippet.review_id,
        "source_id": snippet.source_id,
        "quote": snippet.quote,
        "note": snippet.note or "",
    }


def _aliases_for_cluster_code(code: str, dedup_map: dict[str, str]) -> set[str]:
    aliases = {code}
    aliases.update(orig for orig, canon in dedup_map.items() if canon == code)
    return aliases


def _collect_snippets(
    code: str,
    grouped: dict[str, list[Snippet]],
    dedup_map: dict[str, str],
) -> list[Snippet]:
    seen: set[str] = set()
    collected: list[Snippet] = []
    for alias in _aliases_for_cluster_code(code, dedup_map):
        for snippet in grouped.get(alias, []):
            if snippet.snippet_id in seen:
                continue
            seen.add(snippet.snippet_id)
            collected.append(snippet)
    return sorted(collected, key=lambda s: (s.review_id, s.snippet_id))


def _empty_entry() -> dict[str, Any]:
    return {
        "quote_count": 0,
        "review_count": 0,
        "primary": None,
        "snippets": [],
    }


def build_code_evidence(
    memory: SourceMemory,
    *,
    slug: str,
    research_question: str = "",
    open_codes: set[str] | None = None,
    dedup_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Group source-memory snippets by open code for viewer hover tooltips."""
    dedup_map = dedup_map or {}
    grouped: dict[str, list[Snippet]] = {}
    for snippet in memory.snippets:
        grouped.setdefault(snippet.open_code, []).append(snippet)

    codes_to_export = set(open_codes or grouped.keys())
    by_open_code: dict[str, Any] = {}
    for code in sorted(codes_to_export):
        ordered = _collect_snippets(code, grouped, dedup_map)
        review_ids = sorted({s.review_id for s in ordered})
        payload = [_snippet_payload(s) for s in ordered]
        by_open_code[code] = {
            "quote_count": len(ordered),
            "review_count": len(review_ids),
            "primary": payload[0] if payload else None,
            "snippets": payload,
        }

    return {
        "version": 1,
        "slug": slug,
        "research_question": research_question,
        "generated_at": _iso_now(),
        "by_open_code": by_open_code,
    }


def _resolve_data_csv() -> Path | None:
    raw = os.environ.get("GT_DATA_CSV", "").strip()
    if raw:
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = paths.REPO_ROOT / path
        return path if path.is_file() else None
    return None


def load_source_memory_for_export() -> SourceMemory | None:
    """Rebuild source memory from open-coding markdown when available."""
    if paths.OPEN_CODES_MARKDOWN_PATH.is_file():
        return SourceMemory.build(paths.OPEN_CODES_MARKDOWN_PATH, _resolve_data_csv())
    if paths.SOURCE_MEMORY_PATH.is_file():
        try:
            return SourceMemory.load(paths.SOURCE_MEMORY_PATH)
        except (OSError, ValueError, KeyError, TypeError):
            return None
    return None


def open_codes_from_codebook(codebook_path: Path) -> set[str]:
    """Collect open-code strings referenced by cluster_to_codes."""
    if not codebook_path.is_file():
        return set()
    import json

    with open(codebook_path, encoding="utf-8") as f:
        data = json.load(f)
    cluster_to_codes = data.get("cluster_to_codes") or {}
    codes: set[str] = set()
    for values in cluster_to_codes.values():
        if not isinstance(values, list):
            continue
        for code in values:
            if isinstance(code, str) and code.strip():
                codes.add(code)
    return codes


def dedup_map_from_clustered(clustered_path: Path) -> dict[str, str]:
    if not clustered_path.is_file():
        return {}
    import json

    with open(clustered_path, encoding="utf-8") as f:
        data = json.load(f)
    raw = data.get("dedup_map") or {}
    return {str(k): str(v) for k, v in raw.items() if isinstance(k, str) and isinstance(v, str)}
