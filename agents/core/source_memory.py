"""Deterministic source-memory index for open-coding snippets and example grounding."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Set, TypedDict

from .paths import CODE_ID_MAP_PATH, SOURCE_MEMORY_PATH, ensure_output_dirs
from .utils import log_step

MEMORY_VERSION = 1
ENRICHED_SCHEMA_VERSION = 1

_CODE_EVIDENCE_RE = re.compile(
    r"-\s*Code:\s*(.+?)\s*\n\s+Evidence:\s*\"(.+?)\"\s*"
    r"(?:\n\s+Note:\s*(.+?)\s*)?"
    r"(?=\n\s*-\s*(?:Code:|Applicability:)|\n##|\Z)",
    re.DOTALL | re.IGNORECASE,
)


class GroundedExample(TypedDict):
    snippet_id: str
    review_id: int
    source_id: str | None
    open_code_id: str | None
    open_code: str
    quote: str


@dataclass(frozen=True)
class Snippet:
    snippet_id: str
    review_id: int
    source_id: str | None
    open_code: str
    open_code_id: str | None
    quote: str
    note: str = ""


def normalize_quote(quote: str) -> str:
    """Normalize quote text for exact lookup (strip + NFKC)."""
    return unicodedata.normalize("NFKC", quote.strip())


def parse_open_coding_snippets(md_path: Path) -> List[Snippet]:
    """Parse gt_open_codes_all_reviews.md into review-aware snippet records."""
    if not md_path.is_file():
        return []

    text = md_path.read_text(encoding="utf-8")
    blocks = re.split(r"^## Review (\d+)\s*$", text, flags=re.MULTILINE)
    snippets: List[Snippet] = []
    snippet_idx = 0

    for i in range(1, len(blocks), 2):
        review_id = int(blocks[i])
        block = blocks[i + 1]
        if re.search(r"Applicability:\s*NONE", block, re.IGNORECASE):
            continue
        for m in _CODE_EVIDENCE_RE.finditer(block):
            code = m.group(1).strip()
            quote = m.group(2).strip()
            note = (m.group(3) or "").strip()
            if not quote:
                continue
            snippet_idx += 1
            snippets.append(
                Snippet(
                    snippet_id=f"SNIP-{snippet_idx:04d}",
                    review_id=review_id,
                    source_id=None,
                    open_code=code,
                    open_code_id=None,
                    quote=quote,
                    note=note,
                )
            )
    return snippets


def load_review_source_ids(csv_path: Path) -> Dict[int, str | None]:
    """Map 1-based review_id to optional CSV row id (id / csv_id / review_id column)."""
    if not csv_path.is_file():
        return {}

    try:
        import pandas as pd

        df = pd.read_csv(csv_path)
        id_col = None
        for candidate in ("id", "csv_id", "review_id"):
            if candidate in df.columns:
                id_col = candidate
                break

        out: Dict[int, str | None] = {}
        for idx in range(len(df)):
            review_id = idx + 1
            if id_col is not None:
                val = df.iloc[idx][id_col]
                out[review_id] = None if pd.isna(val) else str(val)
            else:
                out[review_id] = None
        return out
    except ImportError:
        import csv

        out: Dict[int, str | None] = {}
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            id_col = next((c for c in ("id", "csv_id", "review_id") if c in fieldnames), None)
            for idx, row in enumerate(reader):
                review_id = idx + 1
                if id_col is not None:
                    val = (row.get(id_col) or "").strip()
                    out[review_id] = val or None
                else:
                    out[review_id] = None
        return out


def _load_code_id_map(path: Path = CODE_ID_MAP_PATH) -> Dict[str, str]:
    if not path.is_file():
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {str(k): str(v) for k, v in data.get("code_to_id", {}).items()}


@dataclass
class SourceMemory:
    version: int = MEMORY_VERSION
    data_csv: str = ""
    reviews: List[Dict[str, Any]] = field(default_factory=list)
    snippets: List[Snippet] = field(default_factory=list)
    by_snippet_id: Dict[str, int] = field(default_factory=dict)
    by_quote: Dict[str, List[str]] = field(default_factory=dict)
    _by_open_code: Dict[str, List[int]] = field(default_factory=dict, repr=False)

    @classmethod
    def build(
        cls,
        md_path: Path,
        csv_path: Path | None = None,
        code_id_map_path: Path = CODE_ID_MAP_PATH,
    ) -> SourceMemory:
        raw_snippets = parse_open_coding_snippets(md_path)
        source_ids = load_review_source_ids(csv_path) if csv_path else {}
        code_to_id = _load_code_id_map(code_id_map_path)

        snippets: List[Snippet] = []
        review_ids_seen: Set[int] = set()
        for s in raw_snippets:
            review_ids_seen.add(s.review_id)
            snippets.append(
                Snippet(
                    snippet_id=s.snippet_id,
                    review_id=s.review_id,
                    source_id=source_ids.get(s.review_id),
                    open_code=s.open_code,
                    open_code_id=code_to_id.get(s.open_code),
                    quote=s.quote,
                    note=s.note,
                )
            )

        reviews = [
            {"review_id": rid, "source_id": source_ids.get(rid)} for rid in sorted(review_ids_seen)
        ]
        mem = cls(
            version=MEMORY_VERSION,
            data_csv=str(csv_path) if csv_path else "",
            reviews=reviews,
            snippets=snippets,
        )
        mem._build_indexes()
        return mem

    def _build_indexes(self) -> None:
        self.by_snippet_id = {s.snippet_id: i for i, s in enumerate(self.snippets)}
        self.by_quote = {}
        self._by_open_code = {}
        for i, s in enumerate(self.snippets):
            key = normalize_quote(s.quote)
            self.by_quote.setdefault(key, []).append(s.snippet_id)
            self._by_open_code.setdefault(s.open_code, []).append(i)

    def save(self, path: Path = SOURCE_MEMORY_PATH) -> None:
        ensure_output_dirs()
        payload = {
            "version": self.version,
            "data_csv": self.data_csv,
            "reviews": self.reviews,
            "snippets": [
                {
                    "snippet_id": s.snippet_id,
                    "review_id": s.review_id,
                    "source_id": s.source_id,
                    "open_code": s.open_code,
                    "open_code_id": s.open_code_id,
                    "quote": s.quote,
                    "note": s.note,
                }
                for s in self.snippets
            ],
            "by_snippet_id": self.by_snippet_id,
            "by_quote": self.by_quote,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path = SOURCE_MEMORY_PATH) -> SourceMemory:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        snippets = [
            Snippet(
                snippet_id=s["snippet_id"],
                review_id=int(s["review_id"]),
                source_id=s.get("source_id"),
                open_code=s["open_code"],
                open_code_id=s.get("open_code_id"),
                quote=s["quote"],
                note=s.get("note", ""),
            )
            for s in data.get("snippets", [])
        ]
        mem = cls(
            version=int(data.get("version", MEMORY_VERSION)),
            data_csv=str(data.get("data_csv", "")),
            reviews=list(data.get("reviews", [])),
            snippets=snippets,
            by_snippet_id={str(k): int(v) for k, v in data.get("by_snippet_id", {}).items()},
            by_quote={str(k): list(v) for k, v in data.get("by_quote", {}).items()},
        )
        mem._build_indexes()
        return mem

    def snippet_at(self, snippet_id: str) -> Snippet | None:
        idx = self.by_snippet_id.get(snippet_id)
        if idx is None:
            return None
        return self.snippets[idx]

    def pick_snippet_for_code(
        self,
        open_code: str,
        *,
        used: Set[str] | None = None,
        review_hint: int | None = None,
    ) -> Snippet | None:
        """Return first unused snippet for open_code (stable review_id order)."""
        used = used or set()
        indices = sorted(
            self._by_open_code.get(open_code, []),
            key=lambda i: (
                0 if review_hint is not None and self.snippets[i].review_id == review_hint else 1,
                self.snippets[i].review_id,
                i,
            ),
        )
        for i in indices:
            s = self.snippets[i]
            if s.snippet_id not in used:
                return s
        return None

    def ground_quote(
        self,
        quote: str,
        *,
        open_code: str | None = None,
        open_code_id: str | None = None,
    ) -> GroundedExample | None:
        key = normalize_quote(quote)
        candidate_ids = self.by_quote.get(key, [])
        if not candidate_ids:
            log_step("GROUND_WARN", f"No snippet for quote: {quote[:80]!r}")
            return None

        candidates = [self.snippet_at(sid) for sid in candidate_ids]
        candidates = [c for c in candidates if c is not None]

        if open_code_id:
            filtered = [c for c in candidates if c.open_code_id == open_code_id]
            if filtered:
                candidates = filtered
        if open_code:
            filtered = [c for c in candidates if c.open_code == open_code]
            if filtered:
                candidates = filtered

        if len(candidates) > 1:
            candidates.sort(key=lambda c: (c.review_id, c.snippet_id))
            log_step(
                "GROUND_WARN",
                f"Ambiguous quote resolved to {candidates[0].snippet_id} "
                f"({len(candidate_ids)} matches)",
            )
        if not candidates:
            log_step("GROUND_WARN", f"Could not disambiguate quote: {quote[:80]!r}")
            return None
        return to_grounded_example(candidates[0])

    def snippets_for_code_evidence(self) -> tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        """Aggregate snippets into legacy code_evidence / code_notes dicts."""
        from collections import defaultdict

        code_evidence: Dict[str, List[str]] = defaultdict(list)
        code_notes: Dict[str, List[str]] = defaultdict(list)
        for s in self.snippets:
            if s.quote not in code_evidence[s.open_code]:
                code_evidence[s.open_code].append(s.quote)
            if s.note and s.note not in code_notes[s.open_code]:
                code_notes[s.open_code].append(s.note)
        return dict(code_evidence), dict(code_notes)


def to_grounded_example(snippet: Snippet) -> GroundedExample:
    return GroundedExample(
        snippet_id=snippet.snippet_id,
        review_id=snippet.review_id,
        source_id=snippet.source_id,
        open_code_id=snippet.open_code_id,
        open_code=snippet.open_code,
        quote=snippet.quote,
    )


def resolve_snippet(snippet_id: str, memory: SourceMemory | None = None) -> dict | None:
    mem = memory or SourceMemory.load()
    s = mem.snippet_at(snippet_id)
    return to_grounded_example(s) if s else None


def resolve_quote(
    quote: str,
    *,
    open_code: str | None = None,
    open_code_id: str | None = None,
    memory: SourceMemory | None = None,
) -> dict | None:
    mem = memory or SourceMemory.load()
    return mem.ground_quote(quote, open_code=open_code, open_code_id=open_code_id)


def load_or_build_source_memory(
    md_path: Path,
    csv_path: Path | None = None,
    memory_path: Path = SOURCE_MEMORY_PATH,
    code_id_map_path: Path = CODE_ID_MAP_PATH,
) -> SourceMemory:
    """Load source memory; rebuild when missing or markdown is newer."""
    if memory_path.is_file() and md_path.is_file():
        if memory_path.stat().st_mtime >= md_path.stat().st_mtime:
            try:
                return SourceMemory.load(memory_path)
            except (json.JSONDecodeError, KeyError, TypeError):
                log_step("SOURCE_MEMORY", "Corrupt memory file — rebuilding")

    mem = SourceMemory.build(md_path, csv_path, code_id_map_path)
    mem.save(memory_path)
    return mem


def _example_quote(ex: Any) -> str:
    if isinstance(ex, dict):
        return str(ex.get("quote", ""))
    return str(ex)


def ground_criterion_examples(
    item: Dict[str, Any],
    memory: SourceMemory,
    *,
    open_code_resolver: Any | None = None,
    cluster_codes_resolver: Any | None = None,
) -> None:
    """Convert string examples to grounded objects in-place."""
    examples = item.get("examples", [])
    if not examples:
        return

    grounded: List[GroundedExample] = []
    code_ids = item.get("code_ids", [])

    for i, ex in enumerate(examples):
        if isinstance(ex, dict) and ex.get("snippet_id"):
            grounded.append(ex)  # type: ignore[arg-type]
            continue

        quote = _example_quote(ex)
        candidate_codes: List[str] = []
        if cluster_codes_resolver and i < len(code_ids):
            candidate_codes = list(cluster_codes_resolver(code_ids[i]) or [])
        elif open_code_resolver and i < len(code_ids):
            resolved = open_code_resolver(code_ids[i])
            if resolved:
                candidate_codes = [resolved]

        g: GroundedExample | None = None
        for code in candidate_codes:
            g = memory.ground_quote(quote, open_code=code)
            if g:
                break
        if not g:
            g = memory.ground_quote(quote)

        if g:
            grounded.append(g)

    item["examples"] = grounded


def ground_enriched_entries(
    entries: Dict[str, Any] | List[Dict[str, Any]],
    memory: SourceMemory,
    *,
    cluster_to_codes: Dict[str, List[str]] | None = None,
    local_id_to_code: Dict[str, str] | None = None,
    raw_cid_fn: Any | None = None,
) -> None:
    """Walk inclusion/exclusion criteria and ground string examples."""

    def _resolve_lc(cid: str) -> str | None:
        if local_id_to_code:
            return local_id_to_code.get(cid)
        return None

    def _cluster_codes(cid: str) -> List[str]:
        if not cluster_to_codes:
            return []
        raw = raw_cid_fn(cid) if raw_cid_fn else cid
        return list(cluster_to_codes.get(str(raw), []))

    iterable = entries.values() if isinstance(entries, dict) else entries
    for entry in iterable:
        if not isinstance(entry, dict):
            continue
        for section in ("inclusion", "exclusion"):
            for item in entry.get(section, []):
                if local_id_to_code:
                    ground_criterion_examples(item, memory, open_code_resolver=_resolve_lc)
                else:
                    ground_criterion_examples(item, memory, cluster_codes_resolver=_cluster_codes)
