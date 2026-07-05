"""Apply human codebook review edits to local pipeline artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

from .codebook_schema import EnrichedCodebook


class CodebookReviewError(ValueError):
    """Invalid review payload or inconsistent operations."""


@dataclass
class AppliedReviewResult:
    codebook: Dict[str, str]
    cluster_to_codes: Dict[str, List[str]]
    codebook_confidence: Dict[str, Dict[str, Any]]
    provenance: Dict[str, Any]
    orphaned_codes: List[str] = field(default_factory=list)
    skip_refine_cluster_ids: Set[str] = field(default_factory=set)


def _cluster_sort_key(cid: str) -> Tuple[int, str]:
    try:
        return (0, f"{int(cid):010d}")
    except ValueError:
        return (1, cid)


def _sorted_cluster_ids(ids: List[str]) -> List[str]:
    return sorted(ids, key=_cluster_sort_key)


def _apply_merge(cluster_to_codes: Dict[str, List[str]], op: Dict[str, Any]) -> None:
    from_ids = [str(x) for x in op.get("from_cluster_ids") or []]
    target = str(op.get("target_cluster_id") or "")
    if not from_ids or not target:
        raise CodebookReviewError("merge operation requires from_cluster_ids and target_cluster_id")
    merged_codes: List[str] = []
    for cid in from_ids:
        merged_codes.extend(cluster_to_codes.pop(cid, []))
    if target in cluster_to_codes:
        merged_codes = cluster_to_codes[target] + merged_codes
    cluster_to_codes[target] = merged_codes


def _apply_split(cluster_to_codes: Dict[str, List[str]], op: Dict[str, Any]) -> None:
    from_cid = str(op.get("from_cluster_id") or "")
    splits = op.get("splits") or []
    if not from_cid or not splits:
        raise CodebookReviewError("split operation requires from_cluster_id and splits")
    original = list(cluster_to_codes.get(from_cid, []))
    assigned: Set[str] = set()
    for part in splits:
        new_id = str(part.get("new_cluster_id") or "")
        code_ids = [str(c) for c in (part.get("code_ids") or [])]
        if not new_id:
            raise CodebookReviewError("split part missing new_cluster_id")
        for c in code_ids:
            if c not in original:
                raise CodebookReviewError(f"split code {c!r} not in cluster {from_cid}")
            if c in assigned:
                raise CodebookReviewError(f"code {c!r} assigned to multiple split groups")
            assigned.add(c)
        cluster_to_codes[new_id] = code_ids
    cluster_to_codes.pop(from_cid, None)
    unassigned = [c for c in original if c not in assigned]
    if unassigned:
        raise CodebookReviewError(
            f"split of cluster {from_cid} left codes unassigned: {unassigned[:5]}"
        )


def _operation_already_applied(cluster_to_codes: Dict[str, List[str]], op: Dict[str, Any]) -> bool:
    """True when v2.cluster_to_codes already reflects this operation (viewer export)."""
    op_type = op.get("type")
    if op_type == "split":
        from_cid = str(op.get("from_cluster_id") or "")
        if from_cid in cluster_to_codes:
            return False
        new_ids = [str(p.get("new_cluster_id") or "") for p in op.get("splits") or []]
        new_ids = [nid for nid in new_ids if nid]
        return bool(new_ids) and all(nid in cluster_to_codes for nid in new_ids)
    if op_type == "merge":
        from_ids = [str(x) for x in op.get("from_cluster_ids") or []]
        target = str(op.get("target_cluster_id") or "")
        if not target or target not in cluster_to_codes:
            return False
        others = [fid for fid in from_ids if fid and fid != target]
        return not any(fid in cluster_to_codes for fid in others)
    if op_type == "drop":
        cid = str(op.get("cluster_id") or "")
        return bool(cid) and cid not in cluster_to_codes
    return False


def _apply_drop(
    cluster_to_codes: Dict[str, List[str]], op: Dict[str, Any], orphaned: List[str]
) -> None:
    cid = str(op.get("cluster_id") or "")
    if not cid:
        raise CodebookReviewError("drop operation requires cluster_id")
    codes = cluster_to_codes.pop(cid, [])
    if op.get("code_disposition", "delete") == "orphan":
        orphaned.extend(codes)


def _validate_cluster_codes(cluster_to_codes: Dict[str, List[str]]) -> None:
    seen: Dict[str, str] = {}
    for cid, codes in cluster_to_codes.items():
        for c in codes:
            if c in seen:
                raise CodebookReviewError(f"code {c!r} appears in clusters {seen[c]!r} and {cid!r}")
            seen[c] = cid


def apply_codebook_review(
    v2: EnrichedCodebook,
    clustered_codes: Dict[str, Any],
    *,
    review_id: Optional[str] = None,
    approved_at: Optional[str] = None,
) -> AppliedReviewResult:
    """Apply human review v2 payload; returns flat artifacts + provenance."""
    cluster_to_codes: Dict[str, List[str]] = {
        str(k): list(v) for k, v in (v2.get("cluster_to_codes") or {}).items()
    }
    if not cluster_to_codes:
        base = clustered_codes.get("cluster_to_codes") or {}
        cluster_to_codes = {str(k): list(v) for k, v in base.items()}

    orphaned: List[str] = []
    for op in v2.get("operations") or []:
        if _operation_already_applied(cluster_to_codes, op):
            continue
        op_type = op.get("type")
        if op_type == "merge":
            _apply_merge(cluster_to_codes, op)
        elif op_type == "split":
            _apply_split(cluster_to_codes, op)
        elif op_type == "drop":
            _apply_drop(cluster_to_codes, op, orphaned)
        else:
            raise CodebookReviewError(f"unknown operation type: {op_type!r}")

    clusters = v2.get("clusters") or {}
    for cid in list(cluster_to_codes.keys()):
        entry = clusters.get(str(cid), {})
        if isinstance(entry, dict) and entry.get("status") == "drop":
            orphaned.extend(cluster_to_codes.pop(cid, []))

    cluster_to_codes = {cid: codes for cid, codes in cluster_to_codes.items() if codes}
    _validate_cluster_codes(cluster_to_codes)

    old_ids = _sorted_cluster_ids(list(cluster_to_codes.keys()))
    rekeyed: Dict[str, List[str]] = {}
    old_to_new: Dict[str, str] = {}
    for i, old_cid in enumerate(old_ids):
        new_cid = str(i)
        old_to_new[old_cid] = new_cid
        rekeyed[new_cid] = cluster_to_codes[old_cid]

    codebook: Dict[str, str] = {}
    codebook_confidence: Dict[str, Dict[str, Any]] = {}
    provenance_entries: List[Dict[str, Any]] = []
    skip_refine: Set[str] = set()
    approved_ts = approved_at or datetime.now(timezone.utc).isoformat()

    for new_cid, old_cid in zip(rekeyed.keys(), old_ids):
        entry = clusters.get(old_cid, {})
        if not isinstance(entry, dict):
            entry = {}
        label = str(entry.get("label") or f"Cluster {new_cid}")
        description = str(entry.get("description") or "")
        confidence = int(entry.get("confidence", 1) or 1)
        source = str(entry.get("source") or "llm")
        needs_more = bool(entry.get("needs_more_evidence", False))

        codebook[new_cid] = label
        codebook_confidence[new_cid] = {
            "label": label,
            "confidence": confidence,
            "rationale": description,
            "source": source,
            "needs_more_evidence": needs_more,
        }
        if needs_more:
            skip_refine.add(new_cid)
        provenance_entries.append(
            {
                "cluster_id": new_cid,
                "source_cluster_id": old_cid,
                "label": label,
                "source": source,
                "needs_more_evidence": needs_more,
                "approved_at": approved_ts,
                "review_id": review_id,
            }
        )

    provenance = {
        "review_id": review_id,
        "approved_at": approved_ts,
        "entries": provenance_entries,
        "orphaned_codes": orphaned,
        "skip_refine_cluster_ids": sorted(skip_refine),
    }

    return AppliedReviewResult(
        codebook=codebook,
        cluster_to_codes=rekeyed,
        codebook_confidence=codebook_confidence,
        provenance=provenance,
        orphaned_codes=orphaned,
        skip_refine_cluster_ids=skip_refine,
    )


def materialize_clustered_output(
    result: AppliedReviewResult,
    clustered_codes: Dict[str, Any],
) -> Dict[str, Any]:
    """Build updated gt_clustered_codes.json content."""
    all_codes = clustered_codes.get("all_codes") or []
    codes_per_review = clustered_codes.get("codes_per_review") or []
    code_to_idx: Dict[str, int] = {}
    for i, cids in enumerate(result.cluster_to_codes.values()):
        for c in cids:
            code_to_idx[c] = i
    labels = [code_to_idx.get(c, 0) for c in all_codes]
    return {
        "all_codes": all_codes,
        "labels": labels,
        "k": len(result.cluster_to_codes),
        "cluster_to_codes": result.cluster_to_codes,
        "codes_per_review": codes_per_review,
    }
