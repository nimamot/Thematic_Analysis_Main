"""Tests for codebook review edit application."""

import pytest
from agents.core.codebook_edits import CodebookReviewError, apply_codebook_review


def _base_clustered():
    return {
        "all_codes": ["a", "b", "c", "d"],
        "cluster_to_codes": {"0": ["a", "b"], "1": ["c", "d"]},
        "codes_per_review": [[1, ["a", "b"]], [2, ["c", "d"]]],
    }


def test_rename_only_updates_provenance():
    v2 = {
        "version": 1,
        "clusters": {
            "0": {
                "label": "Renamed theme",
                "description": "updated rationale",
                "confidence": 4,
                "source": "llm+edited",
                "needs_more_evidence": False,
                "status": "keep",
            },
            "1": {
                "label": "Cluster 1",
                "description": "",
                "confidence": 3,
                "source": "llm",
                "needs_more_evidence": False,
                "status": "keep",
            },
        },
        "operations": [],
        "cluster_to_codes": {"0": ["a", "b"], "1": ["c", "d"]},
    }
    result = apply_codebook_review(v2, _base_clustered(), review_id="r1")
    assert result.codebook["0"] == "Renamed theme"
    assert result.provenance["entries"][0]["source"] == "llm+edited"
    assert result.codebook_confidence["0"]["rationale"] == "updated rationale"


def test_merge_clusters_combines_codes():
    v2 = {
        "version": 1,
        "clusters": {
            "0": {"label": "Merged", "source": "human", "status": "keep"},
            "1": {"label": "Merged", "source": "human", "status": "keep"},
        },
        "operations": [
            {
                "type": "merge",
                "from_cluster_ids": ["0", "1"],
                "target_cluster_id": "0",
                "label": "Merged",
            }
        ],
        "cluster_to_codes": {"0": ["a", "b"], "1": ["c", "d"]},
    }
    result = apply_codebook_review(v2, _base_clustered())
    assert len(result.cluster_to_codes) == 1
    assert set(result.cluster_to_codes["0"]) == {"a", "b", "c", "d"}


def test_split_cluster_partitions_codes():
    v2 = {
        "version": 1,
        "clusters": {
            "0a": {"label": "Part A", "source": "human", "status": "keep"},
            "0b": {"label": "Part B", "source": "human", "status": "keep"},
            "1": {"label": "Other", "source": "llm", "status": "keep"},
        },
        "operations": [
            {
                "type": "split",
                "from_cluster_id": "0",
                "splits": [
                    {"new_cluster_id": "0a", "label": "Part A", "code_ids": ["a"]},
                    {"new_cluster_id": "0b", "label": "Part B", "code_ids": ["b"]},
                ],
            }
        ],
        "cluster_to_codes": {"0": ["a", "b"], "1": ["c", "d"]},
    }
    result = apply_codebook_review(v2, _base_clustered())
    assert len(result.cluster_to_codes) == 3
    labels = list(result.cluster_to_codes.values())
    assert ["a"] in labels
    assert ["b"] in labels


def test_drop_cluster_orphans_codes():
    v2 = {
        "version": 1,
        "clusters": {
            "0": {"label": "Keep", "source": "llm", "status": "keep"},
            "1": {"label": "Gone", "source": "llm", "status": "drop"},
        },
        "operations": [
            {"type": "drop", "cluster_id": "1", "code_disposition": "orphan"},
        ],
        "cluster_to_codes": {"0": ["a", "b"], "1": ["c", "d"]},
    }
    result = apply_codebook_review(v2, _base_clustered())
    assert len(result.cluster_to_codes) == 1
    assert result.orphaned_codes == ["c", "d"]


def test_needs_more_evidence_sets_skip_refine():
    v2 = {
        "version": 1,
        "clusters": {
            "0": {
                "label": "Uncertain",
                "source": "llm",
                "needs_more_evidence": True,
                "status": "keep",
            },
            "1": {"label": "Clear", "source": "llm", "status": "keep"},
        },
        "operations": [],
        "cluster_to_codes": {"0": ["a", "b"], "1": ["c", "d"]},
    }
    result = apply_codebook_review(v2, _base_clustered())
    assert "0" in result.skip_refine_cluster_ids


def test_split_skipped_when_cluster_to_codes_already_reflects_viewer_export():
    """Viewer v2 exports final membership plus an operations log — do not replay splits."""
    v2 = {
        "version": 1,
        "clusters": {
            "2a": {"label": "Part A", "source": "human", "status": "keep"},
            "2b": {"label": "Part B", "source": "human", "status": "keep"},
            "1": {"label": "Other", "source": "llm", "status": "keep"},
        },
        "operations": [
            {
                "type": "split",
                "from_cluster_id": "0",
                "splits": [
                    {"new_cluster_id": "2a", "label": "Part A", "code_ids": ["a"]},
                    {"new_cluster_id": "2b", "label": "Part B", "code_ids": ["b"]},
                ],
            }
        ],
        "cluster_to_codes": {"2a": ["a"], "2b": ["b"], "1": ["c", "d"]},
    }
    result = apply_codebook_review(v2, _base_clustered())
    values = [set(v) for v in result.cluster_to_codes.values()]
    assert {"a"} in values
    assert {"b"} in values
    assert {"c", "d"} in values


def test_invalid_split_raises():
    v2 = {
        "version": 1,
        "clusters": {},
        "operations": [
            {
                "type": "split",
                "from_cluster_id": "0",
                "splits": [{"new_cluster_id": "0a", "code_ids": ["missing"]}],
            }
        ],
        "cluster_to_codes": {"0": ["a", "b"]},
    }
    with pytest.raises(CodebookReviewError):
        apply_codebook_review(v2, _base_clustered())
