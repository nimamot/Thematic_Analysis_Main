"""Tests for code_evidence.json builder."""

from __future__ import annotations

from agents.core.code_evidence import build_code_evidence
from agents.core.source_memory import Snippet, SourceMemory


def test_build_code_evidence_groups_by_open_code() -> None:
    memory = SourceMemory(
        snippets=[
            Snippet("SNIP-0001", 1, "id-1", "stress", None, "quote a", "note a"),
            Snippet("SNIP-0002", 2, "id-2", "stress", None, "quote b", ""),
            Snippet("SNIP-0003", 3, "id-3", "other", None, "quote c", "note c"),
        ]
    )
    memory._build_indexes()

    payload = build_code_evidence(
        memory,
        slug="default",
        research_question="RQ?",
        open_codes={"stress", "missing"},
    )

    assert payload["version"] == 1
    assert payload["slug"] == "default"
    stress = payload["by_open_code"]["stress"]
    assert stress["quote_count"] == 2
    assert stress["review_count"] == 2
    assert stress["primary"]["quote"] == "quote a"
    assert len(stress["snippets"]) == 2
    assert payload["by_open_code"]["missing"]["primary"] is None
