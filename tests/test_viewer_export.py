"""Tests for viewer-data export helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from agents.core import viewer_export as ve
from agents.core.paths import (
    CLUSTERED_CODES_PATH,
    CODEBOOK_CONFIDENCE_PATH,
    CODEBOOK_PATH,
    GLOBAL_GRAPH_PATH,
    RESEARCH_REPORT_PATH,
)


@pytest.fixture
def viewer_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("VIEWER_DATA_DIR", str(tmp_path))
    return ve.ensure_viewer_scaffold(tmp_path)


@pytest.fixture
def pipeline_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    data_dir = tmp_path / "outputs" / "data"
    data_dir.mkdir(parents=True)
    monkeypatch.setattr(ve, "CODEBOOK_PATH", data_dir / "codebook.json")
    monkeypatch.setattr(ve, "CLUSTERED_CODES_PATH", data_dir / "gt_clustered_codes.json")
    monkeypatch.setattr(ve, "CODEBOOK_CONFIDENCE_PATH", data_dir / "codebook_confidence.json")
    monkeypatch.setattr(ve, "GLOBAL_GRAPH_PATH", data_dir / "gt_global_graph.json")
    monkeypatch.setattr(ve, "RESEARCH_REPORT_PATH", data_dir / "research_report.md")
    monkeypatch.setattr(ve, "OPEN_CODES_MARKDOWN_PATH", data_dir / "gt_open_codes_all_reviews.md")
    monkeypatch.setattr(ve, "COOCCURRENCE_PATH", data_dir / "gt_cooccurrence.json")

    (data_dir / "codebook.json").write_text(
        json.dumps({"codebook": {"1": "Theme A"}, "cluster_to_codes": {"1": ["c1"]}}),
        encoding="utf-8",
    )
    (data_dir / "gt_clustered_codes.json").write_text(
        json.dumps({"cluster_to_codes": {"1": ["c1"]}}),
        encoding="utf-8",
    )
    (data_dir / "codebook_confidence.json").write_text("{}", encoding="utf-8")
    (data_dir / "gt_global_graph.json").write_text(
        json.dumps({"tree": {"name": "What themes emerge?", "type": "root", "children": []}}),
        encoding="utf-8",
    )
    (data_dir / "research_report.md").write_text(
        "## Research question\n\nWhat themes emerge?\n",
        encoding="utf-8",
    )
    return data_dir


def test_ensure_viewer_scaffold_creates_empty_manifest(viewer_root: Path) -> None:
    manifest = json.loads((viewer_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest == {"projects": [], "codebook_reviews": []}


def test_export_codebook_review_updates_manifest(
    viewer_root: Path, pipeline_outputs: Path
) -> None:
    review_dir = ve.export_codebook_review("my-study", "RQ?", root=viewer_root)
    assert review_dir.is_dir()
    assert (review_dir / "codebook.json").is_file()
    assert (review_dir / "gt_clustered_codes.json").is_file()
    manifest = json.loads((viewer_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["codebook_reviews"] == ["my-study"]


def test_export_project_updates_manifest(viewer_root: Path, pipeline_outputs: Path) -> None:
    project_dir = ve.export_project("my-study", root=viewer_root)
    assert (project_dir / "gt_global_graph.json").is_file()
    assert (project_dir / "research_report.md").is_file()
    meta = json.loads((project_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["research_question"] == "What themes emerge?"
    manifest = json.loads((viewer_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["projects"] == ["my-study"]


def test_review_backend_defaults_local_without_supabase(monkeypatch: pytest.MonkeyPatch) -> None:
    from agents.core.codebook_review import review_backend

    monkeypatch.delenv("GT_CODEBOOK_REVIEW_BACKEND", raising=False)
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
    assert review_backend() == "local"


def test_review_backend_prefers_supabase_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    from agents.core.codebook_review import review_backend

    monkeypatch.delenv("GT_CODEBOOK_REVIEW_BACKEND", raising=False)
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "secret")
    assert review_backend() == "supabase"


def test_review_backend_explicit_local_overrides_supabase(monkeypatch: pytest.MonkeyPatch) -> None:
    from agents.core.codebook_review import review_backend

    monkeypatch.setenv("GT_CODEBOOK_REVIEW_BACKEND", "local")
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "secret")
    assert review_backend() == "local"
