"""Tests for pipeline configuration snapshot."""

from __future__ import annotations

import json
import os

import pytest
from agents.core.pipeline_config import (
    CONFIG_KEYS,
    effective_config,
    format_config_report,
    write_config_snapshot,
)


def test_config_keys_nonempty() -> None:
    assert "GT_CODEBOOK_REVIEW" in CONFIG_KEYS
    assert "RESEARCH_QUESTION" in CONFIG_KEYS


def test_effective_config_includes_review_section(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GT_CODEBOOK_REVIEW", "1")
    monkeypatch.setenv("GT_CODEBOOK_REVIEW_BACKEND", "local")
    cfg = effective_config(where="test")
    assert cfg["codebook_review"]["enabled"] is True
    assert cfg["codebook_review"]["resolved_backend"] == "local"


def test_format_config_report_contains_priority() -> None:
    report = format_config_report(effective_config(where="test"))
    assert "pipeline_config.env" in report
    assert "priority" in report.lower()


def test_write_config_snapshot(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    import agents.core.pipeline_config as pc

    logs = tmp_path / "logs"
    logs.mkdir()
    monkeypatch.setattr(pc, "LOGS_DIR", logs)
    monkeypatch.setattr(pc, "SNAPSHOT_PATH", logs / "pipeline_config.json")
    monkeypatch.setattr(pc, "EFFECTIVE_ENV_PATH", logs / "pipeline_config.effective.env")

    path = write_config_snapshot(where="test")
    assert path.is_file()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["_meta"]["where"] == "test"
    assert os.environ.get("GT_CODEBOOK_REVIEW", "0") in (
        data["codebook_review"]["GT_CODEBOOK_REVIEW"],
        str(data["codebook_review"]["GT_CODEBOOK_REVIEW"]),
    )
