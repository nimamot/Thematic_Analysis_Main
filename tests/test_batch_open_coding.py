"""Tests for agents.core.batch_open_coding."""

import json
from unittest.mock import patch

from agents.core import batch_open_coding as boc
from agents.core.state import OPEN_CODING_MAX_RETRIES


def _code_json(outputs: dict) -> str:
    return json.dumps(
        {"responses": [{"id": str(k), "output": v} for k, v in outputs.items()]}
    )


def _validate_json(verdicts: dict) -> str:
    return json.dumps(
        {
            "responses": [
                {"id": str(k), "verdict": v, "feedback": "" if v == "PASS" else "fix it"}
                for k, v in verdicts.items()
            ]
        }
    )


@patch.object(boc, "_batch_size", return_value=8)
@patch.object(boc, "_batch_workers", return_value=1)
@patch("agents.core.batch_open_coding.llm_invoke_with_skill")
def test_happy_path_all_pass(mock_invoke, _bw, _bs):
    reviews = [(1, "text one"), (2, "text two"), (3, "text three")]
    mock_invoke.side_effect = [
        _code_json({1: "- Code: a\n  Evidence: \"x\"\n  Note: n", 2: "- Code: b\n  Evidence: \"y\"\n  Note: n", 3: "- Code: c\n  Evidence: \"z\"\n  Note: n"}),
        _validate_json({1: "PASS", 2: "PASS", 3: "PASS"}),
    ]
    out = boc.run_batched_open_coding(reviews, "RQ?")
    assert len(out) == 3
    assert "Code: a" in out[1]
    assert mock_invoke.call_count == 2


@patch.object(boc, "_batch_size", return_value=8)
@patch.object(boc, "_batch_workers", return_value=1)
@patch("agents.core.batch_open_coding.llm_invoke_with_skill")
def test_retry_fail_then_pass(mock_invoke, _bw, _bs):
    reviews = [(1, "text one")]
    mock_invoke.side_effect = [
        _code_json({1: "- Applicability: NONE\n  Reason: r\n  Evidence: \"e\""}),
        _validate_json({1: "FAIL"}),
        _code_json({1: "- Code: fixed\n  Evidence: \"x\"\n  Note: n"}),
        _validate_json({1: "PASS"}),
    ]
    out = boc.run_batched_open_coding(reviews, "RQ?")
    assert "Code: fixed" in out[1]
    assert mock_invoke.call_count == 4


@patch.object(boc, "_batch_size", return_value=8)
@patch.object(boc, "_batch_workers", return_value=1)
@patch("agents.core.batch_open_coding.llm_invoke_with_skill")
def test_max_retries_keeps_last_codes(mock_invoke, _bw, _bs):
    reviews = [(1, "text one")]
    last_codes = "- Code: last\n  Evidence: \"x\"\n  Note: n"
    side_effects = []
    for attempt in range(OPEN_CODING_MAX_RETRIES + 1):
        side_effects.append(_code_json({1: f"- Code: try{attempt}\n  Evidence: \"x\"\n  Note: n"}))
        side_effects.append(_validate_json({1: "FAIL"}))
    side_effects[-2] = _code_json({1: last_codes})
    mock_invoke.side_effect = side_effects
    out = boc.run_batched_open_coding(reviews, "RQ?")
    assert "Code: last" in out[1]


@patch.object(boc, "_batch_size", return_value=1)
@patch.object(boc, "_batch_workers", return_value=2)
@patch("agents.core.batch_open_coding.llm_invoke_with_skill")
def test_parallel_batches(mock_invoke, _bw, _bs):
    reviews = [(1, "a"), (2, "b")]
    mock_invoke.side_effect = [
        _code_json({1: "- Code: a\n  Evidence: \"x\"\n  Note: n"}),
        _code_json({2: "- Code: b\n  Evidence: \"y\"\n  Note: n"}),
        _validate_json({1: "PASS"}),
        _validate_json({2: "PASS"}),
    ]
    out = boc.run_batched_open_coding(reviews, "RQ?")
    assert len(out) == 2
    assert mock_invoke.call_count == 4


@patch.object(boc, "_batch_size", return_value=8)
@patch.object(boc, "_batch_workers", return_value=1)
@patch("agents.core.batch_open_coding.open_coding")
@patch("agents.core.batch_open_coding.validate_open_codes")
@patch("agents.core.batch_open_coding.llm_invoke_with_skill")
def test_parse_failure_falls_back_to_single_review(
    mock_invoke, mock_validate, mock_open, _bw, _bs
):
    reviews = [(1, "text one")]
    # Two failed batch invokes per phase (retry + fallback) for code and validate.
    mock_invoke.side_effect = ["not json at all"] * 4
    mock_open.invoke.return_value = "- Code: fallback\n  Evidence: \"x\"\n  Note: n"
    mock_validate.invoke.return_value = "PASS"

    out = boc.run_batched_open_coding(reviews, "RQ?")
    assert "Code: fallback" in out[1]
    mock_open.invoke.assert_called_once()
    mock_validate.invoke.assert_called_once()


def test_chunk():
    items = [(i, f"t{i}") for i in range(1, 6)]
    batches = boc._chunk(items, 2)
    assert len(batches) == 3
    assert batches[0] == [(1, "t1"), (2, "t2")]
