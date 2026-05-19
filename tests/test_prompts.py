"""Substring tests for agents.core.prompts (regression guards on prompt copy)."""

from agents.core.prompts import (
    batch_open_coding_prompt,
    batch_validate_open_codes_prompt,
    hierarchy_refine_bucket_prompt,
    high_level_code_generation_prompt,
    open_coding_prompt,
    validate_open_codes_prompt,
)


def test_open_coding_prompt_includes_rq_and_text():
    p = open_coding_prompt("Why burnout?", "I feel exhausted.")
    assert "Why burnout?" in p
    assert "I feel exhausted." in p
    assert "Research Question" in p


def test_open_coding_prompt_includes_validator_feedback():
    p = open_coding_prompt("RQ", "text", validator_feedback="Codes too vague.")
    assert "Codes too vague." in p
    assert "reviewer found issues" in p.lower() or "reviewer" in p.lower()


def test_validate_open_codes_prompt_includes_pass_fail():
    p = validate_open_codes_prompt("RQ", "review body", "- Code: foo")
    assert "PASS" in p
    assert "FAIL" in p
    assert "review body" in p


def test_batch_open_coding_prompt_includes_ids_and_json():
    items = [{"id": "1", "text": "review one"}, {"id": "2", "text": "review two"}]
    p = batch_open_coding_prompt("Why burnout?", items)
    assert "Why burnout?" in p
    assert '"id": "1"' in p
    assert "review two" in p
    assert '"responses"' in p


def test_batch_validate_open_codes_prompt_includes_verdict_json():
    items = [{"id": "3", "text": "t", "generated_codes": '- Code: x\n  Evidence: "q"\n  Note: n'}]
    p = batch_validate_open_codes_prompt("RQ", items)
    assert "RQ" in p
    assert '"verdict"' in p
    assert "generated_codes" in p


def test_high_level_code_generation_prompt_includes_rq():
    p = high_level_code_generation_prompt("- code a", "What drives churn?")
    assert "What drives churn?" in p
    assert "Research Question" in p


def test_hierarchy_refine_bucket_prompt_includes_groups_and_labels():
    p = hierarchy_refine_bucket_prompt(
        "Cluster A",
        "Big bucket",
        "- code1\n- code2",
        "How do users cope?",
        num_groups=4,
    )
    assert "Cluster A" in p
    assert "Big bucket" in p
    assert "exactly 4" in p
    assert "How do users cope?" in p
    assert "code1" in p
