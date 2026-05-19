"""Batched open coding + validation for the CLI (multi-review prompts)."""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Tuple

from .prompts import batch_open_coding_prompt, batch_validate_open_codes_prompt
from .skills import llm_invoke_with_skill
from .state import OPEN_CODING_MAX_RETRIES, _parse_validation_output
from .tools import llm, open_coding, validate_open_codes
from .utils import clean_and_parse_json, log_step, remove_think_tags


def _context_length() -> int:
    raw = os.environ.get("GT_CONTEXT_LENGTH", "8000").strip()
    try:
        return max(2048, int(raw))
    except ValueError:
        return 8000


def _batch_max_tokens() -> int:
    """Completion budget for batched calls; must fit inside GT_CONTEXT_LENGTH with long inputs."""
    ctx = _context_length()
    raw = os.environ.get("GT_BATCH_MAX_TOKENS", "2048").strip()
    try:
        cap = int(raw)
    except ValueError:
        cap = 2048
    # Reserve headroom for prompt + batched review text (typical batch input ~2–6k on 8k ctx).
    return max(512, min(cap, ctx // 2))


def _is_context_length_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "context length" in msg or "maximum context" in msg


def _invoke_with_context_split(
    batch: List[Tuple[int, str]],
    invoke_fn: Callable[[List[Tuple[int, str]]], Dict],
) -> Dict:
    try:
        return invoke_fn(batch)
    except Exception as e:
        if not _is_context_length_error(e) or len(batch) <= 1:
            raise
        mid = max(1, len(batch) // 2)
        ids = [r for r, _ in batch]
        log_step(
            "BATCH_CONTEXT_SPLIT",
            f"ids={ids} -> {ids[:mid]} + {ids[mid:]} ({e})",
        )
        left = _invoke_with_context_split(batch[:mid], invoke_fn)
        right = _invoke_with_context_split(batch[mid:], invoke_fn)
        left.update(right)
        return left


# The batch size is the number of reviews to process in parallel. Should be tuned according to the approximate length of the reviews (given the current context window of 8000).
def _batch_size() -> int:
    raw = os.environ.get("GT_OPEN_CODING_BATCH_SIZE", "4").strip()
    try:
        n = int(raw)
        return max(1, min(32, n))
    except ValueError:
        return 8


def _batch_workers() -> int:
    raw = os.environ.get("GT_OPEN_CODING_BATCH_WORKERS", "4").strip()
    try:
        n = int(raw)
        return max(1, min(16, n))
    except ValueError:
        return 4


def _chunk(items: List[Tuple[int, str]], batch_size: int) -> List[List[Tuple[int, str]]]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def _id_str(review_id: int) -> str:
    return str(review_id)


def _parse_code_batch(raw: str, expected_ids: List[int]) -> Dict[int, str]:
    parsed = clean_and_parse_json(remove_think_tags(raw))
    responses = parsed.get("responses", [])
    if not isinstance(responses, list):
        raise ValueError("Batch code response missing 'responses' list.")
    out: Dict[int, str] = {}
    for entry in responses:
        if not isinstance(entry, dict):
            continue
        rid = entry.get("id")
        if rid is None:
            continue
        try:
            key = int(rid)
        except (TypeError, ValueError):
            key = int(str(rid).strip())
        output = entry.get("output", "")
        out[key] = (output if output is not None else "").strip()
    missing = [i for i in expected_ids if i not in out]
    if missing:
        raise ValueError(f"Batch code response missing ids: {missing}")
    return out


def _parse_validate_batch(raw: str, expected_ids: List[int]) -> Dict[int, Tuple[str, str]]:
    parsed = clean_and_parse_json(remove_think_tags(raw))
    responses = parsed.get("responses", [])
    if not isinstance(responses, list):
        raise ValueError("Batch validate response missing 'responses' list.")
    out: Dict[int, Tuple[str, str]] = {}
    for entry in responses:
        if not isinstance(entry, dict):
            continue
        rid = entry.get("id")
        if rid is None:
            continue
        try:
            key = int(rid)
        except (TypeError, ValueError):
            key = int(str(rid).strip())
        verdict_raw = (entry.get("verdict") or "").strip()
        if verdict_raw.upper() in ("PASS", "FAIL"):
            verdict = verdict_raw.upper()
        else:
            verdict, _ = _parse_validation_output(verdict_raw or str(entry.get("feedback", "")))
        feedback = (entry.get("feedback") or "").strip()
        if verdict == "FAIL" and not feedback:
            feedback = verdict_raw or "Validation failed."
        out[key] = (verdict, feedback)
    missing = [i for i in expected_ids if i not in out]
    if missing:
        raise ValueError(f"Batch validate response missing ids: {missing}")
    return out


def _build_code_items(
    batch: List[Tuple[int, str]],
    feedback_map: Dict[int, str],
) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for rid, text in batch:
        item: Dict[str, Any] = {"id": _id_str(rid), "text": text}
        fb = feedback_map.get(rid)
        if fb:
            item["validator_feedback"] = fb
        items.append(item)
    return items


def _build_validate_items(
    batch: List[Tuple[int, str]],
    codes_by_id: Dict[int, str],
) -> List[Dict[str, Any]]:
    return [
        {
            "id": _id_str(rid),
            "text": text,
            "generated_codes": codes_by_id.get(rid, ""),
        }
        for rid, text in batch
    ]


def _invoke_batch_code(
    batch: List[Tuple[int, str]],
    research_question: str,
    feedback_map: Dict[int, str],
) -> Dict[int, str]:
    def _do(one_batch: List[Tuple[int, str]]) -> Dict[int, str]:
        ids = [rid for rid, _ in one_batch]
        items = _build_code_items(one_batch, feedback_map)
        prompt = batch_open_coding_prompt(research_question, items)
        raw = llm_invoke_with_skill(
            llm,
            "batch_open_coding",
            prompt,
            max_tokens=_batch_max_tokens(),
            batch_ids=",".join(_id_str(i) for i in ids),
        )
        result = _parse_code_batch(raw, ids)
        log_step("BATCH_OPEN_CODING", f"ids={ids}")
        return result

    return _invoke_with_context_split(batch, _do)


def _invoke_batch_validate(
    batch: List[Tuple[int, str]],
    research_question: str,
    codes_by_id: Dict[int, str],
) -> Dict[int, Tuple[str, str]]:
    def _do(one_batch: List[Tuple[int, str]]) -> Dict[int, Tuple[str, str]]:
        ids = [rid for rid, _ in one_batch]
        items = _build_validate_items(one_batch, codes_by_id)
        prompt = batch_validate_open_codes_prompt(research_question, items)
        raw = llm_invoke_with_skill(
            llm,
            "batch_validate_open_codes",
            prompt,
            max_tokens=_batch_max_tokens(),
            batch_ids=",".join(_id_str(i) for i in ids),
        )
        result = _parse_validate_batch(raw, ids)
        log_step(
            "BATCH_VALIDATE_OPEN_CODES",
            f"ids={ids} verdicts={{{', '.join(f'{i}:{v}' for i, (v, _) in result.items())}}}",
        )
        return result

    return _invoke_with_context_split(batch, _do)


def _fallback_code_batch(
    batch: List[Tuple[int, str]],
    research_question: str,
    feedback_map: Dict[int, str],
) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for rid, text in batch:
        fb = feedback_map.get(rid)
        raw = open_coding.invoke(
            {"text": text, "research_question": research_question, "validator_feedback": fb}
        )
        out[rid] = remove_think_tags(raw)
    log_step("BATCH_OPEN_CODING_FALLBACK", f"single-review fallback ids={[r for r, _ in batch]}")
    return out


def _fallback_validate_batch(
    batch: List[Tuple[int, str]],
    research_question: str,
    codes_by_id: Dict[int, str],
) -> Dict[int, Tuple[str, str]]:
    out: Dict[int, Tuple[str, str]] = {}
    for rid, text in batch:
        raw = validate_open_codes.invoke(
            {
                "text": text,
                "generated_codes": codes_by_id.get(rid, ""),
                "research_question": research_question,
            }
        )
        verdict, feedback = _parse_validation_output(remove_think_tags(raw))
        out[rid] = (verdict, feedback)
    log_step("BATCH_VALIDATE_FALLBACK", f"single-review fallback ids={[r for r, _ in batch]}")
    return out


def _run_with_retry(
    batch: List[Tuple[int, str]],
    invoke_fn: Callable[[List[Tuple[int, str]]], Dict],
    fallback_fn: Callable[[List[Tuple[int, str]]], Dict],
) -> Dict:
    try:
        return invoke_fn(batch)
    except (ValueError, json.JSONDecodeError) as e:
        log_step("BATCH_PARSE_RETRY", f"ids={[r for r, _ in batch]} error={e}")
        try:
            return invoke_fn(batch)
        except (ValueError, json.JSONDecodeError):
            return fallback_fn(batch)


def _run_batches_parallel(
    batches: List[List[Tuple[int, str]]],
    worker_fn: Callable[[List[Tuple[int, str]]], Dict],
    workers: int,
) -> Dict:
    merged: Dict = {}
    if not batches:
        return merged
    if workers <= 1 or len(batches) == 1:
        for batch in batches:
            merged.update(worker_fn(batch))
        return merged
    with ThreadPoolExecutor(max_workers=min(workers, len(batches))) as ex:
        futures = {ex.submit(worker_fn, b): b for b in batches}
        for fut in as_completed(futures):
            merged.update(fut.result())
    return merged


def run_batched_open_coding(
    reviews: List[Tuple[int, str]],
    research_question: str,
) -> Dict[int, str]:
    """
    Run batched open coding then validation with retries for FAIL ids.
    Returns review_id -> final open-coding markdown string.
    """
    if not reviews:
        return {}

    batch_size = _batch_size()
    workers = _batch_workers()
    text_by_id = {rid: text for rid, text in reviews}

    codes_by_id: Dict[int, str] = {}
    retry_count: Dict[int, int] = {rid: 0 for rid, _ in reviews}
    feedback_map: Dict[int, str] = {}
    finalized: Dict[int, str] = {}
    total = len(reviews)

    def _on_finalized(rid: int, output: str) -> None:
        if rid not in finalized:
            finalized[rid] = output
            print(
                f"      [{len(finalized)}/{total}] open coding done: review {rid}",
                flush=True,
            )

    def _code_batches(batches: List[List[Tuple[int, str]]]) -> Dict[int, str]:
        def _one(batch: List[Tuple[int, str]]) -> Dict[int, str]:
            return _run_with_retry(
                batch,
                lambda b: _invoke_batch_code(b, research_question, feedback_map),
                lambda b: _fallback_code_batch(b, research_question, feedback_map),
            )

        return _run_batches_parallel(batches, _one, workers)

    def _validate_batches(batches: List[List[Tuple[int, str]]]) -> Dict[int, Tuple[str, str]]:
        def _one(batch: List[Tuple[int, str]]) -> Dict[int, Tuple[str, str]]:
            return _run_with_retry(
                batch,
                lambda b: _invoke_batch_validate(b, research_question, codes_by_id),
                lambda b: _fallback_validate_batch(b, research_question, codes_by_id),
            )

        return _run_batches_parallel(batches, _one, workers)

    # Phase 1: initial coding for all reviews
    all_batches = _chunk(reviews, batch_size)
    codes_by_id.update(_code_batches(all_batches))

    # Phase 2: validate all
    verdicts = _validate_batches(all_batches)
    pending_fail: List[int] = []

    for rid, _ in reviews:
        verdict, fb = verdicts.get(rid, ("FAIL", "Missing validation result."))
        if verdict == "PASS":
            _on_finalized(rid, codes_by_id[rid])
        elif retry_count[rid] < OPEN_CODING_MAX_RETRIES:
            pending_fail.append(rid)
            feedback_map[rid] = fb
        else:
            _on_finalized(rid, codes_by_id[rid])

    # Phase 3: retry loop for failures
    while pending_fail:
        fail_reviews = [(rid, text_by_id[rid]) for rid in pending_fail]
        fail_batches = _chunk(fail_reviews, batch_size)

        for rid in pending_fail:
            retry_count[rid] += 1

        recoded = _code_batches(fail_batches)
        codes_by_id.update(recoded)

        re_verdicts = _validate_batches(fail_batches)
        next_fail: List[int] = []

        for rid in pending_fail:
            verdict, fb = re_verdicts.get(rid, ("FAIL", "Missing validation result."))
            if verdict == "PASS":
                _on_finalized(rid, codes_by_id[rid])
            elif retry_count[rid] < OPEN_CODING_MAX_RETRIES:
                next_fail.append(rid)
                feedback_map[rid] = fb
            else:
                _on_finalized(rid, codes_by_id[rid])

        pending_fail = next_fail

    # Ensure every review has an entry
    for rid, _ in reviews:
        if rid not in finalized:
            finalized[rid] = codes_by_id.get(rid, "")

    return finalized
