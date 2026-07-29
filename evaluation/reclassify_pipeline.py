"""Codebook reclassification reliability pipeline -- standalone from
pipeline.py/prompts.py (original CRB pipeline untouched).

For each main-pipeline run: take the FULL meta-theme codebook (LABEL +
DEFINITION + INCLUSION + EXCLUSION criteria text, no embedded examples) and
have a cross-model classifier independently re-assign every raw review quote
in that run to one of the meta-themes, based only on the written criteria.
Compare against the ground truth (which meta-theme the quote's open code was
actually clustered under in the original pipeline) to get an agreement rate.

Quotes are classified in BATCHES (same codebook context reused across many
quotes per call) since a single run can have 600-800 raw quotes.
"""
from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from .llm_client import call_llm, parse_json_response
from .prompts_reclassify import reclassify_system, reclassify_user

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    model: str
    base_url: str


def _parse_review_evidence(md_path: Path) -> dict[int, list[dict]]:
    """Parse gt_open_codes_all_reviews.md into {review_idx: [{code, evidence}]}."""
    if not md_path.is_file():
        return {}
    text = md_path.read_text(encoding="utf-8")
    blocks = re.split(r"## Review (\d+)", text)
    parsed: dict[int, list[dict]] = {}
    for i in range(1, len(blocks), 2):
        rev_idx = int(blocks[i])
        content = blocks[i + 1]
        items = []
        for m in re.finditer(r'- Code: (.+?)\n\s+Evidence: "(.+?)"', content, re.DOTALL):
            items.append({"code": m.group(1).strip(), "evidence": m.group(2).strip()})
        parsed[rev_idx] = items
    return parsed


# ── Ground truth + codebook construction ──────────────────────────────────

def build_ground_truth(run_dir: Path, enriched_filename: str = "gt_meta_themes_enriched.json") -> list[dict]:
    """
    Returns one row per (review, open_code, quote) triple:
    {review_idx, open_code, quote, true_meta_theme_idx, true_meta_theme_label}

    Mapping path: open_code -> cluster (via cluster_to_codes) -> meta_theme
    (via each meta-theme's cluster_ids). Verified this is a clean partition
    (every cluster belongs to exactly one meta-theme, no orphans, no
    duplicates) -- see the verification run against ai_healthcare_qwen3.6_run1
    before this was built.

    enriched_filename lets this read gt_meta_themes_enriched_full.json (the
    full-evidence regeneration) instead of the original, without touching the
    original file's own reclassify results.
    """
    data_dir = run_dir / "data"
    cl_data = json.loads((data_dir / "gt_clustered_codes.json").read_text(encoding="utf-8"))
    cluster_to_codes: dict[str, list[str]] = {str(k): v for k, v in cl_data.get("cluster_to_codes", {}).items()}
    mt_data = json.loads((data_dir / enriched_filename).read_text(encoding="utf-8"))
    meta_themes = mt_data.get("meta_themes_enriched", [])

    open_code_to_cluster: dict[str, str] = {}
    for cid, codes in cluster_to_codes.items():
        for code in codes:
            open_code_to_cluster[code] = cid

    cluster_to_meta_idx: dict[str, int] = {}
    for mt_idx, mt in enumerate(meta_themes):
        for cid in mt.get("cluster_ids", []):
            cluster_to_meta_idx[str(cid)] = mt_idx

    review_evidence = _parse_review_evidence(data_dir / "gt_open_codes_all_reviews.md")

    rows: list[dict] = []
    for rev_idx, entries in review_evidence.items():
        # Sibling quotes: other evidence extracted from the SAME original
        # review (different open code), used as approximate surrounding
        # context since the original raw review text isn't reliably
        # recoverable (see investigation: data/sampled/*.csv was overwritten
        # after these runs were generated, no backup/git history exists).
        # These siblings ARE genuinely grounded -- they're the same
        # already-extracted-and-verified evidence used elsewhere in the
        # pipeline, just concatenated rather than a single isolated fragment.
        all_quotes_this_review = [e["evidence"] for e in entries if e.get("evidence")]

        for entry in entries:
            code = entry["code"]
            cid = open_code_to_cluster.get(code)
            if cid is None:
                continue  # open code not in any cluster (shouldn't normally happen)
            mt_idx = cluster_to_meta_idx.get(cid)
            if mt_idx is None:
                continue
            siblings = [q for q in all_quotes_this_review if q != entry["evidence"]]
            rows.append({
                "review_idx": rev_idx,
                "open_code": code,
                "quote": entry["evidence"],
                "sibling_quotes": siblings,
                "true_meta_theme_idx": mt_idx,
                "true_meta_theme_label": meta_themes[mt_idx].get("label", ""),
            })
    return rows


DETAIL_LEVELS = ("theme_only", "theme_def", "full")


def format_codebook(
    run_dir: Path,
    enriched_filename: str = "gt_meta_themes_enriched.json",
    detail_level: str = "full",
) -> tuple[str, dict[int, str]]:
    """
    Returns (formatted_text, letter_to_label_map). Always drops the
    "examples" field embedded in each criterion (per instruction: the
    reclassifier must not see the same example quotes that could leak the
    answer for quotes that happen to be verbatim examples).

    detail_level controls how much of the codebook is shown, for the
    ablation experiment requested by the advisor:
      - "theme_only": LABEL only.
      - "theme_def":  LABEL + DEFINITION.
      - "full":       LABEL + DEFINITION + INCLUSION/EXCLUSION criterion
                       text (the original, only mode before this ablation
                       was added).
    """
    if detail_level not in DETAIL_LEVELS:
        raise ValueError(f"detail_level must be one of {DETAIL_LEVELS}, got {detail_level!r}")

    mt_data = json.loads((run_dir / "data" / enriched_filename).read_text(encoding="utf-8"))
    meta_themes = mt_data.get("meta_themes_enriched", [])

    letters = [chr(ord("A") + i) for i in range(len(meta_themes))]
    letter_map: dict[int, str] = {}
    blocks = []
    for letter, mt in zip(letters, meta_themes):
        label = mt.get("label", "")
        letter_map[letters.index(letter)] = label
        block = [f"[{letter}] LABEL: {label}"]

        if detail_level in ("theme_def", "full"):
            definition = mt.get("definition", "")
            block.append(f"    DEFINITION: {definition}")

        if detail_level == "full":
            inclusion = [c.get("criterion", "") for c in mt.get("inclusion", [])]
            exclusion = [c.get("criterion", "") for c in mt.get("exclusion", [])]
            if inclusion:
                block.append("    INCLUSION CRITERIA:")
                block.extend(f"      - {c}" for c in inclusion)
            if exclusion:
                block.append("    EXCLUSION CRITERIA:")
                block.extend(f"      - {c}" for c in exclusion)

        blocks.append("\n".join(block))

    return "\n\n".join(blocks), {i: mt.get("label", "") for i, mt in enumerate(meta_themes)}


# ── Batch classification ───────────────────────────────────────────────────
# NOTE: sibling-quote context (see build_ground_truth()'s sibling_quotes
# field) was tried and found not to move agreement rates -- turned OFF here
# so classify_batch matches the plain, no-context format that produced the
# existing baseline numbers (results_reclassify_full_qwen/gemma4). The
# sibling_quotes field is still computed/threaded through for possible future
# use, just not rendered into the prompt.

def _format_quote_with_context(idx: int, q: dict) -> str:
    return f'{idx}. "{q["quote"]}"'


def classify_batch(
    codebook_text: str,
    quotes: list[dict],
    model: str,
    base_url: str = "http://localhost:8000/v1",
    n_meta_themes: int = 5,
    max_retries: int = 2,
    detail_level: str = "full",
    include_reasoning: bool = False,
) -> dict[int, str]:
    """
    quotes: list of {quote} dicts, 1-indexed by position in this batch.
    Returns {quote_batch_index (1-based): predicted_letter_or_NONE}.

    detail_level MUST match what format_codebook() actually put into
    codebook_text -- the system/user prompt wording only claims to require
    the pieces (label / label+definition / label+definition+inc-exc) that
    were genuinely provided. Earlier this was hardcoded to always claim
    "definition and inclusion/exclusion criteria" regardless of what was
    actually shown, which for theme_only/theme_def runs caused some models
    (esp. gemma-4) to deterministically refuse to classify and instead ask
    the caller to supply the "missing" criteria -- a self-contradictory
    instruction, not a formatting bug, and unfixable by retrying.

    include_reasoning=False drops the per-item "reasoning" field from the
    requested JSON schema -- for verbose models (e.g. qwen3.5-35b-a3b) whose
    reasoning sentences routinely pushed a 25-quote batch past the output
    token budget and truncated mid-response. reasoning was never persisted
    to the output row regardless (see below), so this loses nothing.
    """
    valid_letters = {chr(ord("A") + i) for i in range(n_meta_themes)} | {"NONE"}
    quotes_text = "\n".join(_format_quote_with_context(i + 1, q) for i, q in enumerate(quotes))
    system = reclassify_system(detail_level, include_reasoning=include_reasoning)
    user = reclassify_user(codebook_text, quotes_text, detail_level, include_reasoning=include_reasoning)

    # Scale the output budget with batch size as a safety margin for larger
    # batches (harmless even though the dominant failure mode above turned
    # out to be the prompt contradiction, not truncation).
    max_out_tokens = min(16000, max(4000, len(quotes) * 250))

    for attempt in range(1 + max_retries):
        try:
            raw = call_llm(system, user, model=model, base_url=base_url,
                            temperature=0.0, max_tokens=max_out_tokens)
            data = parse_json_response(raw)
            classifications = data.get("classifications", [])
            result: dict[int, str] = {}
            for c in classifications:
                idx = c.get("quote_index")
                letter = str(c.get("meta_theme", "")).strip().upper()
                if isinstance(idx, int) and 1 <= idx <= len(quotes) and letter in valid_letters:
                    result[idx] = letter
            if len(result) >= len(quotes) * 0.8:  # accept if we got most of them
                return result
            logger.warning("classify_batch: only %d/%d valid classifications (attempt %d/%d)",
                            len(result), len(quotes), attempt + 1, 1 + max_retries)
        except Exception as exc:
            logger.warning("classify_batch failed (attempt %d/%d): %s", attempt + 1, 1 + max_retries, exc)
    return {}


# ── Run orchestration ──────────────────────────────────────────────────────

def reclassify_run(
    run_dir: Path,
    classifier_cfg: ModelConfig,
    batch_size: int = 25,
    max_workers: int = 8,
    enriched_filename: str = "gt_meta_themes_enriched.json",
    detail_level: str = "full",
    include_reasoning: bool = False,
) -> list[dict]:
    """
    Full reclassification for one run. Returns rows:
    {run_name, review_idx, open_code, quote, true_meta_theme_idx,
     true_meta_theme_label, pred_meta_theme_idx, pred_meta_theme_label,
     pred_letter_raw, agree}

    detail_level: see format_codebook() -- "theme_only" / "theme_def" / "full".
    """
    gt_rows = build_ground_truth(run_dir, enriched_filename)
    codebook_text, letter_to_label = format_codebook(run_dir, enriched_filename, detail_level=detail_level)
    n_meta_themes = len(letter_to_label)
    letters = [chr(ord("A") + i) for i in range(n_meta_themes)]

    batches = [gt_rows[i:i + batch_size] for i in range(0, len(gt_rows), batch_size)]

    def _do_batch(batch: list[dict]) -> list[dict]:
        preds = classify_batch(codebook_text, batch, classifier_cfg.model, classifier_cfg.base_url,
                                n_meta_themes=n_meta_themes, detail_level=detail_level,
                                include_reasoning=include_reasoning)
        out = []
        for i, row in enumerate(batch, start=1):
            letter = preds.get(i, "")
            # status distinguishes "we never got a usable classification for
            # this quote" (call_failed) from "the model classified it, and
            # its answer happened to be NONE or disagree" (classified) --
            # call_failed rows must NOT be counted as disagreement in the
            # final agreement rate, or a technical failure would silently
            # masquerade as a real reliability finding.
            if not letter:
                pred_idx = None
                pred_label = ""
                status = "call_failed"
            elif letter == "NONE":
                pred_idx = None
                pred_label = "NONE"
                status = "classified"
            elif letter in letters:
                pred_idx = letters.index(letter)
                pred_label = letter_to_label[pred_idx]
                status = "classified"
            else:
                pred_idx = None
                pred_label = ""
                status = "call_failed"
            out.append({
                "run_name": run_dir.name,
                "review_idx": row["review_idx"],
                "open_code": row["open_code"],
                "quote": row["quote"],
                "sibling_quotes": row.get("sibling_quotes", []),
                "true_meta_theme_idx": row["true_meta_theme_idx"],
                "true_meta_theme_label": row["true_meta_theme_label"],
                "pred_meta_theme_idx": pred_idx,
                "pred_meta_theme_label": pred_label,
                "pred_letter_raw": letter,
                "status": status,
                "agree": (pred_idx == row["true_meta_theme_idx"]) if status == "classified" else None,
                "detail_level": detail_level,
                "classifier_model": classifier_cfg.model,
            })
        return out

    all_rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_do_batch, b) for b in batches]
        for future in as_completed(futures):
            all_rows.extend(future.result())

    classified = [r for r in all_rows if r["status"] == "classified"]
    n_agree = sum(1 for r in classified if r["agree"])
    logger.info("reclassify_run %s: %d quotes, %d classified, %d call_failed, %d agree (%.1f%% of classified)",
                run_dir.name, len(all_rows), len(classified), len(all_rows) - len(classified),
                n_agree, 100 * n_agree / len(classified) if classified else 0)
    return all_rows
