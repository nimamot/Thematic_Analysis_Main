#!/usr/bin/env python3
"""Robust rerun orchestration for the codebook reclassification reliability
pipeline -- mirrors robust_rerun_hicode_run123.py / robust_rerun_coverage.py's
retry structure: identify quotes that failed to get a usable classification
(status=="call_failed"), retry just those (re-batched) up to MAX_ROUNDS times,
and never silently merge a failed quote into the final output as if it were
a real "disagreement" -- unresolved quotes go to a separate file.

Usage:
    python3 -m evaluation.robust_rerun_reclassify \
        --run-dirs run1,run2,... --runs-root <path> \
        --classifier-model gemma4-31b --classifier-url http://localhost:8000/v1 \
        --out-path <path> --unresolved-path <path>
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from evaluation.reclassify_pipeline import (
    ModelConfig, reclassify_run, classify_batch, format_codebook,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MAX_ROUNDS = 8


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def row_key(row: dict) -> tuple:
    return (row["run_name"], row["review_idx"], row["open_code"])


def retry_failed_for_run(run_dir: Path, failed_rows: list[dict], classifier_cfg: ModelConfig,
                          batch_size: int, enriched_filename: str = "gt_meta_themes_enriched.json",
                          detail_level: str = "full", include_reasoning: bool = False) -> list[dict]:
    """Re-batch and reclassify just the previously-failed quotes for one run.

    Chunks failed_rows into batch_size-sized groups before calling
    classify_batch() -- previously this dumped ALL failed rows for a run
    into a single call regardless of count, which made retries WORSE (not
    better) whenever the failure count was large: a bigger single-call batch
    needs a bigger output token budget, so a batch that already failed from
    output truncation would fail even harder on retry.
    """
    codebook_text, letter_to_label = format_codebook(run_dir, enriched_filename, detail_level=detail_level)
    n_meta_themes = len(letter_to_label)
    letters = [chr(ord("A") + i) for i in range(n_meta_themes)]

    out = []
    for chunk_start in range(0, len(failed_rows), batch_size):
        chunk = failed_rows[chunk_start:chunk_start + batch_size]
        quotes = [{"quote": r["quote"], "sibling_quotes": r.get("sibling_quotes", [])} for r in chunk]
        preds = classify_batch(codebook_text, quotes, classifier_cfg.model, classifier_cfg.base_url,
                                n_meta_themes=n_meta_themes, detail_level=detail_level,
                                include_reasoning=include_reasoning)
        for i, row in enumerate(chunk, start=1):
            letter = preds.get(i, "")
            if not letter or letter not in (set(letters) | {"NONE"}):
                status, pred_idx, pred_label = "call_failed", None, ""
            elif letter == "NONE":
                status, pred_idx, pred_label = "classified", None, "NONE"
            else:
                status, pred_idx, pred_label = "classified", letters.index(letter), letter_to_label[letters.index(letter)]
            new_row = dict(row)
            new_row.update({
                "pred_meta_theme_idx": pred_idx,
                "pred_meta_theme_label": pred_label,
                "pred_letter_raw": letter,
                "status": status,
                "agree": (pred_idx == row["true_meta_theme_idx"]) if status == "classified" else None,
                "detail_level": detail_level,
                "classifier_model": classifier_cfg.model,
            })
            out.append(new_row)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dirs", required=True, help="comma-separated run directory names")
    p.add_argument("--runs-root", default=str(REPO_ROOT / "runs"))
    p.add_argument("--classifier-model", required=True)
    p.add_argument("--classifier-url", required=True)
    p.add_argument("--out-path", required=True)
    p.add_argument("--unresolved-path", default=None)
    p.add_argument("--batch-size", type=int, default=25)
    p.add_argument("--enriched-filename", default="gt_meta_themes_enriched.json",
                    help="which enriched meta-theme file to read, e.g. gt_meta_themes_enriched_full.json")
    p.add_argument("--codebook-detail", default="full", choices=["theme_only", "theme_def", "full"],
                    help="how much of the codebook the classifier sees -- ablation condition")
    p.add_argument("--include-reasoning", action="store_true",
                    help="include the per-item 'reasoning' field in the requested JSON schema. "
                         "Off by default -- reasoning text was never persisted to the output row "
                         "anyway, and it routinely pushed batches past the output token budget "
                         "and truncated mid-response for some models (confirmed root cause of a "
                         "wall of otherwise-unexplained parse failures).")
    args = p.parse_args()

    out_path = Path(args.out_path)
    unresolved_path = Path(args.unresolved_path) if args.unresolved_path else out_path.with_suffix(".unresolved.jsonl")
    run_names = [n.strip() for n in args.run_dirs.split(",")]
    cfg = ModelConfig(model=args.classifier_model, base_url=args.classifier_url)

    good: dict[tuple, dict] = {}
    bad_by_run: dict[str, list[dict]] = {}

    for run_name in run_names:
        run_dir = Path(args.runs_root) / run_name
        if not run_dir.is_dir():
            logger.warning("Missing run dir: %s", run_dir)
            continue
        rows = reclassify_run(run_dir, cfg, batch_size=args.batch_size, enriched_filename=args.enriched_filename,
                               detail_level=args.codebook_detail, include_reasoning=args.include_reasoning)
        for r in rows:
            if r["status"] == "call_failed":
                bad_by_run.setdefault(run_name, []).append(r)
            else:
                good[row_key(r)] = r

    round_n = 1
    while bad_by_run and round_n <= MAX_ROUNDS:
        n_bad = sum(len(v) for v in bad_by_run.values())
        logger.warning("Reclassify round %d: retrying %d failed quote(s) across %d run(s)",
                        round_n, n_bad, len(bad_by_run))
        still_bad: dict[str, list[dict]] = {}
        for run_name, failed_rows in bad_by_run.items():
            run_dir = Path(args.runs_root) / run_name
            retried = retry_failed_for_run(run_dir, failed_rows, cfg, args.batch_size, args.enriched_filename,
                                            detail_level=args.codebook_detail, include_reasoning=args.include_reasoning)
            for r in retried:
                if r["status"] == "call_failed":
                    still_bad.setdefault(run_name, []).append(r)
                else:
                    good[row_key(r)] = r
        bad_by_run = still_bad
        round_n += 1

    write_jsonl(list(good.values()), out_path)
    if bad_by_run:
        remaining = [r for rows in bad_by_run.values() for r in rows]
        write_jsonl(remaining, unresolved_path)
        logger.error("UNRESOLVED after %d rounds: %d quote(s) -> %s", MAX_ROUNDS, len(remaining), unresolved_path)
    logger.info("Reclassify done: %d good, %d unresolved -> %s",
                len(good), sum(len(v) for v in bad_by_run.values()), out_path)


if __name__ == "__main__":
    main()
