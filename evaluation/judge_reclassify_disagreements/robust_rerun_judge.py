#!/usr/bin/env python3
"""Robust rerun orchestration for the reclassify-disagreement judge --
mirrors robust_rerun_reclassify.py's retry structure: identify items that
failed to get a usable verdict (status=="call_failed"), retry just those
(re-batched) up to MAX_ROUNDS times, and never silently merge a failed item
into the final output as if it were a real verdict -- unresolved items go
to a separate file.

Usage:
    python3 -m judge_reclassify_disagreements.robust_rerun_judge \
        --judge-model qwen3.6-27b --judge-url http://localhost:8000/v1 \
        --out-path <path> --unresolved-path <path> [--limit N]
"""
from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

from judge_pipeline import ModelConfig, judge_batch, judge_all

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MAX_ROUNDS = 8
ITEM_POOL_PATH = Path(__file__).resolve().parent / "item_pool.jsonl"


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def row_key(row: dict) -> str:
    return row["item_id"]


def retry_failed(failed_items_by_id: dict, item_lookup: dict, cfg: ModelConfig, batch_size: int,
                  checkpoint_path: Path | None = None, round_n: int = 1) -> list[dict]:
    """failed_items_by_id: item_id -> previous output row (unused except for id set).
    Re-batches the ORIGINAL item content (from item_lookup) for a fresh judge attempt.
    Newly-judged rows are also appended to checkpoint_path so a crash mid-retry
    doesn't lose them either.

    Decoding params (temperature etc.) are never touched between rounds -- those
    are part of the reported experimental settings. Instead, item ORDER is
    reshuffled (seeded by round_n, so it's still reproducible) before rechunking
    into batches, so a batch that deterministically produced malformed JSON at
    temperature=0 isn't handed back to the model in the exact same grouping --
    it ends up mixed with different neighbors, which changes the prompt (and
    therefore the greedy-decoded output) without changing any model parameter."""
    batch = [item_lookup[iid] for iid in failed_items_by_id]
    random.Random(round_n).shuffle(batch)
    out = []
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_fh = checkpoint_path.open("a", encoding="utf-8") if checkpoint_path is not None else None
    try:
        for chunk_start in range(0, len(batch), batch_size):
            chunk = batch[chunk_start:chunk_start + batch_size]
            preds = judge_batch(chunk, cfg.model, cfg.base_url, service_tier=cfg.service_tier)
            for i, item in enumerate(chunk, start=1):
                pick = preds.get(i, "")
                if pick not in ("A", "B"):
                    status, matches = "call_failed", None
                else:
                    status, matches = "judged", (pick == item["true_side"])
                row = {
                    "item_id": item["item_id"], "system": item["system"], "dataset": item["dataset"],
                    "run_name": item["run_name"], "review_idx": item["review_idx"], "open_code": item["open_code"],
                    "true_side": item["true_side"], "true_label": item["true_label"], "pred_label": item["pred_label"],
                    "judge_pick": pick, "status": status, "matches_pipeline": matches, "judge_model": cfg.model,
                }
                out.append(row)
                if ckpt_fh is not None and status == "judged":
                    ckpt_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    ckpt_fh.flush()
    finally:
        if ckpt_fh is not None:
            ckpt_fh.close()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--judge-model", required=True)
    p.add_argument("--judge-url", required=True)
    p.add_argument("--out-path", required=True)
    p.add_argument("--unresolved-path", default=None)
    p.add_argument("--batch-size", type=int, default=10)
    p.add_argument("--max-workers", type=int, default=8)
    p.add_argument("--service-tier", default=None, help="e.g. 'flex' for OpenRouter external judges")
    p.add_argument("--limit", type=int, default=None, help="only judge the first N items (for a quick pilot)")
    args = p.parse_args()

    out_path = Path(args.out_path)
    unresolved_path = Path(args.unresolved_path) if args.unresolved_path else out_path.with_suffix(".unresolved.jsonl")
    checkpoint_path = out_path.parent / f".{out_path.stem}_checkpoint.jsonl"
    cfg = ModelConfig(model=args.judge_model, base_url=args.judge_url, service_tier=args.service_tier)

    items = [json.loads(l) for l in ITEM_POOL_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        items = items[:args.limit]
    item_lookup = {it["item_id"]: it for it in items}
    logger.info("Judging %d items with %s (checkpoint: %s)", len(items), args.judge_model, checkpoint_path)

    all_rows = judge_all(items, cfg, batch_size=args.batch_size, max_workers=args.max_workers,
                          checkpoint_path=checkpoint_path)
    good: dict[str, dict] = {}
    bad: dict[str, dict] = {}
    for r in all_rows:
        (good if r["status"] == "judged" else bad)[row_key(r)] = r

    round_n = 1
    while bad and round_n <= MAX_ROUNDS:
        logger.warning("Judge retry round %d: %d item(s) still unresolved", round_n, len(bad))
        retried = retry_failed(bad, item_lookup, cfg, args.batch_size, checkpoint_path=checkpoint_path,
                                round_n=round_n)
        still_bad = {}
        for r in retried:
            if r["status"] == "judged":
                good[row_key(r)] = r
            else:
                still_bad[row_key(r)] = r
        bad = still_bad
        round_n += 1

    write_jsonl(list(good.values()), out_path)
    if bad:
        write_jsonl(list(bad.values()), unresolved_path)
        logger.error("UNRESOLVED after %d rounds: %d item(s) -> %s", MAX_ROUNDS, len(bad), unresolved_path)
    else:
        checkpoint_path.unlink(missing_ok=True)
    logger.info("Judge done: %d good, %d unresolved -> %s", len(good), len(bad), out_path)

    classified = list(good.values())
    n_match = sum(1 for r in classified if r["matches_pipeline"])
    if classified:
        logger.info("Of judged items: %d/%d (%.1f%%) sided with the pipeline's original assignment",
                     n_match, len(classified), 100 * n_match / len(classified))


if __name__ == "__main__":
    main()
