"""Core batch-judging logic for the reclassify-disagreement adjudication --
mirrors reclassify_pipeline.py's classify_batch()/reclassify_run() design
(same retry/status conventions), applied to a flat item pool instead of
per-run quote lists.
"""
from __future__ import annotations

import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llm_client import call_llm, parse_json_response  # type: ignore

from prompts_judge import JUDGE_SYSTEM, build_user_prompt

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    model: str
    base_url: str
    service_tier: str | None = None


def _normalize_pick(raw) -> str:
    """gemma-4-31b sometimes writes 'pick': 'Option B' instead of the
    requested bare 'A'/'B' -- the content is unambiguous, so accept it
    rather than discarding a perfectly good verdict as call_failed."""
    s = str(raw).strip().upper()
    if s in ("A", "B"):
        return s
    m = re.fullmatch(r"OPTION\s*([AB])\.?", s)
    if m:
        return m.group(1)
    return ""


def judge_batch(items: list[dict], model: str, base_url: str, max_retries: int = 2,
                 service_tier: str | None = None) -> dict[int, str]:
    """Returns {item_batch_index (1-based): 'A'|'B'}."""
    user = build_user_prompt(items)
    max_out_tokens = min(8000, max(2000, len(items) * 150))

    for attempt in range(1 + max_retries):
        try:
            raw = call_llm(JUDGE_SYSTEM, user, model=model, base_url=base_url,
                            temperature=0.0, max_tokens=max_out_tokens,
                            service_tier=service_tier)
            data = parse_json_response(raw)
            verdicts = data.get("verdicts", [])
            result: dict[int, str] = {}
            for v in verdicts:
                idx = v.get("item_index")
                pick = _normalize_pick(v.get("pick", ""))
                if isinstance(idx, int) and 1 <= idx <= len(items) and pick in ("A", "B"):
                    result[idx] = pick
            if len(result) >= len(items) * 0.8:
                return result
            logger.warning("judge_batch: only %d/%d valid verdicts (attempt %d/%d)",
                            len(result), len(items), attempt + 1, 1 + max_retries)
        except Exception as exc:
            logger.warning("judge_batch failed (attempt %d/%d): %s", attempt + 1, 1 + max_retries, exc)
    return {}


def judge_all(items: list[dict], cfg: ModelConfig, batch_size: int = 10, max_workers: int = 8,
              checkpoint_path: Path | None = None) -> list[dict]:
    """items: full pool (each a dict from item_pool.jsonl). Returns rows with
    judge_pick / status / matches_pipeline added, one per item.

    If checkpoint_path is given: successfully-judged rows are appended to it
    as each batch finishes, and any item already present there (from a prior,
    interrupted run of this same call) is skipped rather than re-billed --
    mirrors pipeline.py's judge-phase checkpoint (crash/timeout-safe, keyed
    by item_id instead of (codebook_id, code_id, component))."""
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    checkpointed: dict[str, dict] = {}
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        if checkpoint_path.exists():
            for line in checkpoint_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    checkpointed[r["item_id"]] = r
            logger.info("Checkpoint loaded: %d already-judged item(s)", len(checkpointed))

    pending = [it for it in items if it["item_id"] not in checkpointed]
    batches = [pending[i:i + batch_size] for i in range(0, len(pending), batch_size)]

    ckpt_fh = checkpoint_path.open("a", encoding="utf-8") if checkpoint_path is not None else None
    ckpt_lock = threading.Lock()

    def _do_batch(batch: list[dict]) -> list[dict]:
        preds = judge_batch(batch, cfg.model, cfg.base_url, service_tier=cfg.service_tier)
        out = []
        for i, item in enumerate(batch, start=1):
            pick = preds.get(i, "")
            if pick not in ("A", "B"):
                status = "call_failed"
                matches_pipeline = None
            else:
                status = "judged"
                matches_pipeline = (pick == item["true_side"])
            row = {
                "item_id": item["item_id"],
                "system": item["system"],
                "dataset": item["dataset"],
                "run_name": item["run_name"],
                "review_idx": item["review_idx"],
                "open_code": item["open_code"],
                "true_side": item["true_side"],
                "true_label": item["true_label"],
                "pred_label": item["pred_label"],
                "judge_pick": pick,
                "status": status,
                "matches_pipeline": matches_pipeline,
                "judge_model": cfg.model,
            }
            out.append(row)
            if ckpt_fh is not None and status == "judged":
                with ckpt_lock:
                    ckpt_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    ckpt_fh.flush()
        return out

    all_rows: list[dict] = list(checkpointed.values())
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_do_batch, b) for b in batches]
            for future in as_completed(futures):
                all_rows.extend(future.result())
    finally:
        if ckpt_fh is not None:
            ckpt_fh.close()
    return all_rows
