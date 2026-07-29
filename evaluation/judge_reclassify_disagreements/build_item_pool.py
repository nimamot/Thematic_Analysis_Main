#!/usr/bin/env python3
"""Build the deduplicated item pool of reclassify DISAGREEMENTS (full
condition, main pipeline + HICode) for automated LLM-judge adjudication --
same underlying comparison as human_eval_survey_reclassify_disagreement/,
but exhaustive (every qualifying disagreement, not a 50-item sample) and
judged by models instead of a human.

An item = one (run_name, review_idx, open_code, pred_meta_theme_label)
combination where at least one of the 5 reclassifier models (full
condition) disagreed with the pipeline's original assignment and predicted
a concrete (non-NONE) alternative. De-duplicated across the 5 source
models: if two models both predicted the same alternative label for the
same quote, that's one judge question, not two -- the question "does this
quote fit label P or label Q better" only depends on (quote, P, Q), not on
which specific reclassifier model raised it.

Output: item_pool.jsonl -- one row per item, quote + both full codebook
entries (label/definition/inclusion/exclusion), Option A/B order
RANDOMIZED and stored (so the judge prompt-builder just renders it
directly), plus the true identity (never shown to the judge model itself).
"""
import json
import random
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
EVALUATION_DIR = REPO_ROOT / "evaluation"
OUT_DIR = Path(__file__).resolve().parent

MODELS = ["qwen35_9b", "qwen35_27b", "qwen36moe", "gemma4_12b", "gemma4_26b_a4b"]

random.seed(20260726)


def load_disagreements():
    """key -> {quote, true_label, dataset, system, run_name, review_idx, open_code}"""
    by_key = {}
    for m in MODELS:
        path = EVALUATION_DIR / "results_reclassify_ablation" / m / "full" / "reclassified.jsonl"
        with open(path, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                if d["agree"]:
                    continue
                pred = d["pred_meta_theme_label"]
                if not pred or pred == "NONE":
                    continue
                key = (d["run_name"], d["review_idx"], d["open_code"], pred)
                if key in by_key:
                    continue  # de-dup: same (quote, true, pred) triple from another model
                system = "HICode" if d["run_name"].startswith("hicode_") else "main"
                dataset = d["run_name"].replace("hicode_", "").split("_qwen")[0].split("_gemma4")[0]
                by_key[key] = {
                    "quote": d["quote"],
                    "true_label": d["true_meta_theme_label"],
                    "pred_label": pred,
                    "dataset": dataset,
                    "system": system,
                    "run_name": d["run_name"],
                    "review_idx": d["review_idx"],
                    "open_code": d["open_code"],
                }
    return list(by_key.values())


_codebook_cache = {}


def load_codebook(run_name: str) -> dict:
    if run_name in _codebook_cache:
        return _codebook_cache[run_name]
    path = REPO_ROOT / "runs" / run_name / "data" / "gt_meta_themes_enriched_full.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    by_label = {}
    for mt in data.get("meta_themes_enriched", []):
        by_label[mt.get("label", "")] = {
            "definition": mt.get("definition", ""),
            "inclusion": [c.get("criterion", "") for c in mt.get("inclusion", [])],
            "exclusion": [c.get("criterion", "") for c in mt.get("exclusion", [])],
        }
    _codebook_cache[run_name] = by_label
    return by_label


def main():
    rows = load_disagreements()
    print(f"De-duplicated disagreement items (main + HICode, full condition, pred != NONE): {len(rows)}")

    by_system = defaultdict(int)
    for r in rows:
        by_system[r["system"]] += 1
    for sys_, n in by_system.items():
        print(f"  {sys_}: {n}")

    items = []
    for i, r in enumerate(rows, 1):
        codebook = load_codebook(r["run_name"])
        true_entry = codebook.get(r["true_label"], {"definition": "", "inclusion": [], "exclusion": []})
        pred_entry = codebook.get(r["pred_label"], {"definition": "", "inclusion": [], "exclusion": []})

        if random.random() < 0.5:
            option_a = {"label": r["true_label"], **true_entry}
            option_b = {"label": r["pred_label"], **pred_entry}
            true_side = "A"
        else:
            option_a = {"label": r["pred_label"], **pred_entry}
            option_b = {"label": r["true_label"], **true_entry}
            true_side = "B"

        items.append({
            "item_id": f"item_{i:06d}",
            "system": r["system"],
            "dataset": r["dataset"],
            "run_name": r["run_name"],
            "review_idx": r["review_idx"],
            "open_code": r["open_code"],
            "quote": r["quote"],
            "option_a": option_a,
            "option_b": option_b,
            "true_side": true_side,
            "true_label": r["true_label"],
            "pred_label": r["pred_label"],
        })

    out_path = OUT_DIR / "item_pool.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(items)} items -> {out_path}")


if __name__ == "__main__":
    main()
