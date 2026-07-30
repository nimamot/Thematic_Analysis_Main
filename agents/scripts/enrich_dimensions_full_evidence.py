#!/usr/bin/env python3
"""Enrich meta-theme dimensions with definition, inclusion/exclusion criteria,
and examples, using the complete evidence available for each dimension.

Each dimension's definition/inclusion/exclusion text is written from every
open code's every quote across every cluster in that meta-theme (median
~128 open codes/meta-theme across the real data, up to 383 in extreme
cases). Exclusion-boundary criteria are additionally grounded against the
full quote set of every neighboring (other-dimension) cluster, so the model
never has to cite a bare cluster ID (e.g. "CL05") without having seen real
content for it.

Writes to gt_meta_themes_enriched_full.json.

Run AFTER --meta-themes-only has completed:
    python -m agents.scripts.enrich_dimensions_full_evidence

Reads:  gt_meta_themes.json + codebook.json + gt_open_codes_all_reviews.md
Writes: gt_meta_themes_enriched_full.json

Requires sglang to be running (same server used by the main pipeline),
model served as "llm".
"""
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from langchain_openai import ChatOpenAI

from agents.core.paths import (CODEBOOK_PATH, META_THEMES_PATH, DATA_DIR,
                                OPEN_CODES_MARKDOWN_PATH, ensure_output_dirs)
from agents.core.evidence_io import parse_code_evidence
from agents.core.skills import llm_invoke_with_skill
from agents.core.utils import clean_and_parse_json, log_step

ENRICHED_PATH = DATA_DIR / "gt_meta_themes_enriched_full.json"
REQUIRED_KEYS = {"label", "definition", "keywords", "inclusion", "exclusion"}
# No per-cluster cap -- show every quote for every open code in every cluster
# (see module docstring). Set to None to mean "unlimited".
MAX_QUOTES_PER_CLUSTER = None

# Pass 1 schema: criterion + code_ids only, no examples
_DIM_CRITERION_NO_EXAMPLE = {
    "type": "object",
    "properties": {
        "criterion": {"type": "string"},
        "code_ids":  {"type": "array", "items": {"type": "string"}},
    },
    "required": ["criterion", "code_ids"],
    "additionalProperties": False,
}

_DIM_ENTRY_NO_EXAMPLE_SCHEMA = {
    "type": "object",
    "properties": {
        "label":      {"type": "string"},
        "definition": {"type": "string"},
        "keywords":   {"type": "array", "items": {"type": "string"}},
        "inclusion":  {"type": "array", "items": _DIM_CRITERION_NO_EXAMPLE},
        "exclusion":  {"type": "array", "items": _DIM_CRITERION_NO_EXAMPLE},
    },
    "required": ["label", "definition", "keywords", "inclusion", "exclusion"],
    "additionalProperties": False,
}

def _make_induction_format(valid_ids: List[str], all_cluster_ids: List[str] = None) -> dict:
    """Build a response format with enum-constrained code_ids.
    inclusion uses only this meta-theme's cluster IDs;
    exclusion allows any cluster ID (to reference adjacent themes as boundaries).
    """
    def _criterion_item(ids: List[str]) -> dict:
        items_schema = (
            {"type": "string", "enum": sorted(ids)} if ids else {"type": "string"}
        )
        return {
            "type": "object",
            "properties": {
                "criterion": {"type": "string"},
                "code_ids":  {"type": "array", "items": items_schema},
            },
            "required": ["criterion", "code_ids"],
            "additionalProperties": False,
        }

    excl_ids = all_cluster_ids if all_cluster_ids else valid_ids
    schema = {
        "type": "object",
        "properties": {
            "label":      {"type": "string"},
            "definition": {"type": "string"},
            "keywords":   {"type": "array", "items": {"type": "string"}},
            "inclusion":  {"type": "array", "items": _criterion_item(valid_ids)},
            "exclusion":  {"type": "array", "items": _criterion_item(excl_ids)},
        },
        "required": ["label", "definition", "keywords", "inclusion", "exclusion"],
        "additionalProperties": False,
    }
    return {"type": "json_schema", "json_schema": {"name": "dimension_entry_no_example", "schema": schema}}

# Pass 2 schema: full entry with examples
_DIM_CRITERION_ITEM = {
    "type": "object",
    "properties": {
        "criterion": {"type": "string"},
        "code_ids":  {"type": "array", "items": {"type": "string"}},
        "examples":  {"type": "array", "items": {"type": "string"}},
    },
    "required": ["criterion", "code_ids", "examples"],
    "additionalProperties": False,
}

_DIM_ENTRY_SCHEMA = {
    "type": "object",
    "properties": {
        "label":      {"type": "string"},
        "definition": {"type": "string"},
        "keywords":   {"type": "array", "items": {"type": "string"}},
        "inclusion":  {"type": "array", "items": _DIM_CRITERION_ITEM},
        "exclusion":  {"type": "array", "items": _DIM_CRITERION_ITEM},
    },
    "required": ["label", "definition", "keywords", "inclusion", "exclusion"],
    "additionalProperties": False,
}

DIM_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {"name": "dimension_entry", "schema": _DIM_ENTRY_SCHEMA},
}

_llm = ChatOpenAI(
    model="llm",
    openai_api_key="EMPTY",
    openai_api_base=os.environ.get("OPENAI_API_BASE", "http://localhost:8000/v1"),
    temperature=0,
    max_tokens=4096,
    model_kwargs={"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
)


def _quotes_block(cluster_ids: List[str],
                   cluster_to_codes: Dict[str, List[str]],
                   code_evidence: Dict[str, List[str]],
                   cluster_labels: List[str] = None) -> str:
    """Build prompt block with cluster IDs, optional labels, and real participant
    quotes -- ALL of them, no per-cluster cap (see module docstring)."""
    lines = []
    for i, cid in enumerate(cluster_ids):
        padded = f"CL{cid.zfill(2)}"
        header = f"- [{padded}]" + (f" {cluster_labels[i]}" if cluster_labels else "")
        lines.append(header)
        codes = cluster_to_codes.get(cid, [])
        for code in codes:
            for quote in code_evidence.get(code, []):
                lines.append(f'  [QUOTE] "{quote}"')
    return "\n".join(lines)


def _neighbor_quotes_block(own_cluster_ids: List[str],
                            cluster_to_codes: Dict[str, List[str]],
                            code_evidence: Dict[str, List[str]],
                            n_per_cluster: int | None = None) -> str:
    """
    Show ALL real quotes from every OTHER cluster (not belonging to this
    meta-theme), so the exclusion criterion has full grounding for what
    neighboring clusters actually are -- rather than the model citing bare
    cluster IDs (e.g. "CL05") it has never seen real content for.

    n_per_cluster is available as an optional cap (None = unlimited) for
    cases where prompt length or cost needs to be constrained.
    """
    own = set(own_cluster_ids)
    other_cids = sorted(
        (cid for cid in cluster_to_codes if cid not in own),
        key=lambda x: int(x) if x.isdigit() else x,
    )
    lines = []
    for cid in other_cids:
        padded = f"CL{cid.zfill(2)}"
        sample: List[str] = []
        for code in cluster_to_codes.get(cid, []):
            for quote in code_evidence.get(code, []):
                sample.append(quote)
                if n_per_cluster is not None and len(sample) >= n_per_cluster:
                    break
            if n_per_cluster is not None and len(sample) >= n_per_cluster:
                break
        if not sample:
            continue
        lines.append(f"- [{padded}]")
        lines.extend(f'  [QUOTE] "{q}"' for q in sample)
    return "\n".join(lines)


def _quote_in_cluster_evidence(quote: str, cluster_id: str,
                                cluster_to_codes: Dict[str, List[str]],
                                code_evidence: Dict[str, List[str]]) -> bool:
    """Return True if quote is sufficiently similar to any evidence in the cluster."""
    q = " ".join(quote.split()).lower()
    if not q:
        return True
    known: List[str] = []
    for code in cluster_to_codes.get(cluster_id, []):
        known.extend(code_evidence.get(code, []))
    for ev in known:
        ev_n = " ".join(ev.split()).lower()
        if q in ev_n or ev_n in q:
            return True
    if not known:
        return True  # no evidence loaded — accept rather than drop
    q_words = set(q.split())
    if len(q_words) < 6:
        return False
    return max(
        len(q_words & set(" ".join(ev.split()).lower().split())) / len(q_words)
        for ev in known
    ) >= 0.80


def _raw_cid(cid: str) -> str:
    """'CL03' → '3', 'CL00' → '0'."""
    return str(int(cid[2:])) if cid.startswith("CL") and cid[2:].isdigit() else cid


def _validate_and_align(parsed: Dict[str, Any], valid_ids: set,
                         cluster_to_codes: Dict[str, List[str]],
                         code_evidence: Dict[str, List[str]],
                         name: str) -> None:
    """Drop invalid cluster IDs, ungrounded examples, and ungrounded supporting_quotes."""
    for section in ("inclusion", "exclusion"):
        for item in parsed.get(section, []):
            if not isinstance(item, dict):
                continue

            raw_ids = item.get("code_ids", [])
            good_ids = [cid for cid in raw_ids if cid in valid_ids]
            if len(good_ids) < len(raw_ids):
                bad = set(raw_ids) - set(good_ids)
                log_step("DIM_ENRICH_ID_WARN", f"'{name}': invalid code_ids {bad} — removed")
            item["code_ids"] = good_ids

            examples = item.get("examples", [])
            paired = list(zip(good_ids, examples))
            valid_pairs: List[Tuple[str, str]] = []
            for cid, ex in paired:
                if _quote_in_cluster_evidence(ex, _raw_cid(cid), cluster_to_codes, code_evidence):
                    valid_pairs.append((cid, ex))
                else:
                    log_step("DIM_ENRICH_QUOTE_WARN",
                             f"'{name}': example for {cid} not in cluster evidence — dropped")
            if len(good_ids) > len(examples):
                log_step("DIM_ENRICH_ALIGN",
                         f"'{name}': {len(good_ids) - len(examples)} code_ids without examples — dropped")
            item["code_ids"] = [cid for cid, _ in valid_pairs]
            item["examples"] = [ex  for _, ex  in valid_pairs]


def _fill_examples(entry: Dict[str, Any],
                   cluster_to_codes: Dict[str, List[str]],
                   code_evidence: Dict[str, List[str]]) -> None:
    """Programmatically fill examples from real evidence — no LLM involved.
    Takes evs[0] per cited cluster (open coding LLM already chose the best one),
    deduplicates across the entire entry (no quote reused across criteria), caps
    at 3 -- this is the DISPLAY examples field shown alongside each criterion,
    not the evidence used to write the criteria text itself.
    """
    used_globally: set = set()
    for section in ("inclusion", "exclusion"):
        for item in entry.get(section, []):
            examples: List[str] = []
            for cid in item.get("code_ids", []):
                raw_cid = _raw_cid(cid)
                for code in cluster_to_codes.get(raw_cid, []):
                    ev = next(iter(code_evidence.get(code, [])), None)
                    if ev and ev not in used_globally:
                        used_globally.add(ev)
                        examples.append(ev)
                        break
                if len(examples) == 3:
                    break
            item["examples"] = examples


def _enrich_one(name: str, cluster_ids: List[str], cluster_labels: List[str],
                cluster_to_codes: Dict[str, List[str]],
                code_evidence: Dict[str, List[str]]) -> Dict[str, Any]:
    available_ids = ", ".join(f"CL{cid.zfill(2)}" for cid in cluster_ids)
    valid_ids = set(available_ids.replace(" ", "").split(","))
    all_cluster_ids = [f"CL{cid.zfill(2)}" for cid in sorted(cluster_to_codes.keys(), key=lambda x: int(x) if x.isdigit() else x)]
    all_valid_ids = set(all_cluster_ids)

    induction_format = _make_induction_format(list(valid_ids), all_cluster_ids)

    def _run_pass1(prompt: str) -> Dict[str, Any]:
        raw = llm_invoke_with_skill(_llm, "dimension_criteria_induction", prompt,
                                    response_format=induction_format)
        parsed = clean_and_parse_json(raw)
        if not isinstance(parsed, dict) or not REQUIRED_KEYS.issubset(parsed):
            raise ValueError(f"Pass 1 missing keys: {REQUIRED_KEYS - set(parsed)}")
        parsed["label"] = name
        return parsed

    def _filter_and_collect_bad(parsed: Dict[str, Any]) -> set:
        bad_all: set = set()
        for section, allowed in (("inclusion", valid_ids), ("exclusion", all_valid_ids)):
            for item in parsed.get(section, []):
                raw_ids = item.get("code_ids", [])
                bad = {cid for cid in raw_ids if cid not in allowed}
                bad_all |= bad
                item["code_ids"] = [cid for cid in raw_ids if cid in allowed]
        return bad_all

    def _drop_empty_criteria(parsed: Dict[str, Any]) -> int:
        dropped = 0
        for section in ("inclusion", "exclusion"):
            before = parsed.get(section, [])
            after = [item for item in before if item.get("code_ids")]
            dropped += len(before) - len(after)
            parsed[section] = after
        return dropped

    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            quotes_only_text = _quotes_block(cluster_ids, cluster_to_codes, code_evidence)
            neighbor_quotes_text = _neighbor_quotes_block(cluster_ids, cluster_to_codes, code_evidence)
            all_ids_text = ", ".join(all_cluster_ids)
            base_prompt = (
                f"Dimension: {name}\n"
                f"This dimension's Cluster IDs (use for inclusion): {available_ids}\n"
                f"All available Cluster IDs (may use for exclusion boundary criteria): {all_ids_text}\n\n"
                f"Participant quotes by cluster (no labels — derive from the quotes):\n"
                f"{quotes_only_text}\n\n"
                f"ALL quotes from OTHER (neighboring) clusters, for grounding exclusion "
                f"boundary criteria only -- these do NOT belong to this dimension:\n"
                f"{neighbor_quotes_text}"
            )
            parsed = _run_pass1(base_prompt)
            bad_ids = _filter_and_collect_bad(parsed)
            dropped = _drop_empty_criteria(parsed)

            if bad_ids:
                log_step("DIM_ENRICH_RETRY",
                         f"'{name}': invalid IDs {bad_ids} — retrying with explicit feedback")
                retry_prompt = (
                    f"{base_prompt}\n\n"
                    f"CORRECTION: your previous response used these non-existent IDs: "
                    f"{', '.join(sorted(bad_ids))}. "
                    f"For inclusion use only: {available_ids}. "
                    f"For exclusion use only: {all_ids_text}."
                )
                parsed = _run_pass1(retry_prompt)
                bad_ids2 = _filter_and_collect_bad(parsed)
                dropped2 = _drop_empty_criteria(parsed)
                if bad_ids2:
                    log_step("DIM_ENRICH_WARN",
                             f"'{name}': still invalid IDs after retry {bad_ids2} — dropped")
                dropped += dropped2

            if dropped:
                log_step("DIM_ENRICH_DROPPED", f"'{name}': {dropped} criteria dropped (no valid code_ids)")

            _fill_examples(parsed, cluster_to_codes, code_evidence)
            return parsed

        except Exception as exc:
            is_connection_err = "connection" in str(exc).lower() or "timeout" in str(exc).lower()
            if is_connection_err and attempt < max_attempts - 1:
                wait = 10 * (attempt + 1)
                log_step("DIM_ENRICH_RETRY",
                         f"'{name}': connection error on attempt {attempt+1}/{max_attempts} "
                         f"— retrying in {wait}s: {exc}")
                time.sleep(wait)
                continue
            log_step("DIM_ENRICH_WARN", f"'{name}': {exc}")
            return {"label": name, "definition": "", "keywords": [],
                    "inclusion": [], "exclusion": []}


def main() -> None:
    if not META_THEMES_PATH.is_file():
        print(f"Error: {META_THEMES_PATH} not found. Run --meta-themes-only first.")
        sys.exit(1)
    if not CODEBOOK_PATH.is_file():
        print(f"Error: {CODEBOOK_PATH} not found. Run axial step first.")
        sys.exit(1)

    with open(META_THEMES_PATH, encoding="utf-8") as f:
        meta_themes_data = json.load(f)
    with open(CODEBOOK_PATH, encoding="utf-8") as f:
        cb_data = json.load(f)

    codebook: Dict[str, str] = cb_data.get("codebook", {})
    cluster_to_codes: Dict[str, List[str]] = cb_data.get("cluster_to_codes", {})
    meta_themes: List[Dict] = meta_themes_data.get("meta_themes", [])

    if not meta_themes:
        print("Error: no meta_themes found in gt_meta_themes.json.")
        sys.exit(1)

    from agents.core.paths import DEFAULT_DATA_CSV
    csv_path = Path(os.environ.get("GT_DATA_CSV", str(DEFAULT_DATA_CSV)))
    code_evidence, _ = parse_code_evidence(OPEN_CODES_MARKDOWN_PATH, csv_path)
    print(f"Loaded evidence for {len(code_evidence)} codes from {OPEN_CODES_MARKDOWN_PATH.name}")
    print(f"Enriching {len(meta_themes)} dimensions (FULL EVIDENCE, no per-cluster quote cap)...")

    workers = int(os.environ.get("GT_ENRICH_WORKERS", "4"))
    enriched_list = [None] * len(meta_themes)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {}
        for idx, mt in enumerate(meta_themes):
            name = mt.get("name", f"Dimension {idx}")
            cids = [str(cid) for cid in mt.get("cluster_ids", [])]
            cluster_labels = [codebook.get(cid, f"Cluster {cid}") for cid in cids]
            futures[ex.submit(_enrich_one, name, cids, cluster_labels,
                              cluster_to_codes, code_evidence)] = idx

        for fut in as_completed(futures):
            idx = futures[fut]
            entry = fut.result()
            entry["cluster_ids"] = meta_themes[idx].get("cluster_ids", [])
            enriched_list[idx] = entry
            log_step("DIM_ENRICH_DONE", entry["label"])

    failed = [e["label"] for e in enriched_list if not e.get("inclusion") and not e.get("definition")]
    if failed:
        print(f"ERROR: {len(failed)} meta-theme(s) failed enrichment (empty result): {failed}")
        print("Fix the LLM server connection and rerun this script.")
        sys.exit(1)

    ensure_output_dirs()
    out = {"meta_themes_enriched": enriched_list}
    with open(ENRICHED_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"Done. Written to {ENRICHED_PATH}")


if __name__ == "__main__":
    main()
