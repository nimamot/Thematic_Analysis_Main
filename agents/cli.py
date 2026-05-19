"""CLI entry point for the GT pipeline."""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from agents.core.app import app
from agents.core.batch_open_coding import run_batched_open_coding
from agents.core.cooccurrence import write_cooccurrence
from agents.core.paths import (
    CLUSTERED_CODES_PATH,
    CODEBOOK_PATH,
    COOCCURRENCE_PATH,
    DEFAULT_DATA_CSV,
    GLOBAL_GRAPH_PATH,
    GT_CODES_ONLY_PATH,
    HIERARCHY_PATH,
    META_THEMES_PATH,
    OPEN_CODES_MARKDOWN_PATH,
    RESEARCH_REPORT_PATH,
    display_path,
    ensure_output_dirs,
)
from agents.core.report import generate_research_report
from agents.core.utils import extract_codes, log_step, parse_open_codes_markdown


def _base_state(research_question: str) -> dict:
    """Initial state dict for the LangGraph; matches GTState shape."""
    return {
        "research_question": research_question,
        "raw_text": "",
        "open_codes": None,
        "open_codes_validation": None,
        "open_codes_validation_feedback": None,
        "_open_coding_retries": 0,
        "all_codes_for_axial": None,
        "axial_mapping": None,
        "_cluster_refinement_done": False,
        "codebook": None,
        "hierarchy": None,
        "meta_themes": None,
        "global_graph": None,
        "tool_call": None,
        "step": 0,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--open-coding-only",
        action="store_true",
        help="Run only open coding; save gt_codes_only.json and exit (so SGLang can be killed before axial).",
    )
    p.add_argument(
        "--axial-only",
        action="store_true",
        help="Load gt_codes_only.json and run only the axial step in the graph.",
    )
    p.add_argument(
        "--rebuild-codes-json",
        action="store_true",
        help="Re-parse outputs/data/gt_open_codes_all_reviews.md and rewrite gt_codes_only.json (no LLM).",
    )
    p.add_argument(
        "--high-level-only",
        action="store_true",
        help="Load gt_clustered_codes.json and run high-level code generation (LLM must be up).",
    )
    p.add_argument(
        "--refine-only",
        action="store_true",
        help="Run high-level (if needed) then refine_cluster_assignments. Requires codebook.json and gt_clustered_codes.json. LLM must be up.",
    )
    p.add_argument(
        "--hierarchy-only",
        action="store_true",
        help="Run hierarchy construction (intra-cluster sub-theme grouping). LLM must be up.",
    )
    p.add_argument(
        "--graph-only",
        action="store_true",
        help="(Removed) Per-cluster graph step no longer exists; exits with instructions.",
    )
    p.add_argument(
        "--meta-themes-only",
        action="store_true",
        help="Run meta-theme grouping (group cluster labels into a small handful of meta-themes, ~3–7 when many clusters). LLM must be up.",
    )
    p.add_argument(
        "--tree-only",
        action="store_true",
        help="Run tree assembly (build hierarchical tree from meta-themes + hierarchy). No LLM needed.",
    )
    p.add_argument(
        "--global-graph-only", action="store_true", help="Alias for --tree-only (backward compat)."
    )
    p.add_argument(
        "--report-only",
        action="store_true",
        help="Generate research_report.md from gt_global_graph.json (Mistral/SGLang report server must be up).",
    )
    p.add_argument(
        "--cooccurrence-only",
        action="store_true",
        help="Build gt_cooccurrence.json from gt_clustered_codes.json + gt_global_graph.json (no LLM).",
    )
    p.add_argument(
        "--graph-path",
        default=None,
        help="Path to global graph JSON for --report-only / --cooccurrence-only (default: outputs/data/gt_global_graph.json).",
    )
    p.add_argument(
        "--clustered-path",
        default=None,
        help="Path to gt_clustered_codes.json for --cooccurrence-only (default: outputs/data/gt_clustered_codes.json).",
    )
    p.add_argument(
        "--report-api-base",
        default=None,
        help="OpenAI-compatible API base for report (default: env REPORT_OPENAI_BASE or http://localhost:8000/v1).",
    )
    p.add_argument(
        "--report-model",
        default=None,
        help="Model name for report (default: env REPORT_MODEL_NAME or llm).",
    )
    p.add_argument(
        "--skip-cross-cluster",
        action="store_true",
        help="(Deprecated, ignored) Cross-cluster linking removed in tree mode.",
    )
    p.add_argument(
        "--sim-threshold",
        type=float,
        default=0.75,
        help="(Deprecated, ignored) Cosine similarity threshold no longer used in tree mode.",
    )
    p.add_argument(
        "--research-question",
        required=True,
        help="Research question that conditions the entire GT pipeline.",
    )
    p.add_argument(
        "--data",
        default=str(DEFAULT_DATA_CSV),
        help="CSV with a text column: 'text_review' (preferred) or 'review_text'.",
    )
    args = p.parse_args()

    if args.graph_only:
        print(
            "agents.cli: --graph-only was removed in the hierarchical pipeline refactor.",
            file=sys.stderr,
        )
        print(
            "Use: --meta-themes-only (after --hierarchy-only), then --global-graph-only or --tree-only.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    ensure_output_dirs()
    rq = args.research_question
    log_step("RESEARCH_QUESTION", rq)

    if args.high_level_only:
        if not CLUSTERED_CODES_PATH.is_file():
            print(f"Error: {display_path(CLUSTERED_CODES_PATH)} not found. Run axial step first.")
            raise SystemExit(1)
        state = _base_state(rq)
        state["axial_mapping"] = "done"
        final_hl = app.invoke(state, config={"recursion_limit": 25})
        codebook = final_hl.get("codebook") or {}
        log_step(
            "CODEBOOK_COMPLETE",
            f"Generated {len(codebook)} high-level labels. See {display_path(CODEBOOK_PATH)}",
        )
        raise SystemExit(0)

    if args.refine_only:
        if not CODEBOOK_PATH.is_file():
            print(f"Error: {display_path(CODEBOOK_PATH)} not found. Run high-level step first.")
            raise SystemExit(1)
        if not CLUSTERED_CODES_PATH.is_file():
            print(f"Error: {display_path(CLUSTERED_CODES_PATH)} not found. Run axial step first.")
            raise SystemExit(1)
        state = _base_state(rq)
        state["axial_mapping"] = "refine"
        with open(CODEBOOK_PATH, encoding="utf-8") as f:
            existing_cb = json.load(f)
        state["codebook"] = existing_cb.get("codebook", {})
        final_refine = app.invoke(state, config={"recursion_limit": 25})
        summary = final_refine.get("_cluster_refinement_done", False)
        log_step(
            "REFINE_COMPLETE",
            "Cluster refinement finished." if summary else "Refine step completed.",
        )
        raise SystemExit(0)

    if args.hierarchy_only:
        if not CODEBOOK_PATH.is_file():
            print(f"Error: {display_path(CODEBOOK_PATH)} not found. Run high-level step first.")
            raise SystemExit(1)
        state = _base_state(rq)
        state["axial_mapping"] = "hierarchy"
        final_hier = app.invoke(state, config={"recursion_limit": 25})
        log_step("HIERARCHY_COMPLETE", final_hier.get("hierarchy", ""))
        raise SystemExit(0)

    if args.meta_themes_only:
        if not CODEBOOK_PATH.is_file():
            print(f"Error: {display_path(CODEBOOK_PATH)} not found. Run high-level step first.")
            raise SystemExit(1)
        state = _base_state(rq)
        state["axial_mapping"] = "meta_themes"
        final_mt = app.invoke(state, config={"recursion_limit": 25})
        log_step("META_THEMES_COMPLETE", final_mt.get("meta_themes", ""))
        raise SystemExit(0)

    if args.tree_only or args.global_graph_only:
        if not META_THEMES_PATH.is_file():
            print(f"Error: {display_path(META_THEMES_PATH)} not found. Run meta-themes step first.")
            raise SystemExit(1)
        if not HIERARCHY_PATH.is_file():
            print(f"Error: {display_path(HIERARCHY_PATH)} not found. Run hierarchy step first.")
            raise SystemExit(1)
        state = _base_state(rq)
        state["axial_mapping"] = "tree"
        final_tree = app.invoke(state, config={"recursion_limit": 25})
        log_step("TREE_COMPLETE", final_tree.get("global_graph", ""))
        raise SystemExit(0)

    if args.report_only:
        graph_file = Path(args.graph_path) if args.graph_path else GLOBAL_GRAPH_PATH
        if not graph_file.is_file():
            print(
                f"Error: {graph_file} not found. Run tree assembly step first (or pass --graph-path)."
            )
            raise SystemExit(1)
        generate_research_report(
            rq,
            graph_file,
            RESEARCH_REPORT_PATH,
            api_base=args.report_api_base,
            model=args.report_model,
        )
        log_step("RESEARCH_REPORT_COMPLETE", f"Wrote {display_path(RESEARCH_REPORT_PATH)}")
        raise SystemExit(0)

    if args.cooccurrence_only:
        graph_file = Path(args.graph_path) if args.graph_path else GLOBAL_GRAPH_PATH
        clustered_file = Path(args.clustered_path) if args.clustered_path else CLUSTERED_CODES_PATH
        if not clustered_file.is_file():
            print(
                f"Error: {clustered_file} not found. Run axial (and refine) so codes_per_review exists.",
                file=sys.stderr,
            )
            raise SystemExit(1)
        if not graph_file.is_file():
            print(
                f"Error: {graph_file} not found. Run tree assembly first (or pass --graph-path).",
                file=sys.stderr,
            )
            raise SystemExit(1)
        try:
            meta = write_cooccurrence(clustered_file, graph_file, COOCCURRENCE_PATH)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            raise SystemExit(1)
        sk = meta.get("skipped_unmapped_codes", 0)
        log_step(
            "COOCCURRENCE_COMPLETE",
            f"Wrote {display_path(COOCCURRENCE_PATH)} (skipped unmapped codes in reviews: {sk})",
        )
        raise SystemExit(0)

    if args.rebuild_codes_json:
        if not OPEN_CODES_MARKDOWN_PATH.is_file():
            print(
                f"Error: {display_path(OPEN_CODES_MARKDOWN_PATH)} not found. Run open coding first.",
                file=sys.stderr,
            )
            raise SystemExit(1)
        md_text = OPEN_CODES_MARKDOWN_PATH.read_text(encoding="utf-8")
        sections = parse_open_codes_markdown(md_text)
        if not sections:
            print(
                f"Error: no '## Review N' sections in {display_path(OPEN_CODES_MARKDOWN_PATH)}.",
                file=sys.stderr,
            )
            raise SystemExit(1)
        codes_per_review = [(rid, extract_codes(body)) for rid, body in sections]
        all_codes = [code for _, codes in codes_per_review for code in codes]
        with open(GT_CODES_ONLY_PATH, "w", encoding="utf-8") as f:
            json.dump(
                {"all_codes": all_codes, "codes_per_review": codes_per_review},
                f,
                indent=2,
            )
        log_step(
            "CODES_REBUILT",
            f"Total codes: {len(all_codes)} (from {len(codes_per_review)} reviews). See {display_path(GT_CODES_ONLY_PATH)}",
        )
        raise SystemExit(0)

    if args.axial_only:
        with open(GT_CODES_ONLY_PATH, encoding="utf-8") as f:
            data = json.load(f)
        all_codes = data.get("all_codes") or []
        if not all_codes:
            print(
                f"Error: {display_path(GT_CODES_ONLY_PATH)} has no codes. "
                f"Run with --rebuild-codes-json if {display_path(OPEN_CODES_MARKDOWN_PATH)} exists.",
                file=sys.stderr,
            )
            raise SystemExit(1)
        state = _base_state(rq)
        state["all_codes_for_axial"] = all_codes
        final_axial = app.invoke(state, config={"recursion_limit": 25})
        axial_mapping = final_axial.get("axial_mapping", "")
        log_step(
            "AXIAL_COMPLETE",
            axial_mapping[:500] + "..." if len(axial_mapping) > 500 else axial_mapping,
        )
        raise SystemExit(0)

    text_df = pd.read_csv(args.data)
    if "text_review" in text_df.columns:
        text_col = "text_review"
    elif "review_text" in text_df.columns:
        text_col = "review_text"
    else:
        print(
            f"Error: CSV must have column 'text_review' or 'review_text'; got {list(text_df.columns)}",
            file=sys.stderr,
        )
        raise SystemExit(1)
    reviews = text_df[text_col].astype(str).tolist()
    # Batched open coding + validation (GT_OPEN_CODING_BATCH_SIZE, GT_OPEN_CODING_BATCH_WORKERS).
    indexed = list(enumerate(reviews, start=1))
    results = run_batched_open_coding(indexed, rq)
    all_open_codes = [(idx, results[idx]) for idx in sorted(results)]

    with open(OPEN_CODES_MARKDOWN_PATH, "w", encoding="utf-8") as f:
        for review_id, codes in all_open_codes:
            f.write(f"## Review {review_id}\n\n{codes}\n\n")

    codes_per_review = [(review_id, extract_codes(raw)) for review_id, raw in all_open_codes]
    all_codes = [code for _, codes in codes_per_review for code in codes]
    with open(GT_CODES_ONLY_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "all_codes": all_codes,
                "codes_per_review": [(rid, codes) for rid, codes in codes_per_review],
            },
            f,
            indent=2,
        )
    log_step(
        "CODES_EXTRACTED",
        f"Total codes: {len(all_codes)} (from {len(codes_per_review)} reviews). See {display_path(GT_CODES_ONLY_PATH)}",
    )

    if args.open_coding_only:
        log_step("OPEN_CODING_ONLY", "Exiting so SGLang can be killed before axial.")
        raise SystemExit(0)

    state = _base_state(rq)
    state["all_codes_for_axial"] = all_codes
    final_axial = app.invoke(state, config={"recursion_limit": 25})
    axial_mapping = final_axial.get("axial_mapping", "")
    log_step(
        "AXIAL_COMPLETE", axial_mapping[:500] + "..." if len(axial_mapping) > 500 else axial_mapping
    )

    state = _base_state(rq)
    state["axial_mapping"] = "refine"
    final_refine = app.invoke(state, config={"recursion_limit": 25})
    codebook = final_refine.get("codebook") or {}
    log_step(
        "CODEBOOK_REFINE_COMPLETE",
        f"High-level and refinement done. {len(codebook)} clusters. See {display_path(CODEBOOK_PATH)}",
    )


if __name__ == "__main__":
    main()
