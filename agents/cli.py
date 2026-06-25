"""CLI entry point for the GT pipeline."""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from agents.core.app import app
from agents.core.cooccurrence import write_cooccurrence
from agents.core.inference_config import qualitative_enrichment_enabled
from agents.core.codebook_review import (
    human_review_enabled,
    review_meta_from_env,
    review_mode,
    stage_v1_for_review,
    wait_for_approval,
    fetch_and_materialize_approved,
)
from agents.core.state import USE_OPEN_CODES_VALIDATOR
from agents.core.paths import (
    CLUSTERED_CODES_PATH,
    CODEBOOK_PATH,
    CODEBOOK_PROVENANCE_PATH,
    COOCCURRENCE_PATH,
    DEFAULT_DATA_CSV,
    GLOBAL_GRAPH_PATH,
    GT_CODES_ONLY_PATH,
    HIERARCHY_PATH,
    META_THEMES_ENRICHED_PATH,
    META_THEMES_PATH,
    OPEN_CODES_MARKDOWN_PATH,
    RESEARCH_REPORT_PATH,
    SOURCE_MEMORY_PATH,
    display_path,
    ensure_output_dirs,
)
from agents.core.qualitative_enrichment import (
    run_cluster_qualitative_enrichment,
    run_dimension_qualitative_enrichment,
    run_ground_enriched_only,
)
from agents.core.source_memory import SourceMemory
from agents.core.report import generate_research_report
from agents.core.utils import extract_codes, log_step


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
        "codebook_review_id": None,
        "codebook_review_status": None,
        "tool_call": None,
        "step": 0,
    }


def _graph_config(suffix: str | None = None) -> dict:
    slug = os.environ.get("PIPELINE_SLUG", "default").strip() or "default"
    job = os.environ.get("SLURM_JOB_ID", "local")
    thread_id = f"{slug}:{job}"
    if suffix:
        thread_id = f"{thread_id}:{suffix}"
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": 25}


def _require_approved_codebook(skip_review: bool) -> None:
    if skip_review or not human_review_enabled():
        return
    if CODEBOOK_PROVENANCE_PATH.is_file():
        return
    print(
        f"Error: {display_path(CODEBOOK_PROVENANCE_PATH)} missing. "
        "Run --wait-codebook-review or --fetch-codebook-review, or pass --skip-codebook-review.",
        file=sys.stderr,
    )
    raise SystemExit(1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--open-coding-only",
        action="store_true",
        help="Run only open coding; save gt_codes_only.json and exit (SGLang path stops server before axial).",
    )
    p.add_argument(
        "--axial-only",
        action="store_true",
        help="Load gt_codes_only.json and run only the axial step in the graph.",
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
        help="Generate research_report.md from gt_global_graph.json (LLM server must be up).",
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
        help="OpenAI-compatible API base for report (default: REPORT_OPENAI_BASE, else GT_OPENAI_BASE, else http://localhost:8000/v1).",
    )
    p.add_argument(
        "--report-model",
        default=None,
        help="Model name for report (default: REPORT_MODEL_NAME, else GT_LLM_MODEL, else llm).",
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
    p.add_argument(
        "--upload-codebook-review",
        action="store_true",
        help="Upload LLM codebook (v1) to Supabase for human review.",
    )
    p.add_argument(
        "--wait-codebook-review",
        action="store_true",
        help="Poll Supabase until codebook review is approved; materialize local artifacts.",
    )
    p.add_argument(
        "--fetch-codebook-review",
        action="store_true",
        help="One-shot fetch of approved codebook review from Supabase.",
    )
    p.add_argument(
        "--resume-codebook-review",
        action="store_true",
        help="Resume LangGraph checkpoint after codebook review approval (interrupt mode).",
    )
    p.add_argument(
        "--skip-codebook-review",
        action="store_true",
        help="Bypass codebook review gate (automated runs).",
    )
    p.add_argument(
        "--enrich-codebook-only",
        action="store_true",
        help="Run cluster qualitative enrichment (definition, criteria, examples). Requires codebook.json and open-codes markdown. LLM must be up.",
    )
    p.add_argument(
        "--enrich-dimensions-only",
        action="store_true",
        help="Run meta-theme qualitative enrichment. Requires gt_meta_themes.json. LLM must be up.",
    )
    p.add_argument(
        "--ground-enriched-only",
        action="store_true",
        help="Ground string examples in codebook_enriched / meta_themes_enriched to snippet objects (no LLM).",
    )
    p.add_argument(
        "--rebuild-source-memory-only",
        action="store_true",
        help="Rebuild gt_source_memory.json from open-codes markdown and data CSV.",
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
    log_step(
        "OPEN_CODING_VALIDATOR",
        "enabled" if USE_OPEN_CODES_VALIDATOR else "disabled",
    )
    from agents.core.llm_clustering import use_llm_clustering

    log_step(
        "AXIAL_CLUSTERING",
        "llm" if use_llm_clustering() else "embedding",
    )
    log_step(
        "QUALITATIVE_ENRICHMENT",
        "enabled" if qualitative_enrichment_enabled() else "disabled",
    )

    if args.upload_codebook_review:
        slug = os.environ.get("PIPELINE_SLUG", "default").strip() or "default"
        try:
            review_id = stage_v1_for_review(slug, rq, meta=review_meta_from_env())
        except RuntimeError as e:
            print(f"Error: {e}", file=sys.stderr)
            raise SystemExit(1)
        log_step("CODEBOOK_REVIEW_UPLOADED", f"review_id={review_id}")
        raise SystemExit(0)

    if args.wait_codebook_review:
        slug = os.environ.get("PIPELINE_SLUG", "default").strip() or "default"
        review_id = os.environ.get("CODEBOOK_REVIEW_ID", "").strip() or None
        try:
            wait_for_approval(slug, review_id=review_id)
        except (TimeoutError, RuntimeError) as e:
            print(f"Error: {e}", file=sys.stderr)
            raise SystemExit(1)
        log_step("CODEBOOK_REVIEW_APPROVED", f"slug={slug!r}")
        raise SystemExit(0)

    if args.fetch_codebook_review:
        slug = os.environ.get("PIPELINE_SLUG", "default").strip() or "default"
        review_id = os.environ.get("CODEBOOK_REVIEW_ID", "").strip() or None
        try:
            fetch_and_materialize_approved(review_id=review_id, slug=slug)
        except RuntimeError as e:
            print(f"Error: {e}", file=sys.stderr)
            raise SystemExit(1)
        log_step("CODEBOOK_REVIEW_FETCHED", f"slug={slug!r}")
        raise SystemExit(0)

    if args.resume_codebook_review:
        from langgraph.types import Command

        review_id = os.environ.get("CODEBOOK_REVIEW_ID", "").strip() or None
        resume_payload = {"review_id": review_id} if review_id else {}
        state = _base_state(rq)
        state["axial_mapping"] = "refine"
        state["codebook_review_status"] = "approved"
        with open(CODEBOOK_PATH, encoding="utf-8") as f:
            existing_cb = json.load(f)
        state["codebook"] = existing_cb.get("codebook", {})
        try:
            final_refine = app.invoke(Command(resume=resume_payload), config=_graph_config())
        except Exception as e:
            log_step("CODEBOOK_REVIEW_RESUME_FALLBACK", f"checkpoint resume failed: {e}")
            final_refine = app.invoke(state, config=_graph_config("resume-fallback"))
        log_step(
            "REFINE_COMPLETE",
            "Resumed after codebook review."
            if final_refine.get("_cluster_refinement_done")
            else "Resume completed.",
        )
        raise SystemExit(0)

    if args.high_level_only:
        if not CLUSTERED_CODES_PATH.is_file():
            print(f"Error: {display_path(CLUSTERED_CODES_PATH)} not found. Run axial step first.")
            raise SystemExit(1)
        state = _base_state(rq)
        use_interrupt = human_review_enabled() and review_mode() == "interrupt"
        state["axial_mapping"] = "refine" if use_interrupt else "done"
        final_hl = app.invoke(state, config=_graph_config("high-level"))
        codebook = final_hl.get("codebook") or {}
        log_step(
            "CODEBOOK_COMPLETE",
            f"Generated {len(codebook)} high-level labels. See {display_path(CODEBOOK_PATH)}",
        )
        if human_review_enabled() and not use_interrupt:
            slug = os.environ.get("PIPELINE_SLUG", "default").strip() or "default"
            try:
                review_id = stage_v1_for_review(slug, rq, meta=review_meta_from_env())
                log_step("CODEBOOK_REVIEW_UPLOADED", f"review_id={review_id}")
            except RuntimeError as e:
                print(f"Warning: codebook review upload failed: {e}", file=sys.stderr)
        raise SystemExit(0)

    if args.refine_only:
        _require_approved_codebook(args.skip_codebook_review)
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
        if args.skip_codebook_review:
            state["codebook_review_status"] = "skipped"
        elif CODEBOOK_PROVENANCE_PATH.is_file():
            state["codebook_review_status"] = "approved"
        final_refine = app.invoke(state, config=_graph_config("refine"))
        summary = final_refine.get("_cluster_refinement_done", False)
        log_step(
            "REFINE_COMPLETE",
            "Cluster refinement finished." if summary else "Refine step completed.",
        )
        raise SystemExit(0)

    if args.enrich_codebook_only:
        _require_approved_codebook(args.skip_codebook_review)
        if not CODEBOOK_PATH.is_file():
            print(f"Error: {display_path(CODEBOOK_PATH)} not found. Run refine step first.")
            raise SystemExit(1)
        if not OPEN_CODES_MARKDOWN_PATH.is_file():
            print(
                f"Error: {display_path(OPEN_CODES_MARKDOWN_PATH)} not found. Run open coding first.",
                file=sys.stderr,
            )
            raise SystemExit(1)
        try:
            enriched = run_cluster_qualitative_enrichment()
        except (FileNotFoundError, ValueError, RuntimeError) as e:
            print(f"Error: {e}", file=sys.stderr)
            raise SystemExit(1)
        log_step(
            "ENRICH_CODEBOOK_COMPLETE",
            f"Enriched {len(enriched)} clusters. See {display_path(CODEBOOK_PATH)}",
        )
        raise SystemExit(0)

    if args.enrich_dimensions_only:
        if not META_THEMES_PATH.is_file():
            print(
                f"Error: {display_path(META_THEMES_PATH)} not found. Run meta-themes step first."
            )
            raise SystemExit(1)
        try:
            dims = run_dimension_qualitative_enrichment()
        except (FileNotFoundError, ValueError, RuntimeError) as e:
            print(f"Error: {e}", file=sys.stderr)
            raise SystemExit(1)
        log_step(
            "ENRICH_DIMENSIONS_COMPLETE",
            f"Enriched {len(dims)} dimensions. See {display_path(META_THEMES_ENRICHED_PATH)}",
        )
        raise SystemExit(0)

    if args.ground_enriched_only:
        if not OPEN_CODES_MARKDOWN_PATH.is_file():
            print(
                f"Error: {display_path(OPEN_CODES_MARKDOWN_PATH)} not found. Run open coding first.",
                file=sys.stderr,
            )
            raise SystemExit(1)
        if not CODEBOOK_PATH.is_file() and not META_THEMES_ENRICHED_PATH.is_file():
            print(
                "Error: no enriched artifacts found (codebook.json or gt_meta_themes_enriched.json).",
                file=sys.stderr,
            )
            raise SystemExit(1)
        try:
            run_ground_enriched_only(csv_path=Path(args.data))
        except (FileNotFoundError, ValueError, RuntimeError) as e:
            print(f"Error: {e}", file=sys.stderr)
            raise SystemExit(1)
        log_step("GROUND_ENRICHED_COMPLETE", "Grounded examples updated in place.")
        raise SystemExit(0)

    if args.rebuild_source_memory_only:
        if not OPEN_CODES_MARKDOWN_PATH.is_file():
            print(
                f"Error: {display_path(OPEN_CODES_MARKDOWN_PATH)} not found. Run open coding first.",
                file=sys.stderr,
            )
            raise SystemExit(1)
        mem = SourceMemory.build(OPEN_CODES_MARKDOWN_PATH, Path(args.data))
        mem.save(SOURCE_MEMORY_PATH)
        log_step(
            "SOURCE_MEMORY_REBUILT",
            f"Wrote {len(mem.snippets)} snippets to {display_path(SOURCE_MEMORY_PATH)}",
        )
        raise SystemExit(0)

    if args.hierarchy_only:
        if not CODEBOOK_PATH.is_file():
            print(f"Error: {display_path(CODEBOOK_PATH)} not found. Run high-level step first.")
            raise SystemExit(1)
        state = _base_state(rq)
        state["axial_mapping"] = "hierarchy"
        final_hier = app.invoke(state, config=_graph_config("hierarchy"))
        log_step("HIERARCHY_COMPLETE", final_hier.get("hierarchy", ""))
        raise SystemExit(0)

    if args.meta_themes_only:
        if not CODEBOOK_PATH.is_file():
            print(f"Error: {display_path(CODEBOOK_PATH)} not found. Run high-level step first.")
            raise SystemExit(1)
        state = _base_state(rq)
        state["axial_mapping"] = "meta_themes"
        final_mt = app.invoke(state, config=_graph_config("meta-themes"))
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
        final_tree = app.invoke(state, config=_graph_config("tree"))
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

    if args.axial_only:
        with open(GT_CODES_ONLY_PATH, encoding="utf-8") as f:
            data = json.load(f)
        state = _base_state(rq)
        state["all_codes_for_axial"] = data["all_codes"]
        final_axial = app.invoke(state, config=_graph_config("axial"))
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
    all_open_codes = []

    def _code_one(idx_review):
        idx, review = idx_review
        state = _base_state(rq)
        state["raw_text"] = review
        final_state = app.invoke(state, config=_graph_config(f"review-{idx}"))
        return idx, final_state.get("open_codes", "")

    workers = int(os.environ.get("GT_OPEN_CODING_WORKERS", "8"))
    results: dict = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(_code_one, (idx, rev)): idx for idx, rev in enumerate(reviews, start=1)
        }
        for fut in as_completed(futures):
            idx, codes = fut.result()
            results[idx] = codes
            print(
                f"      [{len(results)}/{len(reviews)}] open coding done: review {idx}", flush=True
            )
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

    mem = SourceMemory.build(OPEN_CODES_MARKDOWN_PATH, Path(args.data))
    mem.save(SOURCE_MEMORY_PATH)
    log_step(
        "SOURCE_MEMORY_BUILT",
        f"Indexed {len(mem.snippets)} snippets. See {display_path(SOURCE_MEMORY_PATH)}",
    )

    if args.open_coding_only:
        log_step("OPEN_CODING_ONLY", "Exiting so SGLang can be killed before axial.")
        raise SystemExit(0)

    state = _base_state(rq)
    state["all_codes_for_axial"] = all_codes
    final_axial = app.invoke(state, config=_graph_config("axial"))
    axial_mapping = final_axial.get("axial_mapping", "")
    log_step(
        "AXIAL_COMPLETE", axial_mapping[:500] + "..." if len(axial_mapping) > 500 else axial_mapping
    )

    state = _base_state(rq)
    state["axial_mapping"] = "refine"
    final_refine = app.invoke(state, config=_graph_config("refine"))
    codebook = final_refine.get("codebook") or {}
    log_step(
        "CODEBOOK_REFINE_COMPLETE",
        f"High-level and refinement done. {len(codebook)} clusters. See {display_path(CODEBOOK_PATH)}",
    )


if __name__ == "__main__":
    main()
