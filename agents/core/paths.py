"""Centralized filesystem paths for the GT pipeline."""

from pathlib import Path

AGENTS_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = AGENTS_DIR.parent
VIEWER_DATA_DIR = REPO_ROOT / "viewer-data"
OUTPUTS_DIR = AGENTS_DIR / "outputs"
DATA_DIR = OUTPUTS_DIR / "data"
LOGS_DIR = OUTPUTS_DIR / "logs"
WEIGHTS_DIR = AGENTS_DIR / "weights"
OLD_SLURM_DIR = AGENTS_DIR / "old_slurm.out"
# School burnout test set (see notebooks/prepare_school_burnout_text_review.ipynb)
SCHOOL_BURNOUT_TEXT_CSV = AGENTS_DIR.parent / "data" / "school_burnout_text_review.csv"
# English game/software-style reviews (column `review_text`); cli also accepts `text_review`.
REVIEW_TEXT_ENGLISH_1000_CSV = AGENTS_DIR.parent / "data" / "review_text_english_1000.csv"

REDDIT_COMMENT_TEXT_1000_CSV = AGENTS_DIR.parent / "data" / "reddit_comment_text_1000.csv"
REDDIT_COMMENT_TEXT_10000_CSV = AGENTS_DIR.parent / "data" / "reddit_comment_text_10000.csv"
REDDIT_COMMENT_TEXT_3000_CSV = AGENTS_DIR.parent / "data" / "reddit_comment_text_3000.csv"

DEFAULT_DATA_CSV = SCHOOL_BURNOUT_TEXT_CSV

# DEFAULT_DATA_CSV = REDDIT_COMMENT_TEXT_1000_CSV
# DEFAULT_DATA_CSV = REDDIT_COMMENT_TEXT_3000_CSV
# DEFAULT_DATA_CSV = REDDIT_COMMENT_TEXT_10000_CSV
# DEFAULT_DATA_CSV = REVIEW_TEXT_ENGLISH_1000_CSV
# DEFAULT_DATA_CSV = AGENTS_DIR.parent / "data" / "review_text_english_50k.csv"

GT_CODES_ONLY_PATH = DATA_DIR / "gt_codes_only.json"
CLUSTERED_CODES_PATH = DATA_DIR / "gt_clustered_codes.json"
CODEBOOK_PATH = DATA_DIR / "codebook.json"
CODEBOOK_CONFIDENCE_PATH = DATA_DIR / "codebook_confidence.json"
CODEBOOK_PROVENANCE_PATH = DATA_DIR / "codebook_provenance.json"
CODE_ID_MAP_PATH = DATA_DIR / "gt_code_id_map.json"
SOURCE_MEMORY_PATH = DATA_DIR / "gt_source_memory.json"
META_THEMES_ENRICHED_PATH = DATA_DIR / "gt_meta_themes_enriched.json"
GRAPH_CHECKPOINT_PATH = LOGS_DIR / "gt_graph_checkpoints.db"
HIERARCHY_EDGES_PATH = DATA_DIR / "gt_hierarchy_edges.jsonl"
HIERARCHY_PATH = DATA_DIR / "gt_hierarchy.json"
GRAPH_PATH = DATA_DIR / "gt_graph.json"
CROSS_CLUSTER_EDGES_PATH = DATA_DIR / "gt_cross_cluster_edges.jsonl"
META_THEMES_PATH = DATA_DIR / "gt_meta_themes.json"
GLOBAL_GRAPH_PATH = DATA_DIR / "gt_global_graph.json"
CLEANED_GLOBAL_GRAPH_PATH = DATA_DIR / "gt_global_graph_cleaned.json"
COOCCURRENCE_PATH = DATA_DIR / "gt_cooccurrence.json"
RESEARCH_REPORT_PATH = DATA_DIR / "research_report.md"
LLM_USAGE_PATH = DATA_DIR / "gt_llm_usage.jsonl"

# Mistral 7B Instruct v0.3 for post-pipeline research report (SGLang --model-path).
MISTRAL_INSTRUCT_WEIGHTS_DIR = WEIGHTS_DIR / "Mistral-7B-Instruct-v0.3"
OPEN_CODES_MARKDOWN_PATH = DATA_DIR / "gt_open_codes_all_reviews.md"
GRAPH_HTML_PATH = DATA_DIR / "gt_graph.html"

GT_AGENT_TRACE_LOG_PATH = LOGS_DIR / "gt_agent_trace.log"
SERVER_LOG_PATH = AGENTS_DIR / "server.log"


def ensure_output_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(AGENTS_DIR))
    except ValueError:
        return str(path)
