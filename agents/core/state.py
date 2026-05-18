"""State, routing, and tool dispatch for the GT pipeline."""

import json
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END

from . import utils as _utils
from .paths import CLUSTERED_CODES_PATH, CODEBOOK_PATH
from .tools import (
    axial_coding,
    hierarchy_construction,
    high_level_code_generation,
    meta_theme_grouping,
    open_coding,
    refine_cluster_assignments,
    tree_assembly,
    validate_open_codes,
)
from .utils import log_step, remove_think_tags

# Max retries when validator returns FAIL (open coding)
OPEN_CODING_MAX_RETRIES = 2

# Name -> callable; tool_node uses this to invoke the right tool from state["tool_call"]
TOOLS = {
    "open_coding": open_coding,
    "validate_open_codes": validate_open_codes,
    "axial_coding": axial_coding,
    "high_level_code_generation": high_level_code_generation,
    "refine_cluster_assignments": refine_cluster_assignments,
    "hierarchy_construction": hierarchy_construction,
    "meta_theme_grouping": meta_theme_grouping,
    "tree_assembly": tree_assembly,
}


class GTState(TypedDict, total=False):
    """Schema for graph state; all keys optional (total=False). LangGraph merges node outputs into this."""

    # Inputs
    research_question: str
    raw_text: str
    # Open coding (per-review)
    open_codes: Optional[str]
    open_codes_validation: Optional[str]  # "PASS" | "FAIL"
    open_codes_validation_feedback: Optional[str]
    _open_coding_retries: int
    # Axial phase: codes -> clusters
    all_codes_for_axial: Optional[List[str]]
    axial_mapping: Optional[
        str
    ]  # cluster summary text, or "done"|"refine"|"hierarchy"|"meta_themes"|"tree"
    _cluster_refinement_done: Optional[bool]
    # Downstream outputs
    codebook: Optional[Dict[str, str]]
    hierarchy: Optional[str]
    meta_themes: Optional[str]
    global_graph: Optional[str]
    # Control
    tool_call: Optional[
        Dict[str, Any]
    ]  # {"tool": name, "args": {...}}; agent sets, tool_node clears
    step: int
    _sim_threshold: float
    _skip_cross_cluster: bool


def agent_node(state: GTState):
    """Agent node: deterministic orchestrator; decides which tool to call (and args) from current state."""
    step = state.get("step", 0) + 1
    rq = state.get("research_question", "")
    retries = state.get("_open_coding_retries", 0)

    # --- Open coding: retry with feedback when validator said FAIL ---
    if (
        state.get("open_codes_validation") == "FAIL"
        and retries < OPEN_CODING_MAX_RETRIES
        and state.get("raw_text")
    ):
        feedback = state.get("open_codes_validation_feedback") or ""
        return {
            "tool_call": {
                "tool": "open_coding",
                "args": {
                    "text": state["raw_text"],
                    "research_question": rq,
                    "validator_feedback": feedback,
                },
            },
            "step": step,
            "open_codes_validation": None,
            "open_codes_validation_feedback": None,
            "_open_coding_retries": retries + 1,
        }
    # --- Open coding: after codes exist, run validator (if not yet run) ---
    if (
        state.get("open_codes")
        and state.get("open_codes_validation") is None
        and not state.get("all_codes_for_axial")
    ):
        return {
            "tool_call": {
                "tool": "validate_open_codes",
                "args": {
                    "text": state["raw_text"],
                    "generated_codes": state["open_codes"],
                    "research_question": rq,
                },
            },
            "step": step,
        }
    # --- Open coding: first call for this review (no feedback) ---
    if not state.get("open_codes") and state.get("raw_text"):
        return {
            "tool_call": {
                "tool": "open_coding",
                "args": {"text": state["raw_text"], "research_question": rq},
            },
            "step": step,
        }

    # --- Axial phase: first axial step (embed + cluster) ---
    axial_codes = state.get("all_codes_for_axial")
    if axial_codes is not None and not state.get("axial_mapping"):
        return {
            "tool_call": {
                "tool": "axial_coding",
                "args": {"open_codes": json.dumps(axial_codes)},
            },
            "step": step,
        }
    # --- Refine phase: high-level then refine_cluster_assignments ---
    if state.get("axial_mapping") == "refine" and not state.get("codebook"):
        return {
            "tool_call": {
                "tool": "high_level_code_generation",
                "args": {"cluster_file": str(CLUSTERED_CODES_PATH), "research_question": rq},
            },
            "step": step,
        }
    if (
        state.get("axial_mapping") == "refine"
        and state.get("codebook")
        and not state.get("_cluster_refinement_done")
    ):
        return {
            "tool_call": {
                "tool": "refine_cluster_assignments",
                "args": {
                    "codebook_path": str(CODEBOOK_PATH),
                    "cluster_file": str(CLUSTERED_CODES_PATH),
                },
            },
            "step": step,
        }
    # --- After axial: high-level labels (sentinel "done"), then hierarchy, meta_themes, tree ---
    if state.get("axial_mapping") == "done" and not state.get("codebook"):
        return {
            "tool_call": {
                "tool": "high_level_code_generation",
                "args": {"cluster_file": str(CLUSTERED_CODES_PATH), "research_question": rq},
            },
            "step": step,
        }
    if state.get("axial_mapping") == "hierarchy" and not state.get("hierarchy"):
        return {
            "tool_call": {"tool": "hierarchy_construction", "args": {"research_question": rq}},
            "step": step,
        }
    if state.get("axial_mapping") == "meta_themes" and not state.get("meta_themes"):
        return {
            "tool_call": {"tool": "meta_theme_grouping", "args": {"research_question": rq}},
            "step": step,
        }
    if state.get("axial_mapping") == "tree" and not state.get("global_graph"):
        return {
            "tool_call": {
                "tool": "tree_assembly",
                "args": {"research_question": rq},
            },
            "step": step,
        }
    # No tool to run; return step only (router will send to END or agent depending on state)
    return {"step": step}


def _parse_validation_output(output: str) -> tuple:
    """Parse validator output to (PASS or FAIL, feedback text)."""
    cleaned = output.strip().upper()
    if cleaned.startswith("PASS"):
        return "PASS", output.strip()
    return "FAIL", output.strip()


def router(state: GTState):
    """Router: returns 'tool' | END | 'agent' so LangGraph knows next node. Called after agent_node."""
    # Agent set a tool_call -> run the tool
    if state.get("tool_call"):
        return "tool"
    # Open-coding phase: waiting for validation or done
    if state.get("open_codes") and state.get("all_codes_for_axial") is None:
        validation = state.get("open_codes_validation")
        retries = state.get("_open_coding_retries", 0)
        if validation is None:
            return "agent"  # shouldn't happen; agent would have sent to tool
        if validation == "PASS":
            return END
        if validation == "FAIL" and retries < OPEN_CODING_MAX_RETRIES:
            return "agent"  # agent will schedule open_coding retry
        return END
    # Axial phase (long text): axial done, no validation step
    if state.get("axial_mapping") and state.get("axial_mapping") not in (
        "done",
        "refine",
        "hierarchy",
        "meta_themes",
        "tree",
    ):
        return END
    # Refine phase: run high_level then refine_cluster_assignments, then END
    if state.get("axial_mapping") == "refine":
        if state.get("_cluster_refinement_done"):
            return END
        return "agent"
    # Downstream: more work to do -> agent; else END
    # axial_mapping == "done" is legacy/compat for --high-level-only (bypasses refine); normal flow uses "refine"
    if state.get("axial_mapping") == "done" and not state.get("codebook"):
        return "agent"
    if state.get("codebook"):
        return END
    if state.get("axial_mapping") == "hierarchy" and not state.get("hierarchy"):
        return "agent"
    if state.get("hierarchy"):
        return END
    if state.get("axial_mapping") == "meta_themes" and not state.get("meta_themes"):
        return "agent"
    if state.get("meta_themes"):
        return END
    if state.get("axial_mapping") == "tree" and not state.get("global_graph"):
        return "agent"
    if state.get("global_graph"):
        return END
    return "agent"


def tool_node(state: GTState):
    """Tool node: runs the tool from state['tool_call'], maps output to state updates, clears tool_call."""
    call = state["tool_call"]
    tool_name = call["tool"]

    _utils._active_tool = tool_name
    _utils._active_step = state.get("step")
    try:
        raw_output = TOOLS[tool_name].invoke(call["args"])
    finally:
        _utils._active_tool = None
        _utils._active_step = None
    clean_output = remove_think_tags(raw_output)

    log_step(f"TOOL_OUTPUT ({tool_name})", clean_output)

    updates = {"tool_call": None}  # always clear so router doesn't re-dispatch

    if tool_name == "open_coding":
        # Empty model output would skip validation and re-dispatch open_coding forever
        if not clean_output.strip():
            clean_output = (
                "- Applicability: NONE\n"
                "  Reason: Coder returned no text; cannot validate substantive codes.\n"
                '  Evidence: "(empty output)"\n'
            )
        updates["open_codes"] = clean_output
    elif tool_name == "validate_open_codes":
        verdict, feedback = _parse_validation_output(clean_output)
        updates["open_codes_validation"] = verdict
        updates["open_codes_validation_feedback"] = feedback
    elif tool_name == "axial_coding":
        updates["axial_mapping"] = clean_output
    elif tool_name == "high_level_code_generation":
        try:
            updates["codebook"] = json.loads(clean_output)
        except json.JSONDecodeError:
            updates["codebook"] = {}  # fallback if LLM didn't return valid JSON
    elif tool_name == "refine_cluster_assignments":
        updates["_cluster_refinement_done"] = True
    elif tool_name == "hierarchy_construction":
        updates["hierarchy"] = clean_output
    elif tool_name == "meta_theme_grouping":
        updates["meta_themes"] = clean_output
    elif tool_name == "tree_assembly":
        updates["global_graph"] = clean_output
    return updates
