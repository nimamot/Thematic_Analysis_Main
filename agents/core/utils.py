"""Helpers for the GT pipeline: logging, text parsing, code extraction, LLM token usage."""

import datetime
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional

from .paths import GT_AGENT_TRACE_LOG_PATH, LLM_USAGE_PATH, ensure_output_dirs

LOG_FILE = GT_AGENT_TRACE_LOG_PATH

_active_tool: Optional[str] = None
_active_step: Optional[int] = None
_usage_warned: bool = False


def log_step(step_name: str, content: Any):
    ensure_output_dirs()
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    entry = f"\n[{timestamp}] === {step_name} ===\n{str(content)}\n" + "=" * 30
    print(entry)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(entry)


def remove_think_tags(text: str) -> str:
    """Removes <think> blocks from reasoning models."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned.strip()


def clean_and_parse_json(text: str):
    """Clean thoughts and extract JSON safely."""
    text_no_think = remove_think_tags(text)
    start_idx = text_no_think.find("{")
    end_idx = text_no_think.rfind("}")
    if start_idx == -1 or end_idx == -1:
        raise ValueError("No JSON brackets found.")
    return json.loads(text_no_think[start_idx : end_idx + 1])


_CODE_LINE_RE = re.compile(
    r"^[-*]?\s*Code:\s*(.+)$",
    re.MULTILINE | re.IGNORECASE,
)


def extract_codes(open_coding_text: str) -> list[str]:
    """
    Extract code labels only from open coding output (no Evidence/Note).
    Accepts both "- Code: <code>" (single-review / prompt) and "Code: <code>" (batch path).
    """
    if not open_coding_text or not open_coding_text.strip():
        return []
    codes = _CODE_LINE_RE.findall(open_coding_text)
    return [c.strip() for c in codes if c.strip()]


def parse_open_codes_markdown(text: str) -> list[tuple[int, str]]:
    """Parse ## Review N sections from gt_open_codes_all_reviews.md."""
    parts = re.split(r"^## Review (\d+)\s*$", text, flags=re.MULTILINE)
    out: list[tuple[int, str]] = []
    i = 1
    while i + 1 < len(parts):
        out.append((int(parts[i]), parts[i + 1].strip()))
        i += 2
    return out


def _extract_usage(ai_message: Any) -> Dict[str, Any]:
    """
    Pull token usage from a LangChain AIMessage. Returns dict with
    prompt_tokens, completion_tokens, total_tokens, model, usage_missing.

    Order of attempts:
    1. ai_message.usage_metadata (newer langchain_core: input_tokens / output_tokens / total_tokens)
    2. ai_message.response_metadata.token_usage (older: prompt_tokens / completion_tokens / total_tokens)
    """
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    model: Optional[str] = None

    um = getattr(ai_message, "usage_metadata", None)
    if isinstance(um, dict):
        prompt_tokens = um.get("input_tokens")
        completion_tokens = um.get("output_tokens")
        total_tokens = um.get("total_tokens")

    rm = getattr(ai_message, "response_metadata", None)
    if isinstance(rm, dict):
        if model is None:
            model = rm.get("model_name") or rm.get("model")
        if prompt_tokens is None or completion_tokens is None:
            tu = rm.get("token_usage")
            if isinstance(tu, dict):
                prompt_tokens = (
                    prompt_tokens if prompt_tokens is not None else tu.get("prompt_tokens")
                )
                completion_tokens = (
                    completion_tokens
                    if completion_tokens is not None
                    else tu.get("completion_tokens")
                )
                total_tokens = total_tokens if total_tokens is not None else tu.get("total_tokens")

    usage_missing = prompt_tokens is None and completion_tokens is None
    pt = int(prompt_tokens) if isinstance(prompt_tokens, (int, float)) else 0
    ct = int(completion_tokens) if isinstance(completion_tokens, (int, float)) else 0
    if total_tokens is None:
        tt = pt + ct
    else:
        tt = int(total_tokens) if isinstance(total_tokens, (int, float)) else (pt + ct)

    return {
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "total_tokens": tt,
        "model": model,
        "usage_missing": usage_missing,
    }


def record_llm_usage(
    skill: str,
    ai_message: Any,
    latency_ms: Optional[float] = None,
    labels: Optional[Dict[str, Any]] = None,
    jsonl_path: Path = LLM_USAGE_PATH,
) -> Dict[str, Any]:
    """Append one JSON line with token usage for an LLM call. Pulls active tool/step from module globals."""
    global _usage_warned
    usage = _extract_usage(ai_message)

    record = {
        "ts": datetime.datetime.now().isoformat(timespec="seconds"),
        "step": _active_step,
        "tool": _active_tool,
        "skill": skill,
        "model": usage.get("model"),
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "total_tokens": usage["total_tokens"],
        "latency_ms": round(latency_ms, 2) if isinstance(latency_ms, (int, float)) else None,
        "labels": dict(labels) if labels else {},
    }
    if usage.get("usage_missing"):
        record["usage_missing"] = True
        if not _usage_warned:
            _usage_warned = True
            log_step(
                "LLM_USAGE_WARNING",
                f"AIMessage from skill '{skill}' had no usage_metadata or token_usage; recording zeros.",
            )

    ensure_output_dirs()
    try:
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as e:
        log_step("LLM_USAGE_WRITE_ERROR", f"Failed to append to {jsonl_path}: {e}")

    return record


def summarize_llm_usage(jsonl_path: Path = LLM_USAGE_PATH) -> str:
    """Read the JSONL and produce a small text rollup: grand total + per tool / per step / per skill."""
    if not Path(jsonl_path).is_file():
        return "No LLM usage recorded (file not found)."

    events = []
    try:
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError as e:
        return f"Failed to read {jsonl_path}: {e}"

    if not events:
        return "No LLM usage recorded (file empty)."

    def _zeros() -> Dict[str, int]:
        return {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    grand = _zeros()
    by_tool: Dict[str, Dict[str, int]] = defaultdict(_zeros)
    by_step: Dict[str, Dict[str, int]] = defaultdict(_zeros)
    by_skill: Dict[str, Dict[str, int]] = defaultdict(_zeros)

    for ev in events:
        pt = int(ev.get("prompt_tokens") or 0)
        ct = int(ev.get("completion_tokens") or 0)
        tt = int(ev.get("total_tokens") or (pt + ct))
        for bucket in (
            grand,
            by_tool[str(ev.get("tool"))],
            by_step[str(ev.get("step"))],
            by_skill[str(ev.get("skill"))],
        ):
            bucket["calls"] += 1
            bucket["prompt_tokens"] += pt
            bucket["completion_tokens"] += ct
            bucket["total_tokens"] += tt

    lines = [
        f"Total LLM calls: {grand['calls']}",
        f"  prompt_tokens={grand['prompt_tokens']} completion_tokens={grand['completion_tokens']} total_tokens={grand['total_tokens']}",
        "",
        "By tool:",
    ]
    for k in sorted(by_tool.keys()):
        v = by_tool[k]
        lines.append(
            f"  {k}: calls={v['calls']} in={v['prompt_tokens']} out={v['completion_tokens']} total={v['total_tokens']}"
        )
    lines.append("")
    lines.append("By step:")

    def _step_sort_key(s: str):
        try:
            return (0, int(s))
        except (TypeError, ValueError):
            return (1, s)

    for k in sorted(by_step.keys(), key=_step_sort_key):
        v = by_step[k]
        lines.append(
            f"  step={k}: calls={v['calls']} in={v['prompt_tokens']} out={v['completion_tokens']} total={v['total_tokens']}"
        )
    lines.append("")
    lines.append("By skill:")
    for k in sorted(by_skill.keys()):
        v = by_skill[k]
        lines.append(
            f"  {k}: calls={v['calls']} in={v['prompt_tokens']} out={v['completion_tokens']} total={v['total_tokens']}"
        )
    return "\n".join(lines)
