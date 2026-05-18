"""
Load long-term “skills” from markdown files and inject them into every LLM call.

Skills are stable behavioral instructions (one per phase/tool) that are independent
of dataset/question wording.
"""

from __future__ import annotations

import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from .utils import record_llm_usage


def _truthy(v: str | None) -> bool:
    if v is None:
        return False
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def _skills_dir() -> Path:
    return Path(
        os.environ.get("GT_SKILLS_DIR", str(Path(__file__).resolve().parent.parent / "skills"))
    ).resolve()


def _strip_yaml_frontmatter(text: str) -> str:
    """
    If file starts with YAML frontmatter like:
    ---
    ...
    ---
    content...
    ---
    return only the content part (so we don't inject YAML into the model).
    """
    if not isinstance(text, str):
        return ""
    t = text.lstrip()
    if not t.startswith("---"):
        return text.strip()
    # Work with original string indices to avoid changing content unexpectedly.
    # Find first "----" line and second delimiter line.
    lines = t.splitlines(True)
    if not lines:
        return ""
    # First line should be the opening delimiter.
    if not lines[0].strip().startswith("---"):
        return text.strip()
    # Find closing delimiter line (a line that is exactly --- or --- with whitespace).
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            return "".join(lines[i + 1 :]).strip()
    # No closing delimiter: fall back to full text.
    return text.strip()


@lru_cache(maxsize=128)
def load_skill_text(skill_key: str) -> str:
    """
    Load `agents/skills/{skill_key}.md`.
    Returns empty string when missing or disabled.
    """
    if not _truthy(os.environ.get("GT_USE_SKILLS", "1")):
        return ""

    if not skill_key or not isinstance(skill_key, str):
        return ""

    path = _skills_dir() / f"{skill_key}.md"
    if not path.is_file():
        return ""
    raw = path.read_text(encoding="utf-8")
    return _strip_yaml_frontmatter(raw)


def llm_invoke_with_skill(
    llm,
    skill_key: str,
    human_prompt: str,
    *,
    max_tokens: int | None = None,
    **labels: Any,
) -> str:
    """
    Invoke ChatOpenAI with skills as a SystemMessage + HumanMessage (langchain_core).

    When skills are disabled or the skill file is missing, invokes with a plain string only
    (no system role). That path does not use the fallback concatenation.

    Token usage for the resulting AIMessage is recorded via record_llm_usage; any kwargs
    (e.g. cluster_id, chunk_index, phase) are forwarded as labels on the recorded event.
    """
    sys_text = load_skill_text(skill_key)
    human_prompt = human_prompt if human_prompt is not None else ""
    arg = (
        human_prompt
        if not sys_text
        else [SystemMessage(content=sys_text), HumanMessage(content=human_prompt)]
    )

    invoke_kwargs: dict[str, Any] = {}
    if max_tokens is not None:
        invoke_kwargs["max_tokens"] = max_tokens

    t0 = time.monotonic()
    ai = llm.invoke(arg, **invoke_kwargs)
    latency_ms = (time.monotonic() - t0) * 1000.0
    record_llm_usage(skill_key, ai, latency_ms=latency_ms, labels=labels or None)
    return (ai.content or "").strip()
