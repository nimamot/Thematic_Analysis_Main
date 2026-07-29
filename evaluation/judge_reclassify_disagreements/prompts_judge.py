"""Prompt templates for the automated LLM-judge adjudication of reclassify
disagreements -- binary (2-choice) verdict between the pipeline's original
assignment and an independent reclassifier's alternative pick, blind (the
judge is never told which candidate is which), mirroring
human_eval_survey_reclassify_disagreement/'s design exactly so the two are
directly comparable. No "reasoning" field is requested (see
reclassify_pipeline.py's classify_batch -- a requested reasoning field
previously caused verbose models to blow the output token budget on a
multi-item batch and truncate mid-response).
"""

JUDGE_SYSTEM = """\
You are a qualitative research expert judging, quote by quote, which of two \
candidate themes a participant review quote fits better. Each candidate is \
defined by a label, a definition, and inclusion/exclusion criteria written \
by a separate process -- you are not told where either candidate came from, \
or which one (if either) was the "original" categorization. For EACH quote \
in the batch, pick whichever candidate -- Option A or Option B -- the quote \
fits better, based only on the written definition and criteria. You must \
pick one; there is no "neither" option here."""

JUDGE_USER_TEMPLATE = """\
Below are {n} independent quote-judging questions. Each has a quote and two \
candidate themes (Option A / Option B), each with its own label, \
definition, and inclusion/exclusion criteria.

{items_text}

---
For EACH question above, decide whether the quote fits Option A or Option B \
better.

Respond in JSON, one entry per question index:
{{
  "verdicts": [
    {{"item_index": 1, "pick": "A"}},
    {{"item_index": 2, "pick": "B"}},
    ...
  ]
}}"""


def _format_option(opt: dict) -> str:
    lines = [f'    LABEL: {opt.get("label", "")}', f'    DEFINITION: {opt.get("definition", "")}']
    inclusion = opt.get("inclusion") or []
    exclusion = opt.get("exclusion") or []
    if inclusion:
        lines.append("    INCLUSION CRITERIA:")
        lines.extend(f"      - {c}" for c in inclusion)
    if exclusion:
        lines.append("    EXCLUSION CRITERIA:")
        lines.extend(f"      - {c}" for c in exclusion)
    return "\n".join(lines)


def format_item(idx: int, item: dict) -> str:
    return (
        f'{idx}. QUOTE: "{item["quote"]}"\n'
        f'  Option A:\n{_format_option(item["option_a"])}\n'
        f'  Option B:\n{_format_option(item["option_b"])}'
    )


def build_user_prompt(items: list[dict]) -> str:
    items_text = "\n\n".join(format_item(i + 1, it) for i, it in enumerate(items))
    return JUDGE_USER_TEMPLATE.format(n=len(items), items_text=items_text)
