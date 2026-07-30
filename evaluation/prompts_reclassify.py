"""Prompt templates for the CODEBOOK RECLASSIFICATION reliability test.

Different question from both the CRB challenge pipeline (prompts.py) and the
coverage pipeline (prompts_coverage.py). Those test whether a codebook holds
up against adversarial critique. This tests something more basic: if you
show an INDEPENDENT model only the codebook's rules and a raw review quote,
can it reclassify that quote into the SAME meta-theme the original pipeline
actually assigned it to? High agreement = the codebook's rules are
clear/operational enough for an independent coder to reproduce the same
categorization. Low agreement = the rules alone don't sufficiently constrain
the classification (relies on information not captured in the written
criteria, e.g. embedded examples, tacit judgment, etc).

Classification is done in BATCHES (many quotes per call, same codebook
reused across the batch) since a single run can have 600-800 raw quotes --
one-call-per-quote would be prohibitively expensive across all 40 runs.

detail_level controls how much of the codebook the classifier is shown (see
reclassify_pipeline.format_codebook's DETAIL_LEVELS) -- for the advisor's
ablation experiment. The prompt text MUST describe only what's actually
given for that level: earlier versions unconditionally told every model
"apply the definition and inclusion/exclusion criteria as written" even in
theme_only/theme_def runs where no such criteria were ever shown -- some
models (esp. gemma-4) took that literally and refused to answer, asking the
caller to supply the missing definition/criteria instead of classifying,
which showed up as a wall of deterministic (temperature=0) parse failures
that no amount of retrying could fix, since the instruction was self-
contradictory rather than the model's raw output being malformed.
"""

_BASIS_TEXT = {
    "theme_only": "the category LABEL",
    "theme_def": "the category LABEL and DEFINITION",
    "full": "the written LABEL, DEFINITION, and INCLUSION/EXCLUSION criteria",
}


def reclassify_system(detail_level: str = "full", include_reasoning: bool = False) -> str:
    basis = _BASIS_TEXT[detail_level]
    reasoning_line = (
        "\nWrite the reasoning field in plain, conversational language -- avoid "
        "academic jargon and noun-heavy phrasing."
        if include_reasoning else ""
    )
    return f"""\
You are a qualitative coding expert applying an existing codebook to raw data.
You will be given a set of meta-theme categories and a batch of raw quotes \
from participant reviews. For EACH quote, classify it into exactly ONE of \
the meta-theme categories based ONLY on {basis} -- do not guess based on \
what a "typical" categorization might look like, and do not assume any \
information beyond what is actually given below.
If a quote genuinely does not fit any category based on the information \
given, you may respond "NONE" for that quote rather than forcing a fit.{reasoning_line}"""


def reclassify_user(codebook_text: str, quotes_text: str, detail_level: str = "full",
                     include_reasoning: bool = False) -> str:
    basis = _BASIS_TEXT[detail_level]
    # include_reasoning=False drops the per-item "reasoning" field entirely --
    # added for qwen3.5-35b-a3b specifically, which writes long enough
    # reasoning sentences that a 25-quote batch routinely blew the output
    # token budget and truncated mid-response (deterministic at temperature=0,
    # so retries kept hitting the same wall). The reasoning text was never
    # persisted to the output row anyway (see reclassify_pipeline.py), so
    # dropping it costs nothing and shrinks the output enough to stop the
    # truncation for verbose models.
    if include_reasoning:
        schema_lines = (
            '    {"quote_index": 1, "meta_theme": "<letter or NONE>", "reasoning": "<one sentence>"},\n'
            '    {"quote_index": 2, "meta_theme": "<letter or NONE>", "reasoning": "<one sentence>"},\n'
            '    ...'
        )
    else:
        schema_lines = (
            '    {"quote_index": 1, "meta_theme": "<letter or NONE>"},\n'
            '    {"quote_index": 2, "meta_theme": "<letter or NONE>"},\n'
            '    ...'
        )
    header = f"""\
META-THEME CATEGORIES:

{codebook_text}

---
QUOTES TO CLASSIFY:

{quotes_text}

---
For EACH quote above, decide which meta-theme category (by its letter, e.g. \
"A") it belongs to based on {basis}, or "NONE" if it genuinely fits no \
category.

Respond in JSON, one entry per quote index:
"""
    return header + "{\n  \"classifications\": [\n" + schema_lines + "\n  ]\n}"
