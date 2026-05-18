"""LLM prompt strings for the GT pipeline. Tools import these to keep prompt iteration separate from logic."""

import json
from typing import Any, Dict, List, Optional


def open_coding_prompt(
    research_question: str,
    text: str,
    validator_feedback: Optional[str] = None,
) -> str:
    """Build the open-coding prompt. If validator_feedback is given, include the reviewer feedback block."""
    feedback_section = ""
    if validator_feedback:
        feedback_section = f"""
A reviewer found issues with the previous codes. Use this feedback to improve:
{validator_feedback}

Revise your codes accordingly. Output the same format as below.
"""
    return f"""
You are doing Open Coding for thematic analysis on ONE user review.

Research Question: {research_question}
Focus on aspects of the review that are relevant to the research question above.

Rules:
- Produce **0 to 3** codes, depending on whether the review actually offers material that **answers the research question**.
- If the review is **off-topic**, **too thin**, or contains **nothing** that bears on the question (e.g. the question asks for negative feedback and the review only gives generic praise with no relevant substance), you must **not** invent codes — use the **Applicability: NONE** format below instead.
- When the review **does** contain relevant material, produce 1–3 codes (up to 2 for short reviews under ~50 words unless clearly richer).
- Each code must be a short noun phrase (2–6 words).
- Codes must be distinct (no near-duplicates).
- Each code must name a specific aspect AND its quality or direction — not just a neutral topic.
  Good: "laggy multiplayer matchmaking", "intuitive inventory controls", "frustrating difficulty spike"
  Bad: "matchmaking", "controls", "difficulty"
- Codes must be grounded in the review text — do not invent concepts not present.
{feedback_section}

When there is **nothing** to code for this research question, output **exactly** (no `- Code:` lines):

- Applicability: NONE
  Reason: <one or two sentences explaining why no code is appropriate>
  Evidence: "<short quote from the review, or state that the text does not address the question>"

When there **is** relevant material, output one block per code (no Applicability line):

- Code: <code>
  Evidence: "<short quote from the review>"
  Note: <one short phrase why this code fits>

Review:
{text}
"""


def validate_open_codes_prompt(
    research_question: str,
    text: str,
    generated_codes: str,
) -> str:
    """Return the reviewer prompt with research question, input text, generated codes, and PASS/FAIL instructions."""
    return f"""You are reviewing qualitative codes generated from user feedback for a research question.

Research Question: {research_question}

Input text (the review):
{text}

Coder output:
{generated_codes}

First, determine whether the coder claimed **no applicable codes** (e.g. `- Applicability: NONE` and **no** lines of the form `- Code: ...`).

If **no codes** were claimed:
- PASS only if the review truly has **no** usable, question-relevant content to code (off-topic, empty of substance for the question, or praise/generic text that does not bear on what was asked). The Reason/Evidence should be accurate.
- FAIL if the review **does** contain material that should have been coded, or if the justification is wrong or evasive.

If **one or more** `- Code:` lines are present:
1. Codes are grounded in the data (evidence in the review supports each code).
2. Codes are not duplicates or near-duplicates of each other.
3. Codes are concise concepts (short noun phrases, not vague or hallucinated).
4. Codes are evaluatively specific — each names a concrete aspect AND its quality or
   direction (e.g. "poor enemy AI behaviour", not just "enemy AI").
   A code that is a neutral topic label with no direction should be flagged as FAIL.
5. Each code is **relevant to the research question**; FAIL if a code is forced or tangential when the review does not support it.

If the output mixes `- Applicability: NONE` with `- Code:` lines, or is empty, FAIL.

Respond with exactly one of:
- PASS
or
- FAIL
Issues:
- <first issue>
- <second issue>
...

If PASS, you may add a single line of explanation after PASS. If FAIL, list specific issues so the coder can revise."""


def batch_open_coding_prompt(
    research_question: str,
    items: List[Dict[str, Any]],
) -> str:
    """Build a batched open-coding prompt. Each item: id, text, optional validator_feedback."""
    requests_json = json.dumps(items, indent=2, ensure_ascii=False)
    return f"""
You are doing Open Coding for thematic analysis on MULTIPLE user reviews in one task.

Research Question: {research_question}
Focus on aspects of each review that are relevant to the research question above.

Treat each request **independently** — do not mix evidence or codes across ids.

Rules (apply to **every** review):
- Produce **0 to 3** codes per review, depending on whether that review offers material that **answers the research question**.
- If a review is **off-topic**, **too thin**, or contains **nothing** that bears on the question, use **Applicability: NONE** for that id (no `- Code:` lines in its output).
- When a review **does** contain relevant material, produce 1–3 codes (up to 2 for short reviews under ~50 words unless clearly richer).
- Each code must be a short noun phrase (2–6 words), distinct, evaluatively specific (aspect + quality/direction), grounded in that review's text.
- If an item includes `validator_feedback`, revise that review's codes using the feedback; otherwise code from scratch.

Per-review `output` format (markdown string, same as single-review coding):

When nothing applies:
- Applicability: NONE
  Reason: <one or two sentences>
  Evidence: "<short quote>"

When coding:
- Code: <code>
  Evidence: "<short quote>"
  Note: <one short phrase>

Requests (JSON array):
{requests_json}

Output **only** valid JSON (no markdown fences, no extra text):
{{
  "responses": [
    {{"id": "<same id as request>", "output": "<markdown for that review>"}}
  ]
}}

Include one entry in `responses` for **every** request id, in any order.
"""


def batch_validate_open_codes_prompt(
    research_question: str,
    items: List[Dict[str, Any]],
) -> str:
    """Build a batched validation prompt. Each item: id, text, generated_codes."""
    requests_json = json.dumps(items, indent=2, ensure_ascii=False)
    return f"""
You are reviewing qualitative codes for MULTIPLE reviews in one task.

Research Question: {research_question}

For **each** request, evaluate the coder output for that review only.

Criteria (same as single-review validation):
- If coder used **Applicability: NONE** (no `- Code:` lines): PASS only if the review truly has no question-relevant content; FAIL if it should have been coded.
- If `- Code:` lines exist: codes must be grounded, non-duplicate, concise, evaluatively specific, and relevant to the research question.
- FAIL if output mixes NONE with `- Code:` lines, or is empty.

Requests (JSON array; each has id, text, generated_codes):
{requests_json}

Output **only** valid JSON:
{{
  "responses": [
    {{"id": "<same id>", "verdict": "PASS" or "FAIL", "feedback": "<issues for FAIL; brief note or empty for PASS>"}}
  ]
}}

Include one entry per request id. Use verdict exactly `"PASS"` or `"FAIL"`.
"""


def high_level_code_generation_prompt(bulleted: str, research_question: str) -> str:
    """Return the prompt that asks for JSON with label, confidence, rationale for one cluster."""
    rq_line = ""
    if research_question:
        rq_line = f"\nResearch Question: {research_question}\nGenerate with the research question in mind.\n"
    return f"""The following open codes belong to one cluster:

{bulleted}
{rq_line}
Output a single JSON object with:
- "label": one short high-level label (2-6 words) for this cluster
- "confidence": integer 1-5 (5 = very coherent, 1 = low coherence)
- "rationale": one sentence explaining why the cluster coheres or doesn't

Output ONLY valid JSON, no other text. Example: {{"label": "Interface usability issues", "confidence": 4, "rationale": "Codes all relate to UI and usability."}}"""


def refine_cluster_assignments_prompt(label: str, bulleted: str, other_str: str) -> str:
    """Return the prompt for identifying codes that belong in another cluster (MOVE or NONE).
    other_str lists up to five alternate cluster labels (most similar to ``label`` by embedding).
    """
    return f"""You are reviewing the codes assigned to a cluster labeled "{label}".

Codes in this cluster:
{bulleted}

The only permitted move targets are these clusters (at most five, chosen as the most similar to "{label}" by embedding — you may not suggest any other label):
{other_str}

A code should be moved ONLY if ALL of the following are true:
1. It shares zero conceptual overlap with "{label}" — not just a weaker fit, but genuinely no overlap.
2. It maps unambiguously to exactly one of the clusters listed above — not a toss-up between two.
3. You would bet confidently on this move; any doubt means leave it.

For each code that meets all three criteria, output exactly:
MOVE: "{{code}}" → "{{target cluster label}}"

If no codes meet all three criteria, output: NONE

Do not move codes that are borderline, tangentially related, or where you are choosing the "least bad" option from the list. When in doubt: NONE."""


def relationship_classification_prompt(
    node_a: str,
    node_b: str,
    research_question: str,
) -> str:
    """Single prompt for hierarchy_construction and cross-cluster linking: classify A/B as equivalent/subsumes/subsumed_by/orthogonal."""
    return f"""Given two codes from a thematic analysis:
A: "{node_a}"
B: "{node_b}"

Research Question: {research_question}

Classify the relationship between A and B as exactly one of:
- "equivalent": A and B mean essentially the same thing and should be merged. They must be near-synonyms, not merely related.
- "subsumes": A is a strict generalization of B — every instance of B is necessarily an instance of A. Sharing a topic or being in the same cluster is NOT sufficient.
- "subsumed_by": B is a strict generalization of A — every instance of A is necessarily an instance of B.
- "orthogonal": A and B are distinct concepts. DEFAULT TO THIS unless the relationship is unambiguous.

Important: two codes that share a topic but describe different aspects, qualities, or directions are orthogonal. When in doubt, choose orthogonal.

Output ONLY valid JSON: {{"relation": "<one of the four>", "confidence": <integer 1-5>, "reason": "<brief explanation>"}}"""


def meta_theme_grouping_prompt(labels_json: str, research_question: str) -> str:
    """Prompt to group cluster labels into a small handful of broad meta-themes (about 3–7 when there are many clusters)."""
    rq_line = ""
    if research_question:
        rq_line = (
            f"\nResearch Question: {research_question}\nGroup with the research question in mind.\n"
        )
    return f"""You are organising the results of a thematic analysis into a high-level hierarchy.

Below is a JSON object mapping cluster IDs to their labels:

{labels_json}
{rq_line}
Your task: group ALL of these clusters into a **small handful** of broad meta-themes.
- If there are **six or more** clusters, use about **3 to 7** meta-themes (not a rigid count — balance and coverage matter).
- If there are **fewer than six** clusters, use **2 to N** themes where N is the number of clusters (merge only when labels are truly redundant).
- With **one** cluster, use a single meta-theme.

Rules:
- Every cluster ID must appear in exactly one group — do not drop any.
- Each meta-theme name should be 2–6 words and broader/more abstract than any individual cluster label.
- Aim for roughly balanced groups (avoid putting most clusters in one group).
- If two cluster labels are near-synonyms, place them in the same group.

Output ONLY valid JSON in this exact format:
{{"meta_themes": [
  {{"name": "Meta-Theme Name", "cluster_ids": ["0", "3", "7"]}},
  {{"name": "Another Meta-Theme", "cluster_ids": ["1", "4"]}},
  ...
]}}"""


def intra_cluster_subtheme_prompt(
    cluster_label: str, codes_list: str, research_question: str
) -> str:
    """Prompt to organise codes within a cluster into 2-5 sub-themes."""
    rq_line = ""
    if research_question:
        rq_line = f"\nResearch Question: {research_question}\n"
    return f"""You are building a thematic hierarchy. The cluster below is labelled:
"{cluster_label}"
{rq_line}
It contains these codes (one per line):
{codes_list}

Your task: organise these codes into 2–5 sub-themes that are more specific than the cluster label.

Rules:
- Every code from the list above MUST appear in exactly one sub-theme OR in "ungrouped_codes".
- Prefer **named sub-themes** for almost every code: keep "ungrouped_codes" **empty or as small as possible** — use it only when a code truly fits no sub-theme.
- Add **enough** sub-themes so no single sub-theme carries an enormous list of codes (aim for roughly **≤ ~50 codes per sub-theme** when the batch is large).
- Sub-theme names should be 2–5 words, more specific than "{cluster_label}".
- Do NOT invent codes that are not in the list above.
- Do NOT rename or alter any code — use the exact strings provided.

Output ONLY valid JSON:
{{"sub_themes": [
  {{"name": "Sub-Theme Name", "codes": ["code a", "code b"]}},
  {{"name": "Another Sub-Theme", "codes": ["code c"]}}
], "ungrouped_codes": ["code d"]}}"""


def hierarchy_refine_bucket_prompt(
    cluster_label: str,
    bucket_label: str,
    codes_bulleted: str,
    research_question: str,
    num_groups: int,
) -> str:
    """Split a large code list into exactly num_groups thematic sub-buckets (flat JSON)."""
    rq_line = f"\nResearch Question: {research_question}\n" if research_question else ""
    return f"""You are refining a thematic hierarchy for readability. Too many codes sit under one parent; split them into smaller groups.

Cluster (context): "{cluster_label}"
Parent bucket to subdivide: "{bucket_label}"
{rq_line}
The following codes must be partitioned into exactly {num_groups} non-empty sub-groups. Each code appears in exactly one group.

Codes (use EXACT strings; do not rename or drop any):
{codes_bulleted}

Rules:
- Output exactly {num_groups} sub-themes in the JSON array.
- Each sub-theme has a short name (2–6 words) and a "codes" list.
- Every code from the list above must appear once across all groups.
- Balance sizes when possible (avoid one giant group and many tiny ones).

Output ONLY valid JSON:
{{"sub_themes": [
  {{"name": "Sub-group name", "codes": ["code a", "code b"]}},
  ...
]}}"""


def research_report_prompt(research_question: str, graph_text: str) -> str:
    """Prompt for final qualitative synthesis from the global thematic graph (markdown output)."""
    return f"""You are a qualitative researcher. Your job is to **answer the research question** below using the thematic graph as evidence—not to describe or explain the graph as your main topic.

Research question:
{research_question}

Below is a **thematic graph** as text: **nodes** (theme labels) and **edges** (parent → child relationships from the analysis pipeline). Treat it as the empirical backbone: only claim what these nodes and edges support. Do not invent themes or relations missing from the graph text.

Thematic graph:
{graph_text}

Write your response in **markdown** with exactly these sections:

## Research question
Restate the research question in one sentence (you may quote it).

## Graph structure
One short line only: how many nodes and how many edges (or counts from the graph header). No thematic interpretation in this section.

## Research answer
**3–6 sentences** that **directly answer the research question** (e.g. how commenters frame severity, hope, denial, responsibility—whatever the question asks). Write about people, framings, and themes in the corpus. You may refer to specific theme names when helpful, but **avoid** opening with “the graph shows/reveals” and **avoid** narrating the graph structure (listing branches, saying “the graph is interconnected”) as a substitute for answering the question. If evidence is weak or ambiguous, note that briefly, then still answer with what the graph supports.

Do not output JSON. Do not add long bullet lists unless essential."""
