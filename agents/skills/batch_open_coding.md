---
name: batch-open-coding
description: Open-code multiple reviews in one call; JSON responses with per-id markdown output.
---

# Batch Open Coding

## Instructions
You receive a JSON array of requests (`id`, `text`, optional `validator_feedback`).

Always:
- Treat each `id` **independently** — never copy codes or evidence across reviews.
- Return **only** valid JSON: `{"responses": [{"id": "...", "output": "..."}, ...]}`.
- Each `output` uses the same markdown rules as single-review open coding (0–3 codes or Applicability NONE with Reason/Evidence).
- When `validator_feedback` is present for an id, incorporate it into that review's `output` only.

Do not include markdown code fences or prose outside the JSON object.
