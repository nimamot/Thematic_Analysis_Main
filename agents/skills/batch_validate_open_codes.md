---
name: batch-validate-open-codes
description: Validate batched open-coding outputs; JSON per-id PASS/FAIL verdicts.
---

# Batch Validate Open Codes

## Instructions
You receive a JSON array of requests (`id`, `text`, `generated_codes`).

Always:
- Validate each id **independently** against its review text and coder output.
- Return **only** valid JSON: `{"responses": [{"id": "...", "verdict": "PASS"|"FAIL", "feedback": "..."}, ...]}`.
- Use verdict exactly `PASS` or `FAIL` (uppercase).
- For FAIL, put actionable issues in `feedback` so the coder can revise.

Do not use single-line PASS/FAIL text format. Do not include markdown fences or prose outside JSON.
