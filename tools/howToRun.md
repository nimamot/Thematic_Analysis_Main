# Local viewer (no Supabase, no npm)

The pipeline exports into `viewer-data/` when `GT_VIEWER_EXPORT=1` (default). Researchers open the **Graph Builder** UI with one command:

```bash
python tools/viewer_launcher.py --data-dir ./viewer-data
```

The launcher downloads the pinned viewer release (cached under `~/.cache/graph-builder-viewer/`), copies `viewer-data/` into the served `dist/data/`, and opens the browser. **Restart the launcher** after the pipeline re-exports so the browser gets fresh files.

Pin the viewer version in `tools/.viewer-version` (e.g. `0.2.4`).

---

## Human-in-the-loop codebook review

In `agents/scripts/pipeline_config.env`:

```bash
export GT_CODEBOOK_REVIEW=1
export GT_CODEBOOK_REVIEW_BACKEND=local   # default without Supabase credentials
export PIPELINE_SLUG=my-study
```

After high-level labels the pipeline writes review files to `viewer-data/codebook-reviews/<PIPELINE_SLUG>/` and waits.

1. Run `python tools/viewer_launcher.py --data-dir ./viewer-data` (on HPC: second terminal on the login node while the job waits).
2. Open **Codebook review**, inspect clusters, edit labels, click **Approve & export**.
3. The launcher auto-saves `codebook_v2.json` into `viewer-data/codebook-reviews/<slug>/`; the pipeline detects it and continues.

When the run finishes, **Library** shows the theme graph and research report for that slug.

---

## `viewer-data/` layout

```
viewer-data/
├── manifest.json                 ← lists study slugs (required)
├── projects/
│   └── <slug>/                   ← finished run (Library tab)
│       ├── meta.json
│       ├── gt_global_graph.json
│       ├── research_report.md
│       ├── gt_open_codes_all_reviews.md   (optional)
│       └── cooccurrence.json              (optional)
└── codebook-reviews/
    └── <slug>/                   ← pending review (Codebook review tab)
        ├── meta.json
        ├── codebook.json
        ├── gt_clustered_codes.json
        ├── code_evidence.json             (optional, hover quotes)
        └── codebook_confidence.json       (optional)
```

### `manifest.json`

```json
{
  "projects": ["my-study"],
  "codebook_reviews": ["my-study"]
}
```

Each entry is a **folder name** (slug), not a file path.

### File names

| Purpose | Preferred name | Alternatives |
|---------|----------------|--------------|
| Theme graph | `gt_global_graph.json` | `global_graph.json` |
| Research report | `research_report.md` | `report.md`, `report_markdown.md` |
| Open codes | `gt_open_codes_all_reviews.md` | `open_codes.md` |
| Codebook draft | `codebook.json` | `codebook_v1.json` |
| Cluster map | `gt_clustered_codes.json` | `clustered_codes.json` |

`gt_global_graph.json` must have either a `tree` root or an `edges` array.

### `meta.json` (projects and reviews)

| Field | Description |
|-------|-------------|
| `id`, `slug` | Default to folder name |
| `research_question` | Shown in the project picker |
| `created_at`, `updated_at` | ISO timestamps for the UI chip |

---

## Manual export

```bash
python agents/scripts/export_viewer_data.py --mode project --slug my-study
python agents/scripts/export_viewer_data.py --mode codebook-review --slug my-study
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Stale report or graph in the browser | Restart `viewer_launcher.py` after re-export (data is synced only at launch). |
| Empty project list | Check `manifest.json` lists your slug; click **Load projects**. |
| “Missing file” error | Compare filenames to the table above. |
| Graph shows error | Ensure `gt_global_graph.json` has a `tree` or `edges` field. |
| Review not in queue | Add slug to `codebook_reviews` in `manifest.json`; ensure `codebook.json` and `gt_clustered_codes.json` exist. |
| Review disappeared after approve | Expected — export is saved; queue state is in browser local storage. |
