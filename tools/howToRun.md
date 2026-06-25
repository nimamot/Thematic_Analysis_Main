What goes in the pipeline repo (once)

Copy only these two things:

- `tools/viewer_launcher.py` (~150 lines, no npm)
- `.viewer-version` → e.g. `0.2.0`

The pipeline exports into `viewer-data/` automatically when `GT_VIEWER_EXPORT=1` (default).
On a fresh clone that folder only contains an empty `manifest.json` and placeholder dirs.

Researchers run:

```bash
python tools/viewer_launcher.py --data-dir ./viewer-data
```

Launcher downloads the release (if needed), syncs data, opens the browser.

## Human-in-the-loop codebook review (local)

In `agents/scripts/pipeline_config.env`:

```bash
export GT_CODEBOOK_REVIEW=1
export GT_CODEBOOK_REVIEW_BACKEND=local   # default without Supabase credentials
export PIPELINE_SLUG=my-study
```

After high-level code generation the pipeline writes review files to
`viewer-data/codebook-reviews/<PIPELINE_SLUG>/` and waits. Researchers:

1. Run `python tools/viewer_launcher.py --data-dir ./viewer-data`
2. Approve in the **Codebook review** tab
3. Save the downloaded `*-codebook_v2.json` as
   `viewer-data/codebook-reviews/<PIPELINE_SLUG>/codebook_v2.json`

The pipeline detects the file and continues.

## Manual export

```bash
python agents/scripts/export_viewer_data.py --mode project --slug my-study
python agents/scripts/export_viewer_data.py --mode codebook-review --slug my-study
```
