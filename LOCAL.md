# Running locally (no database)

This guide is for researchers who want to explore pipeline outputs on their own machine — **without Supabase**. The Overview research deck is hidden in this mode; you get the **Library** and **Codebook review** tabs with full functionality.

## Quick start

1. **Install Node.js** (v18 or newer) from [nodejs.org](https://nodejs.org/).
2. Open a terminal in this folder and run:

```bash
npm install
npm run dev:local
```

3. Open the URL shown in the terminal (usually `http://localhost:5173`).
4. Click **Load projects** on the Library tab, or open **Codebook review**.

That’s it. The bundled example study under `public/data/` loads automatically.

---

## What you need on disk

All local inputs live under **`public/data/`**. Vite serves this folder at `/data` when the dev server runs.

```
public/data/
├── manifest.json                 ← lists your studies (required)
├── projects/
│   └── <study-slug>/             ← one folder per Library project
│       ├── meta.json             ← id, slug, research question (required)
│       ├── gt_global_graph.json  ← theme tree or edge graph (required)
│       ├── research_report.md    ← narrative report (required)
│       ├── gt_open_codes_all_reviews.md   (optional)
│       └── cooccurrence.json              (optional)
└── codebook-reviews/
    └── <study-slug>/             ← one folder per pending codebook review
        ├── meta.json             ← id, slug, timestamps (required)
        ├── codebook.json         ← initial codebook draft (required)
        ├── gt_clustered_codes.json ← code-to-cluster map (required)
        └── codebook_confidence.json       (optional)
```

### `manifest.json`

Tells the app which folders to load:

```json
{
  "projects": ["my-study", "another-study"],
  "codebook_reviews": ["my-study"]
}
```

Each string is a **folder name** (slug), not a file path.

### Project metadata — `projects/<slug>/meta.json`

| Field | Required | Description |
|-------|----------|-------------|
| `id` | No | Unique id for share links. Defaults to folder name. |
| `slug` | No | Display slug. Defaults to folder name. |
| `research_question` | No | Shown in the project picker. |
| `created_at` | No | ISO timestamp for the metadata chip. |

### Library files — `projects/<slug>/`

These map directly to pipeline output filenames:

| File (preferred name) | Alternatives accepted | Used for |
|-----------------------|----------------------|----------|
| `gt_global_graph.json` | `global_graph.json` | Theme graph (tree or edges) |
| `research_report.md` | `report.md`, `report_markdown.md` | Research report panel |
| `gt_open_codes_all_reviews.md` | `open_codes.md` | Open-code traceability |
| `cooccurrence.json` | — | Co-occurrence network |

**`gt_global_graph.json`** must be either:

- A **theme tree**: `{ "tree": { "name": "...", "type": "root", "children": [...] } }`
- An **edge graph**: `{ "edges": [{ "parent": "...", "child": "..." }], ... }` (same shape as `codebook.json` exports)

**`gt_open_codes_all_reviews.md`** format:

```markdown
## Review 1

- Code: Some code label
- Evidence: "Quoted text from the review."
- Note: Optional reviewer note.
```

### Codebook review — `codebook-reviews/<slug>/`

| File (preferred name) | Alternatives | Used for |
|-----------------------|--------------|----------|
| `codebook.json` | `codebook_v1.json` | Starting draft clusters |
| `gt_clustered_codes.json` | `clustered_codes.json` | Which codes belong to each cluster |
| `codebook_confidence.json` | — | Per-cluster confidence & rationale |

**`meta.json`** fields:

| Field | Required | Description |
|-------|----------|-------------|
| `id` | No | Review id (defaults to folder name) |
| `slug` | No | Display slug |
| `research_question` | No | Queue label |
| `created_at` | No | ISO timestamp |
| `updated_at` | No | ISO timestamp |

---

## Adding your own pipeline outputs

After your Python pipeline finishes, copy its output files into the matching folders:

```bash
# Example: copy pipeline run into the local viewer
STUDY=my-study
mkdir -p public/data/projects/$STUDY
mkdir -p public/data/codebook-reviews/$STUDY

cp outputs/gt_global_graph.json     public/data/projects/$STUDY/
cp outputs/research_report.md       public/data/projects/$STUDY/
cp outputs/gt_open_codes_all_reviews.md public/data/projects/$STUDY/   # if available
cp outputs/cooccurrence.json        public/data/projects/$STUDY/       # if available

cp outputs/codebook.json            public/data/codebook-reviews/$STUDY/
cp outputs/gt_clustered_codes.json  public/data/codebook-reviews/$STUDY/
cp outputs/codebook_confidence.json public/data/codebook-reviews/$STUDY/  # if available
```

Then add `"my-study"` to both arrays in `public/data/manifest.json`, create `meta.json` files (copy from `example-study`), and click **Load projects** / **Refresh**.

---

## Codebook review: approve & export

In local mode there is **no sign-in**. When you click **Approve & export**:

1. The browser downloads `<review-id>-codebook_v2.json`.
2. The review is removed from the pending queue (stored in your browser’s local storage).

Place the downloaded file back into your pipeline’s output directory so the next pipeline stage can pick it up.

To **re-open a submitted review**, clear the browser’s site data for this app, or use a private/incognito window.

---

## Configuration (optional)

Create `.env.local` in the project root:

```env
# Force local file mode (recommended for researchers)
VITE_DATA_SOURCE=local

# Custom data root (default: /data → public/data/)
# VITE_LOCAL_DATA_ROOT=/data
```

| Variable | Values | Default |
|----------|--------|---------|
| `VITE_DATA_SOURCE` | `local`, `supabase` | `local` if Supabase is not configured; otherwise `supabase` |
| `VITE_LOCAL_DATA_ROOT` | URL path | `/data` |

Use `npm run dev:local` to start with local mode guaranteed.

---

## Distribution for pipeline repos (recommended)

You should **not** copy-paste the whole frontend into the pipeline repo. Use one of these patterns so
changes stay in **this repository** only.

### How it works

```
┌─────────────────────────┐         GitHub Release          ┌──────────────────────────┐
│  Graph_builder (here)   │  ──►  viewer-dist.zip (v0.2.0)  │  Pipeline repo           │
│  npm run package:viewer │                                 │  .viewer-version → 0.2.0 │
└─────────────────────────┘                                 │  tools/viewer_launcher.py│
                                                              │  (copy once, ~150 lines) │
                                                              └────────────┬─────────────┘
                                                                           │
                     pipeline run → export viewer-data/                    │
                     python tools/viewer_launcher.py --data-dir viewer-data
                                                                           ▼
                                                              Browser opens — no npm for researchers
```

1. **You** develop UI here, tag `v0.2.0`, CI attaches `viewer-dist.zip` to the release.
2. **Pipeline repo** pins `.viewer-version` and keeps one launcher script (`tools/viewer_launcher.py`).
3. **Researchers** run the pipeline, then one command — launcher downloads the zip (cached), syncs data, opens the app.

### Frontend maintainer (this repo)

```bash
# After changes, bump version in package.json, then:
git tag v0.2.0 && git push origin v0.2.0
# GitHub Action builds viewer-dist.zip and attaches it to the release.

# Or manually:
npm run package:viewer
# → dist-viewer/viewer-dist.zip
```

### Pipeline maintainer (your backend repo)

Copy **once**:

- `tools/viewer_launcher.py` (from this repo)
- `.viewer-version` with e.g. `0.2.0`

Add a pipeline step that exports outputs into the viewer folder layout (see below), then:

```bash
python tools/viewer_launcher.py --data-dir ./viewer-data
```

Bump `.viewer-version` when you want researchers to get a new UI — no need to merge frontend code.

### Alternative: git submodule (for devs, not researchers)

```bash
# In pipeline repo
git submodule add https://github.com/nimamot/Agentic_Ta_viz.git viewer
cd viewer && npm install && npm run dev:local
```

Good for pipeline developers; researchers should still use `viewer_launcher.py` + releases.

### Alternative: npm git dependency

In pipeline `package.json` (optional, still needs Node):

```json
"dependencies": {
  "thematic-viewer": "github:nimamot/Agentic_Ta_viz#v0.2.0"
}
```

Releases + launcher is simpler for non-frontend researchers.

---

## Copying the module to your backend repo

The **file-based data layer** (for custom integrations) lives in:

```
src/lib/local/          ← loaders, submit, file conventions
src/lib/dataSource.ts   ← switches between local and Supabase
```

That is only needed if you embed the loaders into a **different** React app. For the standard viewer,
use **releases + viewer_launcher.py** instead of copying `src/`.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Empty project list | Check `manifest.json` lists your folder slug; click **Load projects**. |
| “Missing file” error | Compare filenames to the table above; pipeline names like `gt_global_graph.json` are preferred. |
| Graph shows error | Ensure `gt_global_graph.json` has either a `tree` root or `edges` array. |
| Review not in queue | Add slug to `codebook_reviews` in `manifest.json`; ensure `codebook.json` and `gt_clustered_codes.json` exist. |
| Review disappeared after approve | Expected — export is saved; queue state is in browser local storage. |

---

## Production build (optional)

```bash
npm run build:local
npm run preview
```

Serve the `dist/` folder on any static host. Put your `data/` folder next to `index.html` (same layout as `public/data/`).
