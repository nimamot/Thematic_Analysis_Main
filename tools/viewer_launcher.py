#!/usr/bin/env python3
"""
Launch the Graph Builder viewer for pipeline researchers — no npm required.

Pipeline repos keep ONLY this file (+ .viewer-version). The UI is downloaded
from GitHub Releases when needed.

Usage:
  python tools/viewer_launcher.py --data-dir ./viewer-data
  python tools/viewer_launcher.py --data-dir ./viewer-data --version 0.2.0

On approve, the launcher saves codebook_v2.json into --data-dir automatically
(pipeline wait step picks it up; no manual drag-and-drop).

Environment (optional):
  VIEWER_REPO          default: nimamot/Agentic_Ta_viz
  VIEWER_VERSION       overrides .viewer-version
  VIEWER_CACHE_DIR     default: ~/.cache/graph-builder-viewer
  VIEWER_PORT          default: 8765
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
import webbrowser
import zipfile
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


DEFAULT_REPO = "nimamot/Agentic_Ta_viz"
DEFAULT_PORT = 8765
SUBMIT_API_PATH = "/api/codebook-review/submit"

# Viewer download helper (minified name Cue in release bundles).
_CUE_DOWNLOAD_SNIPPET = (
    'function Cue(n,e){const t=new Blob([JSON.stringify(e,null,2)],{type:"application/json"}),'
    'i=URL.createObjectURL(t),r=document.createElement("a");r.href=i,r.download=n,r.click(),'
    "URL.revokeObjectURL(i)}"
)
_CUE_PIPELINE_SNIPPET = (
    "function Cue(n,e){"
    f'fetch("{SUBMIT_API_PATH}",{{method:"POST",headers:{{"Content-Type":"application/json"}},'
    "body:JSON.stringify({filename:n,payload:e})}).catch(()=>{});"
    "const t=new Blob([JSON.stringify(e,null,2)],{type:\"application/json\"}),"
    'i=URL.createObjectURL(t),r=document.createElement("a");r.href=i,r.download=n,r.click(),'
    "URL.revokeObjectURL(i)}"
)


def read_pinned_version(pipeline_root: Path) -> str | None:
    candidates = [
        pipeline_root / name
        for name in (".viewer-version", "viewer-version.txt")
    ]
    candidates.append(pipeline_root / "tools" / ".viewer-version")
    for path in candidates:
        if path.is_file():
            v = path.read_text(encoding="utf-8").strip()
            if v:
                return v
    return os.environ.get("VIEWER_VERSION", "").strip() or None


def release_zip_url(repo: str, version: str) -> str:
    tag = version if version.startswith("v") else f"v{version}"
    return f"https://github.com/{repo}/releases/download/{tag}/viewer-dist.zip"


def cache_dir() -> Path:
    base = os.environ.get("VIEWER_CACHE_DIR", "").strip()
    return Path(base).expanduser() if base else Path.home() / ".cache" / "graph-builder-viewer"


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading viewer from {url} …")
    try:
        with urllib.request.urlopen(url, timeout=120) as resp:
            data = resp.read()
    except urllib.error.HTTPError as e:
        raise SystemExit(
            f"Failed to download viewer ({e.code}). "
            f"Check that release {url} exists and viewer-dist.zip is attached."
        ) from e
    dest.write_bytes(data)
    print(f"Saved to {dest}")


def extract_viewer(zip_path: Path, target: Path) -> None:
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(target)
    if not (target / "dist" / "index.html").is_file():
        for child in target.iterdir():
            if (child / "dist" / "index.html").is_file():
                for item in child.iterdir():
                    dest = target / item.name
                    if dest.exists():
                        if dest.is_dir():
                            shutil.rmtree(dest)
                        else:
                            dest.unlink()
                    shutil.move(str(item), str(dest))
                break
    if not (target / "dist" / "index.html").is_file():
        raise SystemExit("Invalid viewer-dist.zip: missing dist/index.html")


LOCAL_VIEWER_BOOTSTRAP = '<script>window.__GRAPH_BUILDER_VIEWER__={dataSource:"local"}</script>'


def patch_index_for_pipeline_mode(dist_dir: Path) -> None:
    """Force local file mode — pipeline zips must not require Supabase env vars."""
    index = dist_dir / "index.html"
    if not index.is_file():
        return
    text = index.read_text(encoding="utf-8")
    if "__GRAPH_BUILDER_VIEWER__" in text:
        return
    if "<head>" in text:
        text = text.replace("<head>", f"<head>\n    {LOCAL_VIEWER_BOOTSTRAP}", 1)
    else:
        text = LOCAL_VIEWER_BOOTSTRAP + "\n" + text
    index.write_text(text, encoding="utf-8")
    print("Patched index.html for local pipeline mode")


_CUE_PIPELINE_SNIPPET_BROKEN = _CUE_PIPELINE_SNIPPET.replace(
    "body:JSON.stringify({filename:n,payload:e})}).catch(()=>{});",
    "body:JSON.stringify({filename:n,payload:e})}}).catch(()=>{});",
)


def patch_viewer_bundle_for_pipeline_submit(dist_dir: Path) -> None:
    """Hook approve export to POST codebook_v2 to the launcher (auto-save for pipeline)."""
    assets = dist_dir / "assets"
    if not assets.is_dir():
        return
    patched = 0
    repaired = 0
    for js_path in assets.glob("index-*.js"):
        text = js_path.read_text(encoding="utf-8")
        if _CUE_PIPELINE_SNIPPET_BROKEN in text:
            text = text.replace(_CUE_PIPELINE_SNIPPET_BROKEN, _CUE_PIPELINE_SNIPPET)
            js_path.write_text(text, encoding="utf-8")
            repaired += 1
            continue
        if SUBMIT_API_PATH in text:
            continue
        if _CUE_DOWNLOAD_SNIPPET not in text:
            print(f"Warning: could not patch {js_path.name} for pipeline auto-save", file=sys.stderr)
            continue
        js_path.write_text(text.replace(_CUE_DOWNLOAD_SNIPPET, _CUE_PIPELINE_SNIPPET), encoding="utf-8")
        patched += 1
    if repaired:
        print(f"Repaired {repaired} viewer bundle(s) (fixed pipeline auto-save syntax)")
    if patched:
        print(f"Patched {patched} viewer bundle(s) for pipeline auto-save ({SUBMIT_API_PATH})")


def slug_from_export_filename(filename: str) -> str:
    name = Path(filename).name
    suffix = "-codebook_v2.json"
    if name.endswith(suffix):
        slug = name[: -len(suffix)].strip()
        return slug or "default"
    if name == "codebook_v2.json":
        return "default"
    return Path(name).stem or "default"


def save_codebook_v2_export(
    *,
    pipeline_data_src: Path,
    dist_data_dir: Path | None,
    filename: str,
    payload: dict,
) -> Path:
    slug = slug_from_export_filename(filename)
    canonical_name = "codebook_v2.json"
    targets = [pipeline_data_src / "codebook-reviews" / slug / canonical_name]
    if dist_data_dir is not None:
        targets.append(dist_data_dir / "codebook-reviews" / slug / canonical_name)

    body = json.dumps(payload, indent=2) + "\n"
    written: Path | None = None
    for path in targets:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
        written = path
    if written is None:
        raise RuntimeError("no write target for codebook_v2 export")
    return written


def sync_data(data_src: Path, dist_dir: Path) -> None:
    """Copy pipeline viewer-data into dist/data/ (what the static server actually serves)."""
    data_dst = dist_dir / "data"
    if data_dst.exists():
        shutil.rmtree(data_dst)
    if not data_src.is_dir():
        raise SystemExit(f"--data-dir not found: {data_src}")
    manifest = data_src / "manifest.json"
    if not manifest.is_file():
        raise SystemExit(
            f"Missing {manifest}. Pipeline must export viewer-data/ with manifest.json — see tools/howToRun.md."
        )
    shutil.copytree(data_src, data_dst)
    synced_manifest = json.loads((data_dst / "manifest.json").read_text(encoding="utf-8"))
    reviews = synced_manifest.get("codebook_reviews", [])
    projects = synced_manifest.get("projects", [])
    print(f"Synced data from {data_src} → {data_dst}")
    print(f"  codebook_reviews: {reviews}")
    print(f"  projects: {projects}")


def ensure_viewer(repo: str, version: str) -> Path:
    ver = version.lstrip("v")
    root = cache_dir() / ver
    if (root / "dist" / "index.html").is_file():
        print(f"Using cached viewer v{ver} at {root}")
        return root

    url = release_zip_url(repo, ver)
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / "viewer-dist.zip"
        download(url, zip_path)
        extract_viewer(zip_path, root)
    print(f"Installed viewer v{ver} to {root}")
    return root


class PipelineViewerHandler(SimpleHTTPRequestHandler):
    pipeline_data_src: Path
    dist_data_dir: Path

    def do_POST(self) -> None:
        if self.path.split("?", 1)[0] != SUBMIT_API_PATH:
            self.send_error(404, "Not found")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            data = json.loads(raw.decode("utf-8"))
            filename = str(data.get("filename") or "codebook_v2.json")
            payload = data.get("payload")
            if not isinstance(payload, dict):
                raise ValueError("payload must be a JSON object")
            dest = save_codebook_v2_export(
                pipeline_data_src=self.pipeline_data_src,
                dist_data_dir=self.dist_data_dir,
                filename=filename,
                payload=payload,
            )
            print(f"Pipeline auto-save: wrote {dest}")
            body = json.dumps({"ok": True, "path": str(dest)}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            msg = json.dumps({"ok": False, "error": str(exc)}).encode("utf-8")
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(msg)))
            self.end_headers()
            self.wfile.write(msg)
        except OSError as exc:
            msg = json.dumps({"ok": False, "error": str(exc)}).encode("utf-8")
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(msg)))
            self.end_headers()
            self.wfile.write(msg)

    def log_message(self, fmt: str, *args) -> None:
        if args and "200" in str(args[1]):
            return
        super().log_message(fmt, *args)


def serve(dist_dir: Path, port: int, *, pipeline_data_src: Path) -> None:
    os.chdir(dist_dir)
    PipelineViewerHandler.pipeline_data_src = pipeline_data_src.resolve()
    PipelineViewerHandler.dist_data_dir = (dist_dir / "data").resolve()

    handler = lambda *args, **kwargs: PipelineViewerHandler(
        *args, directory=str(dist_dir), **kwargs
    )
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    url = f"http://127.0.0.1:{port}/"
    print(f"\nViewer running at {url}")
    print("Approve in Codebook review → saves codebook_v2.json into --data-dir automatically.")
    print("Press Ctrl+C to stop.\n")
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch Graph Builder viewer (downloaded release).")
    parser.add_argument(
        "--data-dir",
        required=True,
        type=Path,
        help="Folder with manifest.json, projects/, codebook-reviews/ (pipeline export)",
    )
    parser.add_argument("--version", help="Viewer version (default: .viewer-version in cwd)")
    parser.add_argument("--repo", default=os.environ.get("VIEWER_REPO", DEFAULT_REPO))
    parser.add_argument("--port", type=int, default=int(os.environ.get("VIEWER_PORT", DEFAULT_PORT)))
    parser.add_argument("--pipeline-root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    version = args.version or read_pinned_version(args.pipeline_root)
    if not version:
        raise SystemExit(
            "No viewer version. Create .viewer-version in the pipeline repo (e.g. 0.2.0) "
            "or pass --version."
        )

    data_src = args.data_dir.resolve()
    viewer_root = ensure_viewer(args.repo, version)
    dist_dir = viewer_root / "dist"
    patch_index_for_pipeline_mode(dist_dir)
    patch_viewer_bundle_for_pipeline_submit(dist_dir)
    sync_data(data_src, dist_dir)
    serve(dist_dir, args.port, pipeline_data_src=data_src)


if __name__ == "__main__":
    main()
