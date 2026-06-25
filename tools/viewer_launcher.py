#!/usr/bin/env python3
"""
Launch the Graph Builder viewer for pipeline researchers — no npm required.

Pipeline repos keep ONLY this file (+ .viewer-version). The UI is downloaded
from GitHub Releases when needed.

Usage:
  python tools/viewer_launcher.py --data-dir ./viewer-data
  python tools/viewer_launcher.py --data-dir ./viewer-data --version 0.2.0

Environment (optional):
  VIEWER_REPO          default: nimamot/Agentic_Ta_viz
  VIEWER_VERSION       overrides .viewer-version
  VIEWER_CACHE_DIR     default: ~/.cache/graph-builder-viewer
  VIEWER_PORT          default: 8765
"""

from __future__ import annotations

import argparse
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
    # package-viewer.sh zips contents of viewer/ — dist/ should be at target/dist/
    if not (target / "dist" / "index.html").is_file():
        # Sometimes zip has an extra top-level folder
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


def sync_data(data_src: Path, serve_root: Path) -> None:
    data_dst = serve_root / "data"
    if data_dst.exists():
        shutil.rmtree(data_dst)
    if not data_src.is_dir():
        raise SystemExit(f"--data-dir not found: {data_src}")
    manifest = data_src / "manifest.json"
    if not manifest.is_file():
        raise SystemExit(
            f"Missing {manifest}. Pipeline must export viewer-data/ with manifest.json — see LOCAL.md."
        )
    shutil.copytree(data_src, data_dst)
    print(f"Synced data from {data_src} → {data_dst}")


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


class ViewerHandler(SimpleHTTPRequestHandler):
  def __init__(self, *args, directory: str | None = None, **kwargs):
    super().__init__(*args, directory=directory, **kwargs)

  def log_message(self, fmt: str, *args) -> None:
    if args and "200" in str(args[1]):
      return
    super().log_message(fmt, *args)


def serve(dist_dir: Path, port: int) -> None:
    os.chdir(dist_dir)
    handler = lambda *args, **kwargs: ViewerHandler(*args, directory=str(dist_dir), **kwargs)
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    url = f"http://127.0.0.1:{port}/"
    print(f"\nViewer running at {url}")
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

    viewer_root = ensure_viewer(args.repo, version)
    dist_dir = viewer_root / "dist"
    serve_root = dist_dir.parent
    sync_data(args.data_dir.resolve(), serve_root)
    serve(dist_dir, args.port)


if __name__ == "__main__":
    main()
