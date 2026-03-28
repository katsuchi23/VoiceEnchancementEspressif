#!/usr/bin/env python3
"""Serve a lightweight local UI for manual audio triplet review."""

from __future__ import annotations

import argparse
import json
import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote


SUPPORTED_SUFFIXES = {".wav"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a local UI for clean/noisy/result audio review.")
    parser.add_argument("--clean-dir", type=Path, required=True)
    parser.add_argument("--noisy-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8008)
    parser.add_argument("--title", default=None)
    return parser.parse_args()


def list_audio_files(root: Path) -> dict[str, Path]:
    if not root.is_dir():
        raise SystemExit(f"Directory does not exist: {root}")
    return {
        path.name: path
        for path in sorted(root.iterdir())
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    }


def build_manifest(clean_dir: Path, noisy_dir: Path, result_dir: Path) -> dict[str, object]:
    clean_files = list_audio_files(clean_dir)
    noisy_files = list_audio_files(noisy_dir)
    result_files = list_audio_files(result_dir)

    filenames = sorted(set(clean_files) & set(noisy_files) & set(result_files))
    if not filenames:
        raise SystemExit(
            "No shared wav filenames found across clean/noisy/result directories. "
            "Use matching filenames such as 01.wav, 02.wav, ..."
        )

    items = [
        {
            "id": filename.rsplit(".", 1)[0],
            "filename": filename,
            "clean_url": f"/audio/clean/{filename}",
            "noisy_url": f"/audio/noisy/{filename}",
            "result_url": f"/audio/result/{filename}",
        }
        for filename in filenames
    ]

    return {
        "count": len(items),
        "items": items,
        "clean_dir": str(clean_dir.resolve()),
        "noisy_dir": str(noisy_dir.resolve()),
        "result_dir": str(result_dir.resolve()),
    }


HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    :root {{
      --bg: #f4efe6;
      --panel: rgba(255, 251, 245, 0.92);
      --ink: #13212c;
      --muted: #566575;
      --line: rgba(19, 33, 44, 0.12);
      --accent: #0f7a6c;
      --accent-2: #cb5d39;
      --shadow: 0 18px 40px rgba(19, 33, 44, 0.12);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(15, 122, 108, 0.18), transparent 30%),
        radial-gradient(circle at top right, rgba(203, 93, 57, 0.16), transparent 22%),
        linear-gradient(180deg, #fbf8f2 0%, var(--bg) 100%);
      font-family: "IBM Plex Sans", "Avenir Next", "Segoe UI", sans-serif;
    }}
    .shell {{
      width: min(1200px, calc(100vw - 32px));
      margin: 24px auto 48px;
    }}
    .hero {{
      background: linear-gradient(135deg, rgba(255,255,255,0.85), rgba(255,247,238,0.92));
      border: 1px solid rgba(19, 33, 44, 0.08);
      border-radius: 28px;
      box-shadow: var(--shadow);
      padding: 28px;
      position: sticky;
      top: 12px;
      backdrop-filter: blur(12px);
      z-index: 10;
    }}
    .eyebrow {{
      margin: 0 0 6px;
      color: var(--accent);
      font-family: "IBM Plex Mono", "SFMono-Regular", monospace;
      font-size: 12px;
      letter-spacing: 0.14em;
      text-transform: uppercase;
    }}
    h1 {{
      margin: 0;
      font-family: "Space Grotesk", "Avenir Next", sans-serif;
      font-size: clamp(28px, 5vw, 46px);
      line-height: 0.95;
    }}
    .sub {{
      margin: 10px 0 0;
      color: var(--muted);
      max-width: 880px;
      line-height: 1.5;
    }}
    .controls {{
      display: grid;
      grid-template-columns: 1.4fr 0.8fr auto auto auto;
      gap: 12px;
      margin-top: 22px;
      align-items: center;
    }}
    .field, button, select {{
      min-height: 46px;
      border-radius: 14px;
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.82);
      color: var(--ink);
      font: inherit;
    }}
    .field {{
      padding: 0 14px;
    }}
    button, select {{
      padding: 0 14px;
      cursor: pointer;
    }}
    button.primary {{
      background: var(--accent);
      color: white;
      border-color: transparent;
    }}
    .summary {{
      display: flex;
      gap: 16px;
      flex-wrap: wrap;
      margin-top: 16px;
      color: var(--muted);
      font-size: 14px;
    }}
    .pill {{
      padding: 7px 10px;
      border-radius: 999px;
      background: rgba(15, 122, 108, 0.08);
      border: 1px solid rgba(15, 122, 108, 0.14);
    }}
    .grid {{
      margin-top: 24px;
      display: grid;
      gap: 18px;
    }}
    .card {{
      border-radius: 24px;
      background: var(--panel);
      border: 1px solid rgba(19, 33, 44, 0.08);
      box-shadow: var(--shadow);
      overflow: hidden;
    }}
    .card-top {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      padding: 18px 20px 14px;
      border-bottom: 1px solid rgba(19, 33, 44, 0.08);
      align-items: center;
    }}
    .file-id {{
      font-family: "IBM Plex Mono", "SFMono-Regular", monospace;
      font-size: 15px;
      letter-spacing: 0.06em;
      text-transform: uppercase;
    }}
    .review-state {{
      font-size: 13px;
      color: var(--muted);
    }}
    .lanes {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 0;
    }}
    .lane {{
      padding: 18px 20px 20px;
      border-right: 1px solid rgba(19, 33, 44, 0.08);
    }}
    .lane:last-child {{
      border-right: 0;
    }}
    .lane h2 {{
      margin: 0 0 8px;
      font-family: "Space Grotesk", "Avenir Next", sans-serif;
      font-size: 16px;
    }}
    .lane p {{
      margin: 0 0 12px;
      color: var(--muted);
      font-size: 13px;
    }}
    audio {{
      width: 100%;
    }}
    .review-box {{
      display: grid;
      gap: 10px;
      padding: 16px 20px 20px;
      border-top: 1px solid rgba(19, 33, 44, 0.08);
      background: linear-gradient(180deg, rgba(244,239,230,0.24), rgba(255,255,255,0.5));
    }}
    .review-grid {{
      display: grid;
      grid-template-columns: 180px 1fr;
      gap: 12px;
    }}
    textarea {{
      width: 100%;
      min-height: 82px;
      padding: 12px 14px;
      resize: vertical;
      border-radius: 16px;
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.84);
      color: var(--ink);
      font: inherit;
    }}
    .empty {{
      padding: 40px 24px;
      text-align: center;
      color: var(--muted);
    }}
    @media (max-width: 900px) {{
      .controls {{
        grid-template-columns: 1fr;
      }}
      .lanes {{
        grid-template-columns: 1fr;
      }}
      .lane {{
        border-right: 0;
        border-bottom: 1px solid rgba(19, 33, 44, 0.08);
      }}
      .lane:last-child {{
        border-bottom: 0;
      }}
      .review-grid {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <p class="eyebrow">Manual Review</p>
      <h1>{title}</h1>
      <p class="sub">
        Compare aligned <strong>clean</strong>, <strong>noisy</strong>, and <strong>result</strong> clips by shared filename.
        Ratings and notes stay in your browser and can be exported as JSON.
      </p>
      <div class="controls">
        <input id="search" class="field" type="search" placeholder="Filter by filename, like 03 or 10.wav">
        <select id="status-filter">
          <option value="all">Show all</option>
          <option value="unreviewed">Only unreviewed</option>
          <option value="reviewed">Only reviewed</option>
        </select>
        <button id="prev-btn">Previous</button>
        <button id="next-btn">Next</button>
        <button id="export-btn" class="primary">Export Notes</button>
      </div>
      <div class="summary">
        <span class="pill" id="count-pill">0 files</span>
        <span class="pill" id="dirs-pill">Loading...</span>
      </div>
    </section>
    <main id="grid" class="grid"></main>
  </div>
  <script>
    const STORAGE_KEY = "audio-review::{storage_key}";
    const grid = document.getElementById("grid");
    const searchInput = document.getElementById("search");
    const filterSelect = document.getElementById("status-filter");
    const exportBtn = document.getElementById("export-btn");
    const countPill = document.getElementById("count-pill");
    const dirsPill = document.getElementById("dirs-pill");
    const prevBtn = document.getElementById("prev-btn");
    const nextBtn = document.getElementById("next-btn");

    let items = [];
    let visibleItems = [];
    let currentIndex = 0;
    let reviewState = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{{}}");

    function reviewed(entry) {{
      const state = reviewState[entry.filename];
      return !!(state && ((state.rating && state.rating !== "unrated") || (state.notes && state.notes.trim())));
    }}

    function saveState(filename, patch) {{
      reviewState[filename] = Object.assign({{}}, reviewState[filename] || {{}}, patch);
      localStorage.setItem(STORAGE_KEY, JSON.stringify(reviewState));
      render();
    }}

    function applyFilters() {{
      const query = searchInput.value.trim().toLowerCase();
      const status = filterSelect.value;
      visibleItems = items.filter((entry) => {{
        const matchesQuery = !query || entry.filename.toLowerCase().includes(query);
        const isReviewed = reviewed(entry);
        const matchesStatus =
          status === "all" ||
          (status === "reviewed" && isReviewed) ||
          (status === "unreviewed" && !isReviewed);
        return matchesQuery && matchesStatus;
      }});
      if (currentIndex >= visibleItems.length) currentIndex = Math.max(visibleItems.length - 1, 0);
    }}

    function card(entry) {{
      const state = reviewState[entry.filename] || {{}};
      const statusText = reviewed(entry) ? "reviewed" : "unreviewed";
      const wrap = document.createElement("article");
      wrap.className = "card";
      wrap.innerHTML = `
        <div class="card-top">
          <div>
            <div class="file-id">${{entry.filename}}</div>
          </div>
          <div class="review-state">${{statusText}}</div>
        </div>
        <div class="lanes">
          <section class="lane">
            <h2>Clean</h2>
            <p>Reference speech</p>
            <audio controls preload="none" src="${{entry.clean_url}}"></audio>
          </section>
          <section class="lane">
            <h2>Noisy</h2>
            <p>Input mixture</p>
            <audio controls preload="none" src="${{entry.noisy_url}}"></audio>
          </section>
          <section class="lane">
            <h2>Result</h2>
            <p>Enhanced output</p>
            <audio controls preload="none" src="${{entry.result_url}}"></audio>
          </section>
        </div>
        <div class="review-box">
          <div class="review-grid">
            <select data-role="rating">
              <option value="unrated">Unrated</option>
              <option value="much_worse">Much worse</option>
              <option value="worse">Worse</option>
              <option value="same">Same</option>
              <option value="better">Better</option>
              <option value="much_better">Much better</option>
            </select>
            <textarea data-role="notes" placeholder="Notes: speech distortion, residual hiss, pumping, strong suppression, etc."></textarea>
          </div>
        </div>
      `;
      const rating = wrap.querySelector('[data-role="rating"]');
      const notes = wrap.querySelector('[data-role="notes"]');
      rating.value = state.rating || "unrated";
      notes.value = state.notes || "";
      rating.addEventListener("change", () => saveState(entry.filename, {{ rating: rating.value }}));
      notes.addEventListener("change", () => saveState(entry.filename, {{ notes: notes.value }}));
      return wrap;
    }}

    function render() {{
      applyFilters();
      grid.innerHTML = "";
      countPill.textContent = `${{visibleItems.length}} / ${{items.length}} files`;
      const reviewedCount = items.filter(reviewed).length;
      dirsPill.textContent = `${{reviewedCount}} reviewed`;
      if (!visibleItems.length) {{
        grid.innerHTML = '<div class="empty">No files match the current filter.</div>';
        return;
      }}
      visibleItems.forEach((entry) => grid.appendChild(card(entry)));
      const active = grid.children[currentIndex];
      if (active) active.scrollIntoView({{ behavior: "smooth", block: "start" }});
    }}

    function move(delta) {{
      if (!visibleItems.length) return;
      currentIndex = Math.min(Math.max(currentIndex + delta, 0), visibleItems.length - 1);
      const active = grid.children[currentIndex];
      if (active) active.scrollIntoView({{ behavior: "smooth", block: "start" }});
    }}

    searchInput.addEventListener("input", render);
    filterSelect.addEventListener("change", render);
    prevBtn.addEventListener("click", () => move(-1));
    nextBtn.addEventListener("click", () => move(1));
    exportBtn.addEventListener("click", () => {{
      const payload = {{
        title: document.title,
        exported_at: new Date().toISOString(),
        reviews: reviewState
      }};
      const blob = new Blob([JSON.stringify(payload, null, 2)], {{ type: "application/json" }});
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = "audio_review_notes.json";
      link.click();
      URL.revokeObjectURL(url);
    }});
    window.addEventListener("keydown", (event) => {{
      if (event.target.tagName === "TEXTAREA" || event.target.tagName === "INPUT" || event.target.tagName === "SELECT") return;
      if (event.key === "j") move(1);
      if (event.key === "k") move(-1);
    }});

    fetch("/manifest.json")
      .then((response) => response.json())
      .then((manifest) => {{
        items = manifest.items;
        dirsPill.textContent = `${{manifest.result_dir}}`;
        render();
      }})
      .catch((error) => {{
        grid.innerHTML = `<div class="empty">Failed to load manifest: ${{error}}</div>`;
      }});
  </script>
</body>
</html>
"""


def make_handler(manifest: dict[str, object], title: str, clean_dir: Path, noisy_dir: Path, result_dir: Path):
    audio_roots = {
        "clean": clean_dir.resolve(),
        "noisy": noisy_dir.resolve(),
        "result": result_dir.resolve(),
    }

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            path = unquote(self.path)
            if path == "/" or path == "/index.html":
                storage_key = f"{clean_dir.resolve()}|{noisy_dir.resolve()}|{result_dir.resolve()}"
                html = HTML.format(title=title, storage_key=storage_key).encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(html)))
                self.end_headers()
                self.wfile.write(html)
                return

            if path == "/manifest.json":
                payload = json.dumps(manifest).encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return

            parts = path.strip("/").split("/", 2)
            if len(parts) == 3 and parts[0] == "audio" and parts[1] in audio_roots:
                candidate = (audio_roots[parts[1]] / parts[2]).resolve()
                if candidate.parent != audio_roots[parts[1]] or not candidate.exists():
                    self.send_error(HTTPStatus.NOT_FOUND, "File not found")
                    return
                content_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
                data = candidate.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return

            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

        def log_message(self, fmt: str, *args) -> None:
            return

    return Handler


def main() -> None:
    args = parse_args()
    clean_dir = args.clean_dir.resolve()
    noisy_dir = args.noisy_dir.resolve()
    result_dir = args.result_dir.resolve()
    title = args.title or f"Audio Review: {result_dir.name}"

    manifest = build_manifest(clean_dir, noisy_dir, result_dir)
    handler = make_handler(manifest, title, clean_dir, noisy_dir, result_dir)
    server = ThreadingHTTPServer((args.host, args.port), handler)

    print(f"Serving audio review UI on http://{args.host}:{args.port}")
    print(f"Clean:  {clean_dir}")
    print(f"Noisy:  {noisy_dir}")
    print(f"Result: {result_dir}")
    print(f"Matched files: {manifest['count']}")
    server.serve_forever()


if __name__ == "__main__":
    main()
