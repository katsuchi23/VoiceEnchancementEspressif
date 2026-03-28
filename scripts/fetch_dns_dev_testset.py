#!/usr/bin/env python3
"""Download and stage the DNS5 dev testset into the local data directory."""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "dns_challenge"
BENCH_ROOT = PROJECT_ROOT / "data" / "benchmarks"
ARCHIVE_PATH = RAW_ROOT / "V5_dev_testset.zip"
EXTRACT_ROOT = RAW_ROOT / "V5_dev_testset"
URL = "https://dnschallengepublic.blob.core.windows.net/dns5archive/V5_dev_testset.zip"


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        print(f"Archive already exists: {destination}")
        print("If the file is partial, the downloader will resume from the current size.")
    if destination.exists() and destination.stat().st_size > 0 and is_complete_zip(destination):
        print("Archive appears complete. Skipping download.")
        return

    existing_size = destination.stat().st_size if destination.exists() else 0
    headers = {}
    mode = "wb"
    if existing_size > 0:
        headers["Range"] = f"bytes={existing_size}-"
        mode = "ab"
        print(f"Resuming download from byte offset {existing_size}")

    print(f"Downloading {url}")
    with requests.get(url, stream=True, timeout=60, headers=headers) as response:
        response.raise_for_status()
        if response.status_code == 200 and existing_size > 0:
            # Server ignored Range. Restart cleanly.
            existing_size = 0
            mode = "wb"
            print("Server did not honor resume request. Restarting download from zero.")
        total = int(response.headers.get("content-length", 0))
        if response.status_code == 206:
            total += existing_size
        downloaded = existing_size
        with destination.open(mode) as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                handle.write(chunk)
                downloaded += len(chunk)
                if total:
                    percent = downloaded * 100.0 / total
                    print(f"\rDownloaded {downloaded / 1e9:.2f} GB / {total / 1e9:.2f} GB ({percent:.1f}%)", end="")
    print()


def is_complete_zip(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path) as archive:
            return archive.testzip() is None
    except zipfile.BadZipFile:
        return False


def extract_archive(archive_path: Path, extract_root: Path) -> None:
    if extract_root.exists() and any(extract_root.iterdir()):
        print(f"Extracted dataset already exists: {extract_root}")
        return

    extract_root.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {archive_path} to {extract_root}")
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(extract_root)


def find_track_root(extract_root: Path, track_name: str) -> Path | None:
    candidates = [path for path in extract_root.rglob(track_name) if path.is_dir()]
    return sorted(candidates)[0] if candidates else None


def reset_link(link_path: Path, target: Path) -> None:
    if link_path.is_symlink() or link_path.exists():
        if link_path.is_dir() and not link_path.is_symlink():
            shutil.rmtree(link_path)
        else:
            link_path.unlink()
    link_path.parent.mkdir(parents=True, exist_ok=True)
    link_path.symlink_to(target.resolve(), target_is_directory=True)


def stage_links(extract_root: Path) -> None:
    BENCH_ROOT.mkdir(parents=True, exist_ok=True)
    reset_link(BENCH_ROOT / "dns5_dev_testset", extract_root)

    for track_name, alias in [
        ("Track1_Headset", "dns5_track1_headset"),
        ("Track2_Speakerphone", "dns5_track2_speakerphone"),
    ]:
        track_root = find_track_root(extract_root, track_name)
        if track_root is not None:
            reset_link(BENCH_ROOT / alias, track_root)


def main() -> None:
    download_file(URL, ARCHIVE_PATH)
    extract_archive(ARCHIVE_PATH, EXTRACT_ROOT)
    stage_links(EXTRACT_ROOT)

    print("Staged benchmark paths:")
    for path in sorted(BENCH_ROOT.glob("dns5*")):
        print(f"- {path} -> {path.resolve()}")


if __name__ == "__main__":
    main()
