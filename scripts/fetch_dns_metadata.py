#!/usr/bin/env python3
"""Download lightweight DNS metadata assets into the local data directory."""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "dns_challenge"
BENCH_ROOT = PROJECT_ROOT / "data" / "benchmarks"

METADATA_ASSETS = [
    (
        "https://dnschallengepublic.blob.core.windows.net/dns5archive/filelists_headset.zip",
        RAW_ROOT / "filelists_headset.zip",
        RAW_ROOT / "filelists_headset",
        BENCH_ROOT / "dns5_filelists_headset",
    ),
]


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        print(f"Already downloaded: {destination}")
        return
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)


def extract(archive_path: Path, extract_root: Path) -> None:
    if extract_root.exists() and any(extract_root.iterdir()):
        print(f"Already extracted: {extract_root}")
        return
    extract_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(extract_root)


def reset_link(link_path: Path, target: Path) -> None:
    if link_path.is_symlink() or link_path.exists():
        if link_path.is_dir() and not link_path.is_symlink():
            shutil.rmtree(link_path)
        else:
            link_path.unlink()
    link_path.parent.mkdir(parents=True, exist_ok=True)
    link_path.symlink_to(target.resolve(), target_is_directory=True)


def main() -> None:
    for url, archive_path, extract_root, bench_link in METADATA_ASSETS:
        print(f"Fetching {url}")
        download(url, archive_path)
        extract(archive_path, extract_root)
        reset_link(bench_link, extract_root)
        print(f"Staged: {bench_link} -> {bench_link.resolve()}")


if __name__ == "__main__":
    main()
