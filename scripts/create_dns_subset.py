#!/usr/bin/env python3
"""Create a small clean/noise subset from DNS assets for fast iteration."""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


SUPPORTED_SUFFIXES = {".wav", ".flac"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DNS_REPO = PROJECT_ROOT / "submodules" / "DNS-Challenge"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a lightweight DNS clean/noise subset.")
    parser.add_argument("--clean-src", type=Path, default=None, help="Source directory for clean speech.")
    parser.add_argument("--noise-src", type=Path, default=None, help="Source directory for noise clips.")
    parser.add_argument("--clean-dst", type=Path, default=PROJECT_ROOT / "data" / "subset" / "clean")
    parser.add_argument("--noise-dst", type=Path, default=PROJECT_ROOT / "data" / "subset" / "noise")
    parser.add_argument("--num-clean", type=int, default=200)
    parser.add_argument("--num-noise", type=int, default=200)
    parser.add_argument("--mode", choices=("random", "head"), default="random")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def guess_clean_root() -> Path | None:
    candidates = [
        DNS_REPO / "datasets_fullband" / "clean_fullband",
        DNS_REPO / "clean_fullband",
    ]
    return next((path for path in candidates if path.exists()), None)


def guess_noise_root() -> Path | None:
    candidates = [
        DNS_REPO / "datasets_fullband" / "noise_fullband",
        DNS_REPO / "noise_fullband",
    ]
    return next((path for path in candidates if path.exists()), None)


def list_audio_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES)


def choose_files(files: list[Path], count: int, mode: str, seed: int) -> list[Path]:
    if count <= 0:
        return []
    if len(files) < count:
        raise ValueError(f"Requested {count} files but only found {len(files)}.")
    if mode == "head":
        return files[:count]
    rng = random.Random(seed)
    return sorted(rng.sample(files, count))


def copy_subset(files: list[Path], src_root: Path, dst_root: Path) -> None:
    dst_root.mkdir(parents=True, exist_ok=True)
    copied = 0
    for src in files:
        rel = src.relative_to(src_root)
        dst = dst_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1
    print(f"Copied {copied} files into {dst_root}")


def main() -> None:
    args = parse_args()
    clean_src = args.clean_src or guess_clean_root()
    noise_src = args.noise_src or guess_noise_root()

    if clean_src is None or not clean_src.exists():
        raise SystemExit("Could not locate DNS clean speech assets. Pass --clean-src explicitly.")
    if noise_src is None or not noise_src.exists():
        raise SystemExit("Could not locate DNS noise assets. Pass --noise-src explicitly.")

    clean_files = list_audio_files(clean_src)
    noise_files = list_audio_files(noise_src)

    if not clean_files:
        raise SystemExit(f"No audio files found under clean source: {clean_src}")
    if not noise_files:
        raise SystemExit(f"No audio files found under noise source: {noise_src}")

    chosen_clean = choose_files(clean_files, args.num_clean, args.mode, args.seed)
    chosen_noise = choose_files(noise_files, args.num_noise, args.mode, args.seed + 1)

    copy_subset(chosen_clean, clean_src, args.clean_dst)
    copy_subset(chosen_noise, noise_src, args.noise_dst)

    print(f"Clean source: {clean_src}")
    print(f"Noise source: {noise_src}")


if __name__ == "__main__":
    main()
