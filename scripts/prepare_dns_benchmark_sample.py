#!/usr/bin/env python3
"""Create a small local sample set from the DNS5 dev testset noisy clips."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCH_ROOT = PROJECT_ROOT / "data" / "benchmarks"
DEFAULT_OUTPUT = BENCH_ROOT / "dns5_dev_sample" / "noisy"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a small sample from DNS5 dev testset noisy clips.")
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=16)
    return parser.parse_args()


def infer_input_dir() -> Path:
    candidates = [
        BENCH_ROOT / "dns5_track1_headset",
        BENCH_ROOT / "dns5_dev_testset",
    ]
    for root in candidates:
        if root.exists():
            noisy_dirs = [path for path in root.rglob("*") if path.is_dir() and "noisy" in path.name.lower()]
            if noisy_dirs:
                return sorted(noisy_dirs)[0]
    raise SystemExit("Could not find a staged DNS noisy directory. Run `make dns-dev-testset` first.")


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir or infer_input_dir()
    wavs = sorted(input_dir.rglob("*.wav"))
    if not wavs:
        raise SystemExit(f"No wav files found under {input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    chosen = wavs[: args.limit]
    for src in chosen:
        shutil.copy2(src, args.output_dir / src.name)

    print(f"Copied {len(chosen)} wav files from {input_dir}")
    print(f"Sample output: {args.output_dir}")


if __name__ == "__main__":
    main()
