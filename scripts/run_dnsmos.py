#!/usr/bin/env python3
"""Wrapper around the DNS-Challenge local DNSMOS scorer."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DNSMOS_DIR = PROJECT_ROOT / "submodules" / "DNS-Challenge" / "DNSMOS"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DNSMOS against a folder of wav files.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--personalized", action="store_true", help="Use personalized MOS scoring.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not DNSMOS_DIR.exists():
        raise SystemExit(f"DNSMOS directory not found: {DNSMOS_DIR}")
    if not args.input_dir.exists():
        raise SystemExit(f"Input directory not found: {args.input_dir}")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "dnsmos_local.py",
        "-t",
        str(args.input_dir.resolve()),
        "-o",
        str(args.output_csv.resolve()),
    ]
    if args.personalized:
        cmd.append("-p")

    subprocess.run(cmd, check=True, cwd=DNSMOS_DIR)
    print(f"DNSMOS results written to {args.output_csv}")


if __name__ == "__main__":
    main()
