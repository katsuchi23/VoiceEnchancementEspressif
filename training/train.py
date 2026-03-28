#!/usr/bin/env python3
"""Training entrypoint placeholder for speech enhancement experiments."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a baseline speech enhancement model")
    parser.add_argument("--config", type=Path, default=Path("training/configs/default.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/checkpoints"))
    parser.add_argument("--run-name", type=str, default="debug_nsnet2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("[TODO] Implement model training loop")
    print(f"Config: {args.config}")
    print(f"Output dir: {args.output_dir}")
    print(f"Run name: {args.run_name}")


if __name__ == "__main__":
    main()
