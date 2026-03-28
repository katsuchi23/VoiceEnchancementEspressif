#!/usr/bin/env python3
"""Dataset utilities placeholder for noisy-clean pair loading."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


def list_wavs(root: Path) -> Iterable[Path]:
    """Yield wav files recursively from root."""
    return root.rglob("*.wav")
