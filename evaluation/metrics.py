#!/usr/bin/env python3
"""Objective metrics for paired speech enhancement evaluation."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from pesq import pesq
from pystoi import stoi


SUPPORTED_SUFFIXES = {".wav", ".flac"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate paired clean/enhanced folders.")
    parser.add_argument("--enhanced-dir", type=Path, required=True)
    parser.add_argument("--clean-dir", type=Path, required=True)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--output", type=Path, default=None, help="Optional CSV output path.")
    return parser.parse_args()


def list_audio(root: Path) -> dict[str, Path]:
    return {
        path.name: path
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    }


def load_mono(path: Path, sample_rate: int) -> np.ndarray:
    audio, _ = librosa.load(path, sr=sample_rate, mono=True)
    return audio.astype(np.float32)


def align_length(clean: np.ndarray, enhanced: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    length = min(len(clean), len(enhanced))
    return clean[:length], enhanced[:length]


def si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    ref_energy = float(np.dot(reference, reference) + 1e-12)
    projection = (float(np.dot(estimate, reference)) / ref_energy) * reference
    noise = estimate - projection
    ratio = (np.sum(projection ** 2) + 1e-12) / (np.sum(noise ** 2) + 1e-12)
    return float(10.0 * np.log10(ratio))


def evaluate_pair(clean_path: Path, enhanced_path: Path, sample_rate: int) -> dict[str, float | str]:
    clean = load_mono(clean_path, sample_rate)
    enhanced = load_mono(enhanced_path, sample_rate)
    clean, enhanced = align_length(clean, enhanced)

    if len(clean) < sample_rate // 4:
        raise ValueError("Audio clip is too short for stable metric computation.")

    result: dict[str, float | str] = {
        "file": clean_path.name,
        "clean_path": str(clean_path),
        "enhanced_path": str(enhanced_path),
        "duration_sec": round(len(clean) / sample_rate, 3),
        "stoi": float(stoi(clean, enhanced, sample_rate, extended=False)),
        "si_sdr": si_sdr(clean, enhanced),
    }

    try:
        result["pesq_wb"] = float(pesq(sample_rate, clean, enhanced, "wb"))
    except Exception as exc:  # pragma: no cover - PESQ can fail on malformed clips
        result["pesq_wb"] = float("nan")
        result["pesq_error"] = str(exc)

    return result


def mean_metric(rows: list[dict[str, float | str]], key: str) -> float:
    values = [float(row[key]) for row in rows if key in row and not np.isnan(float(row[key]))]
    return float(np.mean(values)) if values else float("nan")


def write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if not args.enhanced_dir.exists():
        raise SystemExit(f"Enhanced dir not found: {args.enhanced_dir}")
    if not args.clean_dir.exists():
        raise SystemExit(f"Clean dir not found: {args.clean_dir}")

    clean_map = list_audio(args.clean_dir)
    enhanced_map = list_audio(args.enhanced_dir)
    shared_names = sorted(set(clean_map) & set(enhanced_map))

    if not shared_names:
        raise SystemExit("No matching filenames found between clean and enhanced directories.")

    rows = [evaluate_pair(clean_map[name], enhanced_map[name], args.sample_rate) for name in shared_names]

    print(f"Matched files: {len(rows)}")
    print(f"Mean PESQ-WB: {mean_metric(rows, 'pesq_wb'):.4f}")
    print(f"Mean STOI: {mean_metric(rows, 'stoi'):.4f}")
    print(f"Mean SI-SDR: {mean_metric(rows, 'si_sdr'):.4f}")

    if args.output is not None:
        write_csv(args.output, rows)
        print(f"Detailed report: {args.output}")


if __name__ == "__main__":
    main()
