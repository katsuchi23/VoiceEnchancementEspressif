#!/usr/bin/env python3
"""Generate paired noisy-clean debug data without relying on DNS metadata CSVs."""

from __future__ import annotations

import argparse
import csv
import math
import random
import shutil
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


SUPPORTED_SUFFIXES = {".wav", ".flac"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a small paired debug dataset for model bring-up.")
    parser.add_argument("--clean-root", type=Path, default=PROJECT_ROOT / "data" / "samples" / "dns_preview" / "clean")
    parser.add_argument("--noise-root", type=Path, default=PROJECT_ROOT / "data" / "samples" / "dns_preview" / "noise")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "data" / "processed" / "debug")
    parser.add_argument("--num-examples", type=int, default=128)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--duration-sec", type=float, default=2.0)
    parser.add_argument("--snr-min", type=float, default=-5.0)
    parser.add_argument("--snr-max", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def list_audio_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES)


def load_mono(path: Path, sample_rate: int) -> np.ndarray:
    audio, _ = librosa.load(path, sr=sample_rate, mono=True)
    return audio.astype(np.float32)


def fit_length(audio: np.ndarray, target_len: int, rng: random.Random) -> np.ndarray:
    if len(audio) == target_len:
        return audio
    if len(audio) > target_len:
        start = rng.randint(0, len(audio) - target_len)
        return audio[start : start + target_len]
    repeats = math.ceil(target_len / max(len(audio), 1))
    tiled = np.tile(audio, repeats)
    return tiled[:target_len]


def rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio)) + 1e-12))


def mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> tuple[np.ndarray, np.ndarray]:
    clean_rms = rms(clean)
    noise_rms = rms(noise)
    if noise_rms == 0.0:
        return clean.copy(), noise.copy()
    target_noise_rms = clean_rms / (10 ** (snr_db / 20.0) + 1e-12)
    scaled_noise = noise * (target_noise_rms / noise_rms)
    noisy = clean + scaled_noise
    peak = np.max(np.abs(noisy))
    if peak > 0.99:
        scale = 0.99 / peak
        clean = clean * scale
        scaled_noise = scaled_noise * scale
        noisy = noisy * scale
    return noisy.astype(np.float32), scaled_noise.astype(np.float32)


def numeric_id(index: int, total: int) -> str:
    width = max(2, len(str(total)))
    return f"{index + 1:0{width}d}"


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    target_len = int(args.duration_sec * args.sample_rate)

    clean_files = list_audio_files(args.clean_root)
    noise_files = list_audio_files(args.noise_root)

    if not clean_files:
        raise SystemExit(
            f"No clean audio found in {args.clean_root}. "
            "Run `make dns-preview` first or pass --clean-root explicitly."
        )
    if not noise_files:
        raise SystemExit(
            f"No noise audio found in {args.noise_root}. "
            "Run `make dns-preview` first or pass --noise-root explicitly."
        )

    clean_out = args.output_root / "clean"
    noise_out = args.output_root / "noise"
    noisy_out = args.output_root / "noisy"
    manifest_path = args.output_root / "manifest.csv"

    if args.output_root.exists():
        shutil.rmtree(args.output_root)

    for path in (clean_out, noise_out, noisy_out):
        path.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "example_id",
                "clean_source",
                "noise_source",
                "snr_db",
                "sample_rate",
                "duration_sec",
                "clean_path",
                "noise_path",
                "noisy_path",
            ],
        )
        writer.writeheader()

        for idx in range(args.num_examples):
            clean_src = rng.choice(clean_files)
            noise_src = rng.choice(noise_files)
            snr_db = rng.uniform(args.snr_min, args.snr_max)

            clean = fit_length(load_mono(clean_src, args.sample_rate), target_len, rng)
            noise = fit_length(load_mono(noise_src, args.sample_rate), target_len, rng)
            noisy, scaled_noise = mix_at_snr(clean, noise, snr_db)

            example_id = numeric_id(idx, args.num_examples)
            clean_path = clean_out / f"{example_id}.wav"
            noise_path = noise_out / f"{example_id}.wav"
            noisy_path = noisy_out / f"{example_id}.wav"

            sf.write(clean_path, clean, args.sample_rate)
            sf.write(noise_path, scaled_noise, args.sample_rate)
            sf.write(noisy_path, noisy, args.sample_rate)

            writer.writerow(
                {
                    "example_id": example_id,
                    "clean_source": str(clean_src),
                    "noise_source": str(noise_src),
                    "snr_db": f"{snr_db:.3f}",
                    "sample_rate": args.sample_rate,
                    "duration_sec": args.duration_sec,
                    "clean_path": str(clean_path),
                    "noise_path": str(noise_path),
                    "noisy_path": str(noisy_path),
                }
            )

    print(f"Wrote {args.num_examples} paired examples to {args.output_root}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
