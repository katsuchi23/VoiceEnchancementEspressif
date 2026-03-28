#!/usr/bin/env python3
"""Stage a tiny paired speech-enhancement preview set for local debugging."""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from remotezip import RemoteZip


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DEV_ROOT = PROJECT_ROOT / "data" / "raw" / "dns_challenge" / "V5_dev_testset"
PREVIEW_ROOT = PROJECT_ROOT / "data" / "samples" / "dns_preview"
DEV_TESTSET_URL = "https://dnschallengepublic.blob.core.windows.net/dns5archive/V5_dev_testset.zip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage a tiny paired DNS enhancement preview set.")
    parser.add_argument("--source-root", type=Path, default=RAW_DEV_ROOT)
    parser.add_argument("--output-root", type=Path, default=PREVIEW_ROOT)
    parser.add_argument("--track", choices=("track1", "track2"), default="track1")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--prefer-remote", action="store_true", help="Fetch directly from the remote DNS zip.")
    parser.add_argument("--random-sample", action="store_true", help="Randomly sample clips instead of taking the first N.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducible random sampling.")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--snr-min", type=float, default=0.0)
    parser.add_argument("--snr-max", type=float, default=15.0)
    return parser.parse_args()


def find_track_root(source_root: Path, track: str) -> Path:
    target_name = "Track1_Headset" if track == "track1" else "Track2_Speakerphone"
    matches = [path for path in source_root.rglob(target_name) if path.is_dir()]
    if not matches:
        raise SystemExit(
            f"Could not find {target_name} under {source_root}. "
            "Run `make dns-dev-testset` after the archive download finishes."
        )
    return sorted(matches)[0]


def find_named_dir(root: Path, keyword: str) -> Path | None:
    matches = [path for path in root.rglob("*") if path.is_dir() and keyword.lower() in path.name.lower()]
    return sorted(matches)[0] if matches else None


def choose_names(names: list[str], limit: int, random_sample: bool, seed: int) -> list[str]:
    if not names:
        return []
    ordered = sorted(names)
    if not random_sample:
        return ordered[:limit]
    rng = random.Random(seed)
    if limit >= len(ordered):
        rng.shuffle(ordered)
        return ordered
    return sorted(rng.sample(ordered, limit))


def choose_paths(paths: list[Path], limit: int, random_sample: bool, seed: int) -> list[Path]:
    if not paths:
        return []
    ordered = sorted(paths)
    if not random_sample:
        return ordered[:limit]
    rng = random.Random(seed)
    if limit >= len(ordered):
        rng.shuffle(ordered)
        return ordered
    return sorted(rng.sample(ordered, limit))


def load_mono(path: Path, sample_rate: int) -> np.ndarray:
    audio, _ = librosa.load(path, sr=sample_rate, mono=True)
    return audio.astype(np.float32)


def rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio)) + 1e-12))


def pink_noise(num_samples: int, rng: np.random.Generator) -> np.ndarray:
    white = rng.standard_normal(num_samples)
    spectrum = np.fft.rfft(white)
    scale = np.sqrt(np.arange(1, len(spectrum) + 1, dtype=np.float32))
    pink = np.fft.irfft(spectrum / scale, n=num_samples)
    return pink.astype(np.float32)


def brown_noise(num_samples: int, rng: np.random.Generator) -> np.ndarray:
    white = rng.standard_normal(num_samples).astype(np.float32)
    brown = np.cumsum(white)
    brown /= np.max(np.abs(brown)) + 1e-12
    return brown.astype(np.float32)


def hum_noise(num_samples: int, sample_rate: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(num_samples, dtype=np.float32) / sample_rate
    base = rng.choice([50.0, 60.0])
    hum = (
        0.7 * np.sin(2 * np.pi * base * t)
        + 0.2 * np.sin(2 * np.pi * 2 * base * t)
        + 0.1 * np.sin(2 * np.pi * 3 * base * t)
    )
    return hum.astype(np.float32)


def babble_like_noise(num_samples: int, sample_rate: int, rng: np.random.Generator) -> np.ndarray:
    noise = np.zeros(num_samples, dtype=np.float32)
    t = np.arange(num_samples, dtype=np.float32) / sample_rate
    for _ in range(6):
        freq = rng.uniform(120.0, 1200.0)
        amp = rng.uniform(0.1, 0.4)
        mod = 0.5 * (1.0 + np.sin(2 * np.pi * rng.uniform(2.0, 6.0) * t + rng.uniform(0, 2 * np.pi)))
        noise += amp * mod * np.sin(2 * np.pi * freq * t + rng.uniform(0, 2 * np.pi))
    return noise.astype(np.float32)


def choose_noise(num_samples: int, sample_rate: int, rng: np.random.Generator) -> tuple[str, np.ndarray]:
    noise_type = rng.choice(["white", "pink", "brown", "hum", "babble"])
    if noise_type == "white":
        noise = rng.standard_normal(num_samples).astype(np.float32)
    elif noise_type == "pink":
        noise = pink_noise(num_samples, rng)
    elif noise_type == "brown":
        noise = brown_noise(num_samples, rng)
    elif noise_type == "hum":
        noise = hum_noise(num_samples, sample_rate, rng)
    else:
        noise = babble_like_noise(num_samples, sample_rate, rng)
    noise /= np.max(np.abs(noise)) + 1e-12
    return noise_type, noise.astype(np.float32)


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


def prepare_dirs(output_root: Path) -> tuple[Path, Path, Path]:
    if output_root.exists():
        shutil.rmtree(output_root)
    clean_out = output_root / "clean"
    noise_out = output_root / "noise"
    noisy_out = output_root / "noisy"
    clean_out.mkdir(parents=True, exist_ok=True)
    noise_out.mkdir(parents=True, exist_ok=True)
    noisy_out.mkdir(parents=True, exist_ok=True)
    return clean_out, noise_out, noisy_out


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "example_id",
                "clean_source",
                "noise_type",
                "snr_db",
                "sample_rate",
                "clean_path",
                "noise_path",
                "noisy_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def numeric_id(index: int, total: int) -> str:
    width = max(2, len(str(total)))
    return f"{index + 1:0{width}d}"


def stage_remote(args: argparse.Namespace) -> None:
    track_name = "Track1_Headset" if args.track == "track1" else "Track2_Speakerphone"
    clean_prefix = f"V5_dev_testset/{track_name}/enrol/"
    clean_out, noise_out, noisy_out = prepare_dirs(args.output_root)

    rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)
    rows: list[dict[str, str]] = []

    with RemoteZip(DEV_TESTSET_URL, support_suffix_range=False) as archive:
        names = archive.namelist()
        clean_names = choose_names(
            [name for name in names if name.startswith(clean_prefix) and name.endswith(".wav")],
            args.limit,
            args.random_sample,
            args.seed,
        )

        if not clean_names:
            raise SystemExit(f"No clean reference clips found in remote archive for {track_name}")

        total = len(clean_names)
        for idx, name in enumerate(clean_names):
            example_id = numeric_id(idx, total)
            clean_tmp = args.output_root / f"{example_id}_tmp.wav"
            with archive.open(name) as src, clean_tmp.open("wb") as handle:
                shutil.copyfileobj(src, handle)

            clean = load_mono(clean_tmp, args.sample_rate)
            clean_tmp.unlink()

            noise_type, noise = choose_noise(len(clean), args.sample_rate, np_rng)
            snr_db = rng.uniform(args.snr_min, args.snr_max)
            noisy, scaled_noise = mix_at_snr(clean, noise, snr_db)

            clean_path = clean_out / f"{example_id}.wav"
            noise_path = noise_out / f"{example_id}.wav"
            noisy_path = noisy_out / f"{example_id}.wav"

            sf.write(clean_path, clean, args.sample_rate)
            sf.write(noise_path, scaled_noise, args.sample_rate)
            sf.write(noisy_path, noisy, args.sample_rate)

            rows.append(
                {
                    "example_id": example_id,
                    "clean_source": name,
                    "noise_type": noise_type,
                    "snr_db": f"{snr_db:.3f}",
                    "sample_rate": str(args.sample_rate),
                    "clean_path": str(clean_path),
                    "noise_path": str(noise_path),
                    "noisy_path": str(noisy_path),
                }
            )

    write_manifest(args.output_root / "manifest.csv", rows)
    print(f"Sampling mode: {'random' if args.random_sample else 'first_n'} (seed={args.seed})")
    print(f"Staged clean clips: {len(rows)} -> {clean_out}")
    print(f"Staged noise clips: {len(rows)} -> {noise_out}")
    print(f"Staged noisy clips: {len(rows)} -> {noisy_out}")
    print(f"Manifest: {args.output_root / 'manifest.csv'}")


def stage_local(args: argparse.Namespace) -> None:
    track_root = find_track_root(args.source_root, args.track)
    clean_dir = find_named_dir(track_root, "enrol") or find_named_dir(track_root, "clean")
    clean_files = choose_paths(
        list(clean_dir.rglob("*.wav")) if clean_dir is not None else [],
        args.limit,
        args.random_sample,
        args.seed,
    )
    if not clean_files:
        raise SystemExit(f"No clean reference clips found under {track_root}")

    clean_out, noise_out, noisy_out = prepare_dirs(args.output_root)
    rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)
    rows: list[dict[str, str]] = []

    total = len(clean_files)
    for idx, clean_src in enumerate(clean_files):
        example_id = numeric_id(idx, total)
        clean = load_mono(clean_src, args.sample_rate)
        noise_type, noise = choose_noise(len(clean), args.sample_rate, np_rng)
        snr_db = rng.uniform(args.snr_min, args.snr_max)
        noisy, scaled_noise = mix_at_snr(clean, noise, snr_db)

        clean_path = clean_out / f"{example_id}.wav"
        noise_path = noise_out / f"{example_id}.wav"
        noisy_path = noisy_out / f"{example_id}.wav"

        sf.write(clean_path, clean, args.sample_rate)
        sf.write(noise_path, scaled_noise, args.sample_rate)
        sf.write(noisy_path, noisy, args.sample_rate)

        rows.append(
            {
                "example_id": example_id,
                "clean_source": str(clean_src),
                "noise_type": noise_type,
                "snr_db": f"{snr_db:.3f}",
                "sample_rate": str(args.sample_rate),
                "clean_path": str(clean_path),
                "noise_path": str(noise_path),
                "noisy_path": str(noisy_path),
            }
        )

    write_manifest(args.output_root / "manifest.csv", rows)
    print(f"Sampling mode: {'random' if args.random_sample else 'first_n'} (seed={args.seed})")
    print(f"Track root: {track_root}")
    print(f"Staged clean clips: {len(rows)} -> {clean_out}")
    print(f"Staged noise clips: {len(rows)} -> {noise_out}")
    print(f"Staged noisy clips: {len(rows)} -> {noisy_out}")
    print(f"Manifest: {args.output_root / 'manifest.csv'}")


def main() -> None:
    args = parse_args()
    if args.prefer_remote or not args.source_root.exists():
        stage_remote(args)
    else:
        stage_local(args)


if __name__ == "__main__":
    main()
