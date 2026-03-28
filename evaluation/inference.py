#!/usr/bin/env python3
"""Run offline enhancement inference with staged baseline models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import soundfile as sf
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_SUFFIXES = {".wav"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run enhancement inference on wav files.")
    parser.add_argument("--model", choices=("nsnet2", "gtcrn"), required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto", help="For GTCRN: auto, cpu, or cuda")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--model-path", type=Path, default=None, help="Optional explicit model/checkpoint path.")
    return parser.parse_args()


def list_audio_files(root: Path) -> list[Path]:
    return sorted(path for path in root.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES)


def load_mono(path: Path) -> tuple[torch.Tensor, int]:
    audio, sample_rate = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]
    return torch.from_numpy(audio), sample_rate


def save_audio(path: Path, audio: torch.Tensor, sample_rate: int) -> None:
    sf.write(path, audio.detach().cpu().numpy(), sample_rate)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def nsnet2_defaults() -> Path:
    return PROJECT_ROOT / "models" / "baseline" / "nsnet2" / "nsnet2-20ms-baseline.onnx"


def gtcrn_defaults() -> Path:
    return PROJECT_ROOT / "models" / "baseline" / "gtcrn" / "checkpoints" / "model_trained_on_dns3.tar"


def load_nsnet2(model_path: Path):
    module_dir = PROJECT_ROOT / "models" / "baseline" / "nsnet2"
    sys.path.insert(0, str(module_dir))
    from enhance_onnx import NSnet2Enhancer  # pylint: disable=import-error

    return NSnet2Enhancer(str(model_path))


def run_nsnet2(input_dir: Path, output_dir: Path, model_path: Path, sample_rate: int) -> None:
    enhancer = load_nsnet2(model_path)
    files = list_audio_files(input_dir)
    if not files:
        raise SystemExit(f"No wav files found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for path in files:
        audio, sr = load_mono(path)
        if sr != sample_rate:
            raise SystemExit(f"{path} has sample rate {sr}, expected {sample_rate}")
        enhanced = enhancer(audio.numpy(), sr)
        sf.write(output_dir / path.name, enhanced, sr)
        print(f"Enhanced {path.name} -> {output_dir / path.name}")


def load_gtcrn(model_path: Path, device: torch.device):
    module_dir = PROJECT_ROOT / "models" / "baseline" / "gtcrn"
    sys.path.insert(0, str(module_dir))
    from gtcrn import GTCRN  # pylint: disable=import-error

    checkpoint = torch.load(model_path, map_location=device)
    model = GTCRN().to(device).eval()
    model.load_state_dict(checkpoint["model"])
    return model


def run_gtcrn(input_dir: Path, output_dir: Path, model_path: Path, sample_rate: int, device_name: str) -> None:
    device = resolve_device(device_name)
    model = load_gtcrn(model_path, device)
    files = list_audio_files(input_dir)
    if not files:
        raise SystemExit(f"No wav files found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    window = torch.hann_window(512, device=device).pow(0.5)

    for path in files:
        audio, sr = load_mono(path)
        if sr != sample_rate:
            raise SystemExit(f"{path} has sample rate {sr}, expected {sample_rate}")

        audio = audio.to(device)
        spec = torch.stft(audio, 512, 256, 512, window, return_complex=True)
        spec_ri = torch.view_as_real(spec)
        with torch.no_grad():
            enhanced_spec_ri = model(spec_ri[None])[0]
        enhanced_spec = torch.view_as_complex(enhanced_spec_ri.contiguous())
        enhanced = torch.istft(enhanced_spec, 512, 256, 512, window, return_complex=False)
        save_audio(output_dir / path.name, enhanced, sr)
        print(f"Enhanced {path.name} -> {output_dir / path.name}")


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not input_dir.is_dir():
        raise SystemExit(f"Input dir does not exist: {input_dir}")

    if args.model == "nsnet2":
        model_path = (args.model_path or nsnet2_defaults()).resolve()
        run_nsnet2(input_dir, output_dir, model_path, args.sample_rate)
    else:
        model_path = (args.model_path or gtcrn_defaults()).resolve()
        run_gtcrn(input_dir, output_dir, model_path, args.sample_rate, args.device)


if __name__ == "__main__":
    main()
