#!/usr/bin/env python3
"""Report baseline model size and runtime characteristics."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import onnx
import soundfile as sf
import torch
from onnx import numpy_helper


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PRESETS = {
    "gtcrn": {
        "backend": "pytorch",
        "model_path": PROJECT_ROOT / "models" / "baseline" / "gtcrn" / "checkpoints" / "model_trained_on_dns3.tar",
        "module_dir": PROJECT_ROOT / "models" / "baseline" / "gtcrn",
        "module_name": "gtcrn",
        "class_name": "GTCRN",
        "checkpoint_key": "model",
    },
    "nsnet2": {
        "backend": "onnx",
        "model_path": PROJECT_ROOT / "models" / "baseline" / "nsnet2" / "nsnet2-20ms-baseline.onnx",
        "module_dir": PROJECT_ROOT / "models" / "baseline" / "nsnet2",
        "module_name": "enhance_onnx",
        "class_name": "NSnet2Enhancer",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect model size and runtime on a sample wav.")
    parser.add_argument("--model", choices=sorted(PRESETS), required=True)
    parser.add_argument(
        "--sample-wav",
        type=Path,
        default=PROJECT_ROOT / "data" / "samples" / "dns_preview" / "noisy" / "01.wav",
    )
    parser.add_argument("--device", default="auto", help="For PyTorch models: auto, cpu, or cuda")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def load_mono(path: Path) -> tuple[torch.Tensor, int]:
    audio, sample_rate = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]
    return torch.from_numpy(audio), sample_rate


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def timing_summary(duration_sec: float, timings_ms: list[float]) -> dict[str, float]:
    avg_ms = sum(timings_ms) / len(timings_ms)
    total_ms = sum(timings_ms)
    rtf = (avg_ms / 1000.0) / duration_sec
    return {
        "audio_duration_sec": duration_sec,
        "avg_latency_ms": avg_ms,
        "total_processing_ms": total_ms,
        "total_processing_sec": total_ms / 1000.0,
        "min_latency_ms": min(timings_ms),
        "max_latency_ms": max(timings_ms),
        "real_time_factor": rtf,
        "x_realtime": 1.0 / rtf if rtf > 0 else float("inf"),
    }


def load_gtcrn(spec: dict[str, object], device: torch.device):
    sys.path.insert(0, str(spec["module_dir"]))
    module = __import__(str(spec["module_name"]), fromlist=[str(spec["class_name"])])
    cls = getattr(module, str(spec["class_name"]))
    checkpoint = torch.load(spec["model_path"], map_location=device)
    model = cls().to(device).eval()
    model.load_state_dict(checkpoint[str(spec["checkpoint_key"])])
    return model


def measure_gtcrn(spec: dict[str, object], sample_wav: Path, sample_rate: int, device_name: str, runs: int, warmup: int):
    device = resolve_device(device_name)
    model = load_gtcrn(spec, device)
    audio, sr = load_mono(sample_wav)
    if sr != sample_rate:
        raise SystemExit(f"Sample wav {sample_wav} has sample rate {sr}, expected {sample_rate}")

    audio = audio.to(device)
    window = torch.hann_window(512, device=device).pow(0.5)

    def enhance_once() -> None:
        spec_c = torch.stft(audio, 512, 256, 512, window, return_complex=True)
        spec_ri = torch.view_as_real(spec_c)
        with torch.no_grad():
            enhanced_ri = model(spec_ri[None])[0]
        enhanced_c = torch.view_as_complex(enhanced_ri.contiguous())
        torch.istft(enhanced_c, 512, 256, 512, window, return_complex=False)

    for _ in range(warmup):
        enhance_once()
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    timings_ms: list[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        enhance_once()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timings_ms.append((time.perf_counter() - start) * 1000.0)

    param_count = sum(parameter.numel() for parameter in model.parameters())
    param_bytes = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
    return {
        "backend": "pytorch",
        "device": str(device),
        "parameter_count": int(param_count),
        "parameter_bytes": int(param_bytes),
        **timing_summary(len(audio) / sample_rate, timings_ms),
    }


def load_nsnet2(spec: dict[str, object]):
    sys.path.insert(0, str(spec["module_dir"]))
    module = __import__(str(spec["module_name"]), fromlist=[str(spec["class_name"])])
    cls = getattr(module, str(spec["class_name"]))
    return cls(str(spec["model_path"]))


def measure_nsnet2(spec: dict[str, object], sample_wav: Path, sample_rate: int, runs: int, warmup: int):
    enhancer = load_nsnet2(spec)
    audio, sr = load_mono(sample_wav)
    if sr != sample_rate:
        raise SystemExit(f"Sample wav {sample_wav} has sample rate {sr}, expected {sample_rate}")

    wav = audio.numpy()
    for _ in range(warmup):
        enhancer(wav, sr)

    timings_ms: list[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        enhancer(wav, sr)
        timings_ms.append((time.perf_counter() - start) * 1000.0)

    model = onnx.load(spec["model_path"], load_external_data=False)
    initializers = list(model.graph.initializer)
    param_count = sum(int(numpy_helper.to_array(initializer).size) for initializer in initializers)
    param_bytes = sum(int(numpy_helper.to_array(initializer).nbytes) for initializer in initializers)
    return {
        "backend": "onnx",
        "device": "cpu",
        "parameter_count": int(param_count),
        "parameter_bytes": int(param_bytes),
        "onnx_node_count": len(model.graph.node),
        **timing_summary(len(audio) / sample_rate, timings_ms),
    }


def main() -> None:
    args = parse_args()
    spec = dict(PRESETS[args.model])
    model_path = Path(spec["model_path"]).resolve()
    sample_wav = args.sample_wav.resolve()

    if not model_path.exists():
        raise SystemExit(f"Model file not found: {model_path}")
    if not sample_wav.exists():
        raise SystemExit(f"Sample wav not found: {sample_wav}")

    if spec["backend"] == "pytorch":
        runtime = measure_gtcrn(spec, sample_wav, args.sample_rate, args.device, args.runs, args.warmup)
    else:
        runtime = measure_nsnet2(spec, sample_wav, args.sample_rate, args.runs, args.warmup)

    report = {
        "model": args.model,
        "model_path": str(model_path),
        "model_file_size_mb": file_size_mb(model_path),
        "sample_wav": str(sample_wav),
        "sample_rate": args.sample_rate,
        "runs": args.runs,
        "warmup": args.warmup,
        **runtime,
    }

    print(f"Model: {report['model']}")
    print(f"Path: {report['model_path']}")
    print(f"File size: {report['model_file_size_mb']:.3f} MB")
    print(f"Parameters: {report['parameter_count']:,}")
    print(f"Parameter memory: {report['parameter_bytes'] / (1024 * 1024):.3f} MB")
    if "onnx_node_count" in report:
        print(f"ONNX nodes: {report['onnx_node_count']}")
    print(f"Device: {report['device']}")
    print(f"Sample clip: {report['sample_wav']}")
    print(f"Audio duration: {report['audio_duration_sec']:.3f} s")
    print(f"Average latency: {report['avg_latency_ms']:.3f} ms")
    print(f"Total processing time: {report['total_processing_sec']:.3f} s ({report['total_processing_ms']:.3f} ms)")
    print(f"Latency range: {report['min_latency_ms']:.3f} ms .. {report['max_latency_ms']:.3f} ms")
    print(f"Real-time factor: {report['real_time_factor']:.4f}")
    print(f"Speed: {report['x_realtime']:.2f}x real-time")

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2) + "\n")
        print(f"JSON report: {args.output_json}")


if __name__ == "__main__":
    main()
