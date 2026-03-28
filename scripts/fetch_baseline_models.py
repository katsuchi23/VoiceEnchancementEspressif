#!/usr/bin/env python3
"""Fetch pinned baseline model artifacts into models/baseline."""

from __future__ import annotations

import hashlib
from pathlib import Path

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = PROJECT_ROOT / "models" / "baseline"

GTCRN_COMMIT = "3862c44808dca492ea5a8a145d2dc2a1028d08c8"
NSNET2_COMMIT = "a2c7487e12d06d709aeebe5659c21bbf6e1a47aa"


MODELS: dict[str, dict[str, object]] = {
    "gtcrn": {
        "repo": "https://github.com/Xiaobin-Rong/gtcrn",
        "commit": GTCRN_COMMIT,
        "files": [
            "README.md",
            "LICENSE",
            "gtcrn.py",
            "checkpoints/model_trained_on_dns3.tar",
        ],
    },
    "nsnet2": {
        "repo": "https://github.com/microsoft/DNS-Challenge",
        "commit": NSNET2_COMMIT,
        "subdir": "NSNet2-baseline",
        "files": [
            "README.md",
            "enhance_onnx.py",
            "featurelib.py",
            "nsnet2-20ms-baseline.onnx",
        ],
    },
}


def raw_url(repo_url: str, commit: str, relpath: str) -> str:
    repo_path = repo_url.removeprefix("https://github.com/")
    return f"https://raw.githubusercontent.com/{repo_path}/{commit}/{relpath}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)


def patch_nsnet2_compatibility(target_dir: Path) -> None:
    featurelib_path = target_dir / "featurelib.py"
    text = featurelib_path.read_text()
    old = "from scipy.signal import blackman, blackmanharris\n"
    if old in text:
        featurelib_path.write_text(text.replace(old, ""))


def write_source_note(model_name: str, spec: dict[str, object], target_dir: Path) -> None:
    subdir = spec.get("subdir")
    files = spec["files"]
    lines = [
        f"# {model_name.upper()} baseline",
        "",
        f"Source repo: {spec['repo']}",
        f"Pinned commit: {spec['commit']}",
    ]
    if subdir:
        lines.append(f"Upstream subdirectory: {subdir}")
    lines.extend(
        [
            "",
            "Fetched files:",
        ]
    )
    for relpath in files:
        local_path = target_dir / relpath
        lines.append(f"- `{relpath}`")
        if local_path.exists():
            lines.append(f"  sha256: `{sha256(local_path)}`")
    lines.append("")
    lines.append("Files in this directory are copied from the pinned upstream revision for reproducibility.")
    (target_dir / "SOURCE.md").write_text("\n".join(lines) + "\n")


def fetch_model(model_name: str, spec: dict[str, object]) -> None:
    target_dir = BASELINE_ROOT / model_name
    target_dir.mkdir(parents=True, exist_ok=True)
    for legacy_name in ("infer.py", "run_nsnet2.py"):
        legacy_path = target_dir / legacy_name
        if legacy_path.exists():
            legacy_path.unlink()

    base_rel_prefix = spec.get("subdir")
    for relpath in spec["files"]:
        upstream_relpath = f"{base_rel_prefix}/{relpath}" if base_rel_prefix else relpath
        url = raw_url(spec["repo"], spec["commit"], upstream_relpath)
        destination = target_dir / relpath
        download_file(url, destination)
        print(f"Fetched {model_name}: {relpath}")

    if model_name == "nsnet2":
        patch_nsnet2_compatibility(target_dir)

    write_source_note(model_name, spec, target_dir)


def main() -> None:
    BASELINE_ROOT.mkdir(parents=True, exist_ok=True)
    for model_name, spec in MODELS.items():
        fetch_model(model_name, spec)
    print(f"Baseline models staged under {BASELINE_ROOT}")


if __name__ == "__main__":
    main()
