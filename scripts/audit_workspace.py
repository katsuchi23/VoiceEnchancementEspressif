#!/usr/bin/env python3
"""Inspect local project readiness for the ESP32 speech enhancement workflow."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DNS_REPO = PROJECT_ROOT / "submodules" / "DNS-Challenge"


def status_line(label: str, ok: bool, detail: str) -> str:
    state = "OK" if ok else "MISSING"
    return f"[{state}] {label}: {detail}"


def first_existing(*paths: Path) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def main() -> None:
    checks: list[str] = []

    checks.append(status_line("Project root", PROJECT_ROOT.exists(), str(PROJECT_ROOT)))
    checks.append(status_line("Git metadata", (PROJECT_ROOT / ".git").exists(), str(PROJECT_ROOT / ".git")))
    checks.append(status_line("README", (PROJECT_ROOT / "README.md").exists(), str(PROJECT_ROOT / "README.md")))
    checks.append(status_line("Virtual environment", (PROJECT_ROOT / ".venv" / "bin" / "python").exists(), str(PROJECT_ROOT / ".venv")))

    checks.append(status_line("DNS submodule", DNS_REPO.exists(), str(DNS_REPO)))
    checks.append(
        status_line(
            "DNSMOS models",
            (DNS_REPO / "DNSMOS" / "DNSMOS" / "sig_bak_ovr.onnx").exists(),
            str(DNS_REPO / "DNSMOS" / "DNSMOS" / "sig_bak_ovr.onnx"),
        )
    )
    checks.append(status_line("WAcc script", (DNS_REPO / "WAcc" / "WAcc.py").exists(), str(DNS_REPO / "WAcc" / "WAcc.py")))

    clean_root = first_existing(
        DNS_REPO / "datasets_fullband" / "clean_fullband",
        DNS_REPO / "clean_fullband",
    )
    noise_root = first_existing(
        DNS_REPO / "datasets_fullband" / "noise_fullband",
        DNS_REPO / "noise_fullband",
    )
    rir_root = first_existing(
        DNS_REPO / "datasets" / "impulse_responses",
        DNS_REPO / "datasets_fullband" / "impulse_responses",
        DNS_REPO / "impulse_responses",
    )

    checks.append(status_line("DNS clean speech assets", clean_root is not None, str(clean_root or "not downloaded")))
    checks.append(status_line("DNS noise assets", noise_root is not None, str(noise_root or "not downloaded")))
    checks.append(status_line("DNS RIR assets", rir_root is not None, str(rir_root or "not downloaded")))

    synth_cfg = DNS_REPO / "noisyspeech_synthesizer.cfg"
    missing_metadata = [
        DNS_REPO / "datasets" / "acoustic_params" / "RIR_table_simple.csv",
        DNS_REPO / "datasets" / "acoustic_params" / "cleanspeech_table_t60_c50.csv",
    ]
    checks.append(status_line("DNS synthesizer config", synth_cfg.exists(), str(synth_cfg)))
    checks.append(
        status_line(
            "DNS synthesizer metadata CSVs",
            all(path.exists() for path in missing_metadata),
            ", ".join(str(path) for path in missing_metadata),
        )
    )

    checks.append(status_line("Subset directory", (PROJECT_ROOT / "data" / "subset").exists(), str(PROJECT_ROOT / "data" / "subset")))
    checks.append(
        status_line(
            "Debug processed pairs",
            (PROJECT_ROOT / "data" / "processed" / "debug").exists(),
            str(PROJECT_ROOT / "data" / "processed" / "debug"),
        )
    )
    checks.append(
        status_line(
            "Training config",
            (PROJECT_ROOT / "training" / "configs" / "default.yaml").exists(),
            str(PROJECT_ROOT / "training" / "configs" / "default.yaml"),
        )
    )

    print("Workspace audit")
    print("================")
    for line in checks:
        print(line)

    print("\nNotes")
    print("-----")
    print("- DNS-Challenge in this revision includes dataset download scripts, DNSMOS, and WAcc tooling.")
    print("- It does not include a complete baseline training repo for NSNet2 or GTCRN.")
    print("- The upstream synthesizer references metadata CSVs that are not present by default in this checkout.")
    print("- Use the local debug dataset generator first, then move to the upstream DNS workflow once the missing metadata is resolved.")


if __name__ == "__main__":
    main()
