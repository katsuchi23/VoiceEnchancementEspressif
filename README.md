# Low-Power Voice Enhancement for Espressif

This repository is a practical starter workspace for a lightweight speech enhancement project aimed at Espressif deployment.

Current focus:
- standard speech enhancement / noise suppression
- small local debug datasets first
- reproducible baseline comparison with `NSNet2` and `GTCRN`
- offline evaluation before quantization and ESP-DL export

The original brief is included as [Espressif Industry Projects 2025.pdf](./Espressif%20Industry%20Projects%202025.pdf).

## What This Repo Does

This repo already supports:
- environment setup for CPU or GPU PyTorch
- DNS-Challenge as a git submodule
- tiny preview dataset generation under `data/samples/`
- generated paired debug data under `data/processed/`
- downloading staged baseline models into `models/baseline/`
- offline inference for `NSNet2` and `GTCRN`
- objective paired metrics: PESQ, STOI, SI-SDR
- DNSMOS scoring
- a lightweight browser UI for manual listening and comparison
- model size and runtime checks before quantization

What is still scaffolding:
- custom model training loop refinement
- quantization/export flow for ESP-DL
- embedded deployment code

## Repository Layout

```text
.
├── data/
│   ├── raw/                 # Raw downloads and staged upstream assets
│   ├── samples/             # Tiny clean/noisy preview set for quick iteration
│   ├── subset/              # Optional small clean/noise subset from DNS assets
│   ├── processed/           # Generated paired debug data
│   └── benchmarks/          # DNS filelists and staged benchmark helpers
├── evaluation/
│   ├── inference.py         # Offline inference runner for baseline models
│   ├── metrics.py           # PESQ / STOI / SI-SDR on paired folders
│   └── model_stats.py       # Model size and runtime summary
├── experiments/
│   ├── checkpoints/
│   ├── logs/
│   └── reports/
├── models/
│   ├── baseline/            # Staged upstream baselines and checkpoints
│   └── custom/              # Your own model code later
├── quantization/
├── requirements/
├── scripts/
├── submodules/
│   └── DNS-Challenge/
├── training/
├── Makefile
└── README.md
```

## Fresh Clone

Clone with submodules:

```bash
git clone --recurse-submodules <your-repo-url>
cd espressif_project
```

If the repo is already cloned:

```bash
make submodules
```

## Environment Setup

### System prerequisites

Ubuntu / Debian example:

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip git build-essential ffmpeg poppler-utils
```

### Python environment

CPU setup:

```bash
make setup
```

Explicit GPU setup:

```bash
make setup-gpu
```

Activate the environment:

```bash
source .venv/bin/activate
```

Verify imports:

```bash
make smoke
```

Notes:
- `make setup` defaults to CPU PyTorch
- `make setup-gpu` installs the CUDA PyTorch wheel
- the default CUDA wheel channel is `cu124`
- override it if needed:

```bash
PYTORCH_CUDA_FLAVOR=cu121 make setup-gpu
```

## Recommended First-Time Workflow

Run these in order:

```bash
make submodules
make setup-gpu
source .venv/bin/activate
make smoke
make audit
make dns-preview
make baselines
```

That gives you:
- the DNS submodule
- a working Python environment
- a 10-file clean/noisy preview set
- the staged baseline models

## Data Strategy

The project intentionally starts small.

### 1. Preview set

Build a tiny aligned set for local debugging:

```bash
make dns-preview
```

Default output:
- `data/samples/dns_preview/clean`
- `data/samples/dns_preview/noise`
- `data/samples/dns_preview/noisy`
- `data/samples/dns_preview/manifest.csv`

By default it creates `10` files named:
- `01.wav`
- `02.wav`
- ...

Custom sample count:

```bash
SEED=7 LIMIT=15 make dns-preview
```

### 2. Generated paired debug set

If you want a larger synthetic paired set derived from the preview pool:

```bash
NUM_EXAMPLES=128 make debug-pairs
```

Output:
- `data/processed/debug/clean`
- `data/processed/debug/noise`
- `data/processed/debug/noisy`
- `data/processed/debug/manifest.csv`

### 3. Full DNS assets later

If you want full DNS benchmark or training assets later:

```bash
make dns-metadata
make dns-dev-testset
```

The DNS filelists under `data/benchmarks/` are metadata only, not actual audio.

More details are in [data/README.md](./data/README.md).

## Baseline Models

Two baselines from the project brief are staged locally:
- `NSNet2`
- `GTCRN`

Fetch them into `models/baseline/`:

```bash
make baselines
```

Result:
- `models/baseline/nsnet2/`
- `models/baseline/gtcrn/`

Source pinning is recorded in:
- `models/baseline/nsnet2/SOURCE.md`
- `models/baseline/gtcrn/SOURCE.md`

Important note:
- `NSNet2` is staged as an ONNX model
- `GTCRN` is staged as a PyTorch checkpoint

## Offline Inference

The shared offline runner is:

```bash
python evaluation/inference.py --help
```

### Run NSNet2 on the 10 noisy preview files

```bash
python evaluation/inference.py \
  --model nsnet2 \
  --input-dir data/samples/dns_preview/noisy \
  --output-dir experiments/reports/nsnet2_dns_preview
```

### Run GTCRN on the same files

```bash
python evaluation/inference.py \
  --model gtcrn \
  --input-dir data/samples/dns_preview/noisy \
  --output-dir experiments/reports/gtcrn_dns_preview \
  --device auto
```

Both commands match files by directory contents. For the preview set, that means `01.wav` through `10.wav`.

## Manual Listening UI

To compare `clean`, `noisy`, and `result` side by side in the browser:

```bash
make review-ui
```

Default mapping:
- clean: `data/samples/dns_preview/clean`
- noisy: `data/samples/dns_preview/noisy`
- result: `experiments/reports/nsnet2_dns_preview`

Open:

```text
http://127.0.0.1:8008
```

To review GTCRN results instead:

```bash
RESULT_DIR=experiments/reports/gtcrn_dns_preview make review-ui
```

The UI:
- aligns files by shared filename
- gives 3 audio players per row
- supports notes and quick ratings
- stores notes in browser local storage
- can export notes as JSON

### Open from another device on the same network

```bash
HOST=0.0.0.0 PORT=8008 make review-ui
hostname -I
```

Then open:

```text
http://<your-local-ip>:8008
```

## Objective Metrics

### Paired intrusive metrics

Use when you have aligned clean references:

```bash
make metrics
```

Default:
- enhanced: `data/processed/debug/noisy`
- clean: `data/processed/debug/clean`

Custom example:

```bash
ENHANCED_DIR=experiments/reports/nsnet2_dns_preview \
CLEAN_DIR=data/samples/dns_preview/clean \
make metrics
```

### DNSMOS

Run DNSMOS on a folder of wav files:

```bash
make dnsmos
```

Custom example:

```bash
INPUT_DIR=experiments/reports/nsnet2_dns_preview \
OUTPUT=experiments/reports/nsnet2_dnsmos.csv \
make dnsmos
```

## Model Size and Runtime Check

To inspect model size and practical runtime before quantization:

```bash
make model-stats
```

Default model:
- `nsnet2`

Run GTCRN instead:

```bash
MODEL=gtcrn make model-stats
```

This reports:
- model file size
- parameter count
- parameter memory
- device used
- average latency
- total processing time
- real-time factor
- speed relative to real time

JSON reports are written to `experiments/reports/`.

### What real-time factor means

```text
RTF = processing time / audio duration
```

Interpretation:
- `RTF < 1.0` means faster than real-time
- `RTF = 1.0` means exactly real-time
- `RTF > 1.0` means slower than real-time

## Useful Make Targets

Show the command list:

```bash
make help
```

Most useful targets:

```bash
make submodules
make setup
make setup-gpu
make smoke
make audit
make dns-preview
make debug-pairs
make baselines
make review-ui
make metrics
make dnsmos
make model-stats
```

## Notes About Publishing

This repo intentionally does not commit:
- downloaded datasets under `data/`
- virtual environments
- large generated artifacts

The staged baseline binaries are reproducible through:

```bash
make baselines
```

So a fresh GitHub clone can rebuild the working baseline environment without storing the large model files directly in git history.

## Next Practical Steps

1. Keep `dns_preview` as the lightweight listening/debugging set.
2. Use `NSNet2` and `GTCRN` as reference baselines.
3. Add your own lightweight model under `models/custom/`.
4. Reuse `evaluation/inference.py`, `evaluation/metrics.py`, and `evaluation/model_stats.py` to compare it against the baselines.
5. Only then move into quantization and ESP-DL export.
