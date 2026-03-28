SHELL := /bin/bash
.DEFAULT_GOAL := help

VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

help:
	@echo "Available targets:"
	@echo "  make setup            - Create venv and install dev dependencies with CPU PyTorch"
	@echo "  make setup-cpu        - Same as make setup"
	@echo "  make setup-gpu        - Create venv and install dev dependencies with CUDA PyTorch"
	@echo "  make setup-base       - Create venv and install base dependencies with CPU PyTorch"
	@echo "  make setup-base-cpu   - Same as make setup-base"
	@echo "  make setup-base-gpu   - Create venv and install base dependencies with CUDA PyTorch"
	@echo "  make submodules       - Initialize/update git submodules"
	@echo "  make audit            - Inspect workspace, DNS assets, and tooling status"
	@echo "  make smoke            - Run environment import smoke test"
	@echo "  make subset           - Create a small clean/noise subset from DNS assets"
	@echo "  make debug-pairs      - Generate paired noisy-clean debug data locally"
	@echo "  make review-ui        - Launch browser UI for clean/noisy/result manual review"
	@echo "  make model-stats      - Report model size and runtime on a sample wav"
	@echo "  make metrics          - Compute PESQ/STOI/SI-SDR on paired folders"
	@echo "  make dnsmos           - Run DNSMOS on a folder of wav files"
	@echo "  make dns-dev-testset  - Download and stage DNS5 dev testset under data/"
	@echo "  make dns-metadata     - Download lightweight DNS metadata into data/"
	@echo "  make dns-preview      - Stage a tiny DNS audio preview set under data/samples/"
	@echo "  make baselines        - Download baseline model code and checkpoints into models/baseline/"
	@echo "  make dns-benchmark-sample - Prepare a small noisy sample from DNS dev testset"
	@echo "  make dnsmos-dev-sample - Run DNSMOS on the staged DNS noisy sample"
	@echo "  make dns-headset      - Run DNS5 headset download script (review script first)"
	@echo "  make clean-venv       - Remove virtual environment"

setup:
	PYTORCH_FLAVOR=cpu bash scripts/setup_env.sh dev

setup-cpu:
	PYTORCH_FLAVOR=cpu bash scripts/setup_env.sh dev

setup-gpu:
	PYTORCH_FLAVOR=gpu bash scripts/setup_env.sh dev

setup-base:
	PYTORCH_FLAVOR=cpu bash scripts/setup_env.sh base

setup-base-cpu:
	PYTORCH_FLAVOR=cpu bash scripts/setup_env.sh base

setup-base-gpu:
	PYTORCH_FLAVOR=gpu bash scripts/setup_env.sh base

submodules:
	bash scripts/bootstrap_submodules.sh

audit:
	$(PYTHON) scripts/audit_workspace.py

smoke:
	bash scripts/smoke_test.sh

subset:
	$(PYTHON) scripts/create_dns_subset.py --num-clean 200 --num-noise 200

debug-pairs:
	$(PYTHON) scripts/make_debug_dataset.py --num-examples $${NUM_EXAMPLES:-128}

review-ui:
	$(PYTHON) scripts/audio_review_app.py \
		--clean-dir $${CLEAN_DIR:-data/samples/dns_preview/clean} \
		--noisy-dir $${NOISY_DIR:-data/samples/dns_preview/noisy} \
		--result-dir $${RESULT_DIR:-experiments/reports/nsnet2_dns_preview} \
		--host $${HOST:-127.0.0.1} \
		--port $${PORT:-8008}

model-stats:
	$(PYTHON) evaluation/model_stats.py \
		--model $${MODEL:-nsnet2} \
		--sample-wav $${SAMPLE_WAV:-data/samples/dns_preview/noisy/01.wav} \
		--device $${DEVICE:-auto} \
		--runs $${RUNS:-5} \
		--warmup $${WARMUP:-1} \
		--output-json $${OUTPUT:-experiments/reports/$${MODEL:-nsnet2}_model_stats.json}

metrics:
	$(PYTHON) evaluation/metrics.py --enhanced-dir $${ENHANCED_DIR:-data/processed/debug/noisy} --clean-dir $${CLEAN_DIR:-data/processed/debug/clean} --output $${OUTPUT:-experiments/reports/debug_metrics.csv}

dnsmos:
	$(PYTHON) scripts/run_dnsmos.py --input-dir $${INPUT_DIR:-data/processed/debug/noisy} --output-csv $${OUTPUT:-experiments/reports/debug_dnsmos.csv}

dns-dev-testset:
	$(PYTHON) scripts/fetch_dns_dev_testset.py

dns-metadata:
	$(PYTHON) scripts/fetch_dns_metadata.py

dns-preview:
	$(PYTHON) scripts/stage_dns_preview.py --prefer-remote --random-sample --seed $${SEED:-42} --limit $${LIMIT:-10}

baselines:
	$(PYTHON) scripts/fetch_baseline_models.py

dns-benchmark-sample:
	$(PYTHON) scripts/prepare_dns_benchmark_sample.py --limit $${LIMIT:-16}

dnsmos-dev-sample:
	$(PYTHON) scripts/run_dnsmos.py --input-dir $${INPUT_DIR:-data/benchmarks/dns5_dev_sample/noisy} --output-csv $${OUTPUT:-experiments/reports/dns5_dev_sample_dnsmos.csv}

# DNS scripts may be dry-run by default. Review downloader script before running.
dns-headset:
	cd submodules/DNS-Challenge && chmod +x download-dns-challenge-5-headset-training.sh && ./download-dns-challenge-5-headset-training.sh

clean-venv:
	rm -rf .venv
