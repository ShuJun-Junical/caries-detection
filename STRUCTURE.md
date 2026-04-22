# Repository Structure Guide

This document explains the role of each major directory and key files.

## Root Files
- `README.md`: user-facing quick start, environment setup, and command examples.
- `AGENTS.md`: instructions and conventions for AI assistants.
- `pyproject.toml`: lightweight project metadata.
- `requirements-ultralytics.txt`: dependencies for Ultralytics workflows.
- `requirements-v5.txt`: dependencies for YOLOv5 workflows.
- `SLURM_CLUSTER_INFO.md`: cluster hardware and partition notes.

## Source Directories
- `scripts/common/`
  - `io_utils.py`: shared helpers for YAML loading, config merge, path handling, and timestamp tags.
  - (moved) `dataset_filters.py`: functionality moved to `tools/prepare_caries_only_data.py`; builds temporary training dataset views (for example, caries-only filtering) without modifying original dataset files.
- `scripts/train/`
  - `train_ultralytics.py`: train entry for v8, v11, latest. Default ignores class `other` (id=1); pass `--keep-others` to disable filtering. Pass `--use-attention` to inject CBAM attention blocks.
  - `train_yolov5.py`: train entry for local YOLOv5 clone. Default ignores class `other` (id=1); pass `--keep-others` to disable filtering. Pass `--use-attention` to switch to `yolov5s-transformer.yaml` (C3TR).
  - `run_all.py`: sequential smoke run across selected model families, supports `--keep-others` and `--use-attention` passthrough.
- `scripts/eval/`
  - `compare_test_models.py`: evaluate and compare metrics across model families, generate CSV and chart.
- `scripts/slurm/`
  - `train_ultralytics.sbatch`: Ultralytics training on standard GPU nodes. Slurm wrappers focus on device/venv/family; training-related flags (e.g. `use_attention`, `workers`) are read from the consolidated YAML (`configs/models.yaml`) or passed via the training CLI when running directly.
  - `train_ultralytics_p100.sbatch`: Ultralytics training for P100-compatible environment. Slurm wrappers focus on device/venv/family; training-related flags are configured in YAML or via CLI.
  - `train_yolov5.sbatch`: YOLOv5 training on standard GPU nodes. Slurm wrappers focus on device/venv; configure training flags in YAML or via CLI.
  - `train_yolov5_p100.sbatch`: YOLOv5 training for P100-compatible environment. Slurm wrappers focus on device/venv; configure training flags in YAML or via CLI.
  - `submit_examples.sh`: sample batch submission helper.

## Config and Data
- `configs/`
  - `data.caries.yaml`: dataset root, split paths, and class names.
  - `models.yaml`: consolidated per-family model defaults (preferred). Contains subsections `yolov5`, `yolov8`, `yolov11`, `latest` with only model-specific overrides (checkpoint, data, project/name, and family-specific flags).
  - Legacy per-family files (kept for compatibility): `models.yolov8.yaml`, `models.yolov11.yaml`, `models.latest.yaml`, `models.yolov5.yaml`, `models.yolov5.p100.yaml`.
- `dataset/`
  - `train/`, `val/`, `test/` with `images/` and `labels/` in YOLO format.

## Runtime and Artifacts
- `checkpoints/`: local initial checkpoints for training starts.
- `runs/`: generated training/evaluation outputs.
  - `runs/tmp_data/caries_only/`: auto-generated temporary single-class dataset view used by default training wrappers.
- `logs/`: runtime logs, especially Slurm stdout/stderr in `logs/slurm/`.

## Tooling and Vendor Code
- `tools/`
  - `prepare_caries_only_data.py`: CLI/tool to create a temporary single-class (caries-only) dataset view; replaces `scripts/common/dataset_filters.py` and `scripts/train/prepare_caries_only_data.py`.
  - `check_dataset.py`: dataset integrity and label format checks.
  - `ddp_test.py`, `ddp_gloo_test.py`, `ddp_gloo_cuda_test.py`: distributed environment probes.
- `third_party/yolov5/`: vendored upstream YOLOv5 code, treated as external source.

## Current Cleanup Policy
- Keep only requested artifacts under `runs/`.
- Keep `logs/` clean unless logs are needed for active debugging.
