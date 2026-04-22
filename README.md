# Caries Detection Training

Training and evaluation workflows for dental caries detection across YOLOv5, YOLOv8, YOLOv11, and latest Ultralytics family models.

## Overview

- Dataset format: YOLO detection format
- Canonical configs: configs directory
- Canonical entry scripts: scripts directory
- Runtime artifacts: runs and logs directories

## Project Layout

- configs: dataset, model, and hyperparameter YAML files
- scripts/common: shared helper functions
- scripts/train: training entry points
- scripts/eval: evaluation and comparison scripts
- scripts/slurm: Slurm submit wrappers
- tools: dataset and distributed environment checks
- checkpoints: local starter weights
- third_party/yolov5: upstream vendor code

See [STRUCTURE.md](STRUCTURE.md) for a complete directory-by-directory guide.

## Prerequisites

- Linux
- Python 3.10+
- uv installed

## Environment Setup

```bash
cd /home/junical/caries-detection

uv venv .venv-ultra
uv venv .venv-v5

uv pip install --python .venv-ultra/bin/python -r requirements-ultralytics.txt

mkdir -p third_party
git clone https://github.com/ultralytics/yolov5 third_party/yolov5
uv pip install --python .venv-v5/bin/python -r requirements-v5.txt
```

## Developer Workflow

Recommended order before long runs:

1. Validate dataset
2. Run dry-run style smoke checks
3. Launch full training
4. Compare test metrics

### Dataset Check

```bash
make dataset-check
```

### Training Commands

```bash
make train-v8
make train-v11
make train-latest
make train-v5
```

Default behavior for training scripts:

- Class `other` (id=1) is ignored during training.
- Original dataset files are not modified.
- A temporary single-class dataset view is generated under `runs/tmp_data/caries_only`.

To keep `other` and train in two-class mode, run script entry points with `--keep-others`:

```bash
python -m scripts.train.train_ultralytics --family v11 --keep-others
python -m scripts.train.train_yolov5 --yolov5-dir third_party/yolov5 --keep-others
python -m scripts.train.run_all --targets v5 v11 --epochs 1 --keep-others
```

To enable the attention mechanism during training, prefer configuring it in YAML or use the training CLI flag:

- Set `use_attention: true` in the top-level of `configs/models.yaml` (or in a per-family section in `configs/models.yaml`) so Slurm wrappers pick it up automatically.
- Or run training directly with `--use-attention`:

```bash
python -m scripts.train.train_ultralytics --family v11 --use-attention
python -m scripts.train.train_yolov5 --yolov5-dir third_party/yolov5 --use-attention
python -m scripts.train.run_all --targets v5 v8 v11 latest --epochs 1 --use-attention
```

Attention behavior by family:

- Ultralytics (v8/v11/latest): inject CBAM attention blocks before training.
- YOLOv5: switch to upstream `yolov5s-transformer.yaml` (C3TR) architecture.

### Smoke Run (all families)

```bash
make run-all
```

### Compare Metrics on Test Split

```bash
make compare-test
```

## Slurm Usage

Standard GPU nodes:

```bash
make slurm-ultra
make slurm-v5
```

P100 nodes with dedicated environments:

```bash
make slurm-ultra-p100
make slurm-v5-p100
```

Direct submission with overrides (Slurm wrappers only control device/venv/family by default):

```bash
FAMILY=v11 sbatch scripts/slurm/train_ultralytics.sbatch
WEIGHTS=yolov5m.pt sbatch scripts/slurm/train_yolov5.sbatch
FAMILY=v8 VENV_DIR=$PWD/.venv-ultra-p100 sbatch scripts/slurm/train_ultralytics_p100.sbatch
WEIGHTS=yolov5s.pt VENV_DIR=$PWD/.venv-v5-p100 sbatch scripts/slurm/train_yolov5_p100.sbatch
```

To enable attention in Slurm jobs, set `use_attention: true` in the top-level or per-family section of `configs/models.yaml`, or run the training module directly with `--use-attention`.

## Config Notes

- Dataset config: [configs/data.caries.yaml](configs/data.caries.yaml)
- Consolidated model defaults: [configs/models.yaml](configs/models.yaml) (preferred; contains per-family subsections `yolov5`, `yolov8`, `yolov11`, `latest`)
- Legacy per-family YAMLs (kept for compatibility): `configs/models.yolov5.yaml`, `configs/models.yolov8.yaml`, `configs/models.yolov11.yaml`, `configs/models.latest.yaml`.

Current class setup:

- nc: 2
- names: caries, other

Training-time note:

- By default, wrappers train as single-class (`caries`) by filtering labels in a generated temporary view.
- Use `--keep-others` to keep `other` in training.

## Artifact Policy

- runs and logs are runtime outputs, not source of truth
- source of truth stays in configs and scripts
- clean artifacts with:

```bash
make clean-artifacts
```

## Important Guardrails

- Do not edit vendor code under third_party/yolov5 unless explicitly required
- Do not start long training automatically after code edits
- Do not delete dataset or checkpoints without explicit confirmation
