# Caries Detection Training

Training and evaluation workflows for dental caries detection across v5, v8, v11, and v26 model families.

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
- tools: dataset and distributed environment checks (see `tools/prepare_caries_only_data.py` for building a caries-only dataset view)
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

The raw dataset uses the standard YOLO layout with `images/` and `labels/` under each split. In this repository, the `dataset/{train,valid,test}/labels` entries are symlinks to the existing `yolo/` folders, so scripts and trainers can read labels from `labels/` directly.

### Build caries-only dataset view

Before training, you can create the single-class dataset view with:

```bash
python tools/prepare_caries_only_data.py --data dataset/data.caries.yaml --out-dir dataset/caries_only
```

### Training Commands

```bash
make train-v8
make train-v11
make train-v26
make train-v5
```

Default behavior for training scripts:

- Dataset selection is family-specific in `configs/models.yaml`.
- `v5`, `v8`, `v11`, and `v26` train on `configs/data.caries.yaml`.
- Original dataset files are not modified.

Training hyperparameters now live only in `configs/models.yaml`. The direct training entry points only take config plus device settings, so the same parameter does not exist in both YAML and CLI.

### Parameter Control Matrix

| Parameter | Where to control | Notes |
| --- | --- | --- |
| epochs | configs/models.yaml | Training hyperparameter; no CLI override in training entry scripts. |
| imgsz | configs/models.yaml | Training hyperparameter; used by both Ultralytics and YOLOv5 wrappers. |
| batch | configs/models.yaml | Training hyperparameter; no Slurm direct override by default. |
| patience | configs/models.yaml | Ultralytics-side training behavior. |
| optimizer | configs/models.yaml | Ultralytics-side training behavior. |
| lr0 | configs/models.yaml | Learning-rate base value from config only. |
| weight_decay | configs/models.yaml | Optimization hyperparameter from config only. |
| cache | configs/models.yaml | Passed to train pipeline from config only. |
| use_attention | configs/models.yaml | Enables CBAM injection (Ultralytics) or transformer cfg (YOLOv5). |
| model | configs/models.yaml family section | Family checkpoint path. |
| data | configs/models.yaml family section | Family dataset YAML path. |
| project | configs/models.yaml family section | Output root for that family. |
| name | configs/models.yaml family section | Run name for that family. |
| amp | configs/models.yaml family section | Family-specific Ultralytics option. |
| device | CLI / Slurm env | Training script requires `--device`; Slurm sets `DEVICE`. |
| workers | CLI / Slurm env | Training script requires `--workers`; Slurm sets or auto-computes `WORKERS`. |
| family | CLI / Slurm env (Ultralytics only) | `train_ultralytics.py` requires `--family`; Slurm sets `FAMILY`. |
| config | CLI / Slurm env | `--config` selects model YAML; Slurm sets `CFG`. |
| yolov5_dir | CLI / Slurm env (YOLOv5 only) | `--yolov5-dir`; Slurm sets `YOLOV5_DIR`. |

Rule of thumb: if it changes model training behavior, put it in `configs/models.yaml`; if it is cluster/device runtime wiring, pass it via CLI/Slurm.

```bash
python -m scripts.train.train_ultralytics --family v11 --device 0 --workers 8
python -m scripts.train.train_yolov5 --device 0 --workers 8
python -m scripts.train.run_all --targets v5 v8 v11 v26 --device 0 --workers 8
```

Set `use_attention: true` in `configs/models.yaml` if you want attention enabled. For two-class training, point the family entry at `dataset/caries_only/data.caries_only.generated.yaml` after regenerating that dataset view.

Attention behavior by family:

- Ultralytics (v8/v11/v26; v26 uses official YOLO26 weights): inject CBAM attention blocks before training.
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

Direct submission with overrides (Slurm wrappers only control device/venv/family/workers by default):

```bash
FAMILY=v11 sbatch scripts/slurm/train_ultralytics.sbatch
DEVICE=0 WORKERS=8 sbatch scripts/slurm/train_yolov5.sbatch
FAMILY=v8 VENV_DIR=$PWD/.venv-ultra-p100 sbatch scripts/slurm/train_ultralytics_p100.sbatch
DEVICE=0 WORKERS=8 VENV_DIR=$PWD/.venv-v5-p100 sbatch scripts/slurm/train_yolov5_p100.sbatch
```

To enable attention in Slurm jobs, set `use_attention: true` in the top-level or per-family section of `configs/models.yaml`.

## Config Notes

- Dataset config: [dataset/data.caries.yaml](dataset/data.caries.yaml)
- Consolidated model defaults: [configs/models.yaml](configs/models.yaml) (preferred; contains per-family subsections `v5`, `v8`, `v11`, `v26`)
- Dataset split layout: `dataset/train|valid|test/images` and `dataset/train|valid|test/labels`, where each `labels/` symlink points to the existing `yolo/` directory.

Current class setup:

- nc: 2
- names: caries, other

Training-time note:

- To train a family on the generated single-class view, point its `data` entry in `configs/models.yaml` at `dataset/caries_only/data.caries_only.generated.yaml`.
- To train on the full two-class dataset, point the family entry at `dataset/data.caries.yaml`.

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
