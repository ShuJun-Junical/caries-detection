# Caries Detection Training (YOLOv5, YOLOv8, YOLOv11, Latest)

This project provides training scripts for caries detection on the local YOLO-format dataset in `dataset/`.

## Project Structure

- `configs/`: dataset and model config files
- `scripts/common/`: shared utilities
- `scripts/train/`: training entry scripts
- `tools/`: dataset validation utilities
- `runs/`: training outputs
- `logs/`: reports and logs
- `checkpoints/`: optional manual checkpoint storage

## Prerequisites

- Linux
- Python 3.10+
- `uv` installed

## Environment Setup (uv)

Create two isolated environments to avoid dependency conflicts:

```bash
cd /home/junical/caries-detection
uv venv .venv-ultra
uv venv .venv-v5

uv pip install --python .venv-ultra/bin/python -r requirements-ultralytics.txt
```

For YOLOv5, clone repo first:

```bash
mkdir -p third_party
git clone https://github.com/ultralytics/yolov5 third_party/yolov5
uv pip install --python .venv-v5/bin/python -r requirements-v5.txt
```

## Data Config

Dataset config is in `configs/data.caries.yaml`:

- `nc: 2`
- `names: [caries, other]`

## 1) Check Dataset Integrity

```bash
uv run --python .venv-ultra/bin/python python tools/check_dataset.py --data configs/data.caries.yaml
```

A JSON report will be generated at `logs/dataset_check_report.json`.

## 2) Train YOLOv8

```bash
uv run --python .venv-ultra/bin/python python -m scripts.train.train_ultralytics --family v8
```

## 3) Train YOLOv11

```bash
uv run --python .venv-ultra/bin/python python -m scripts.train.train_ultralytics --family v11
```

## 4) Train Latest Stable (Ultralytics)

`latest` uses `configs/models.latest.yaml`. Update `model:` there when a new family is released.

```bash
uv run --python .venv-ultra/bin/python python -m scripts.train.train_ultralytics --family latest
```

## 5) Train YOLOv5

```bash
uv run --python .venv-v5/bin/python python -m scripts.train.train_yolov5 --yolov5-dir third_party/yolov5 --weights yolov5s.pt --epochs 100
```

## 6) Run Multiple Targets

Smoke test all targets with 1 epoch:

```bash
python -m scripts.train.run_all --targets v5 v8 v11 latest --epochs 1
```

## 7) Run on Slurm GPU Nodes

The cluster provides GPU partitions such as `dlq` and `hpcq`.
Current local venvs use CUDA builds (`torch 2.11.0+cu128`) and are validated on RTX3090 nodes.
The provided sbatch scripts exclude TITAN X nodes (`compute02-03`) because they are not compatible with this torch build.

P100 nodes need a separate CUDA-compatible environment built for `sm_60`. Use the dedicated P100 sbatch wrappers below; do not reuse `.venv-ultra` or `.venv-v5` on P100.

Submit the Ultralytics job:

```bash
sbatch scripts/slurm/train_ultralytics.sbatch
```

Submit the P100-compatible Ultralytics job:

```bash
sbatch scripts/slurm/train_ultralytics_p100.sbatch
```

Submit the YOLOv5 job:

```bash
sbatch scripts/slurm/train_yolov5.sbatch
```

Submit the P100-compatible YOLOv5 job:

```bash
sbatch scripts/slurm/train_yolov5_p100.sbatch
```

Override defaults with environment variables at submission time:

```bash
FAMILY=v11 EPOCHS=100 sbatch scripts/slurm/train_ultralytics.sbatch
WEIGHTS=yolov5m.pt CFG=configs/models.yolov5.yaml HYP=configs/hyp.base.yaml EPOCHS=100 sbatch scripts/slurm/train_yolov5.sbatch
FAMILY=v8 VENV_DIR=$PWD/.venv-ultra-p100 sbatch scripts/slurm/train_ultralytics_p100.sbatch
WEIGHTS=yolov5s.pt VENV_DIR=$PWD/.venv-v5-p100 sbatch scripts/slurm/train_yolov5_p100.sbatch
```

The Slurm scripts check both `torch.cuda.is_available()` and device compute capability, and fail early on incompatible GPUs. The P100 wrappers lower the minimum compute capability to `6.0` and assume a P100-compatible torch build.

## Notes

- YOLOv5 script requires local `third_party/yolov5/train.py`.
- Training outputs are written under `runs/` and ignored by git.
- If CUDA is unavailable, set `--device cpu` in training commands.
