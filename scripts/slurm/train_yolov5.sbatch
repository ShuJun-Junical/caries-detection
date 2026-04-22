#!/usr/bin/env bash
#SBATCH --job-name=caries-v5
#SBATCH --partition=dlq
#SBATCH --gres=gpu:1
#SBATCH --exclude=compute[02-03]
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/%j-%x.out
#SBATCH --error=logs/slurm/%j-%x.err
#SBATCH --chdir=/home/junical/caries-detection

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-/home/junical/caries-detection}"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv-v5}"
YOLOV5_DIR="${YOLOV5_DIR:-$ROOT_DIR/third_party/yolov5}"
CFG="${CFG:-configs/models.yaml}"
MIN_CC_MAJOR="${MIN_CC_MAJOR:-7}"
MIN_CC_MINOR="${MIN_CC_MINOR:-0}"
export MIN_CC_MAJOR MIN_CC_MINOR

cd "$ROOT_DIR"
source "$VENV_DIR/bin/activate"

python - <<'PY'
import os
import torch
print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available in the active environment. Use a CUDA-enabled local venv for GPU training.")
major, minor = torch.cuda.get_device_capability(0)
print(f"device_cc={major}.{minor}")
min_cc_major = int(os.environ["MIN_CC_MAJOR"])
min_cc_minor = int(os.environ["MIN_CC_MINOR"])
if (major, minor) < (min_cc_major, min_cc_minor):
  raise SystemExit(f"Allocated GPU has CC {major}.{minor}, which is below the required CC {min_cc_major}.{min_cc_minor} for this environment.")
PY

cmd=(python -m scripts.train.train_yolov5
  --yolov5-dir "$YOLOV5_DIR"
  --cfg "$CFG"
)

srun "${cmd[@]}"
