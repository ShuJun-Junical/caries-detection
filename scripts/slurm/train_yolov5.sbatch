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
DEVICE="${DEVICE:-auto}"
MIN_CC_MAJOR="${MIN_CC_MAJOR:-7}"
MIN_CC_MINOR="${MIN_CC_MINOR:-0}"
export MIN_CC_MAJOR MIN_CC_MINOR

cd "$ROOT_DIR"
source "$VENV_DIR/bin/activate"

if [[ -z "${WORKERS:-}" ]]; then
  WORKERS=$SLURM_CPUS_PER_TASK
fi

if [[ "$DEVICE" == "auto" ]]; then
  gpu_count="${SLURM_GPUS_ON_NODE:-}"
  if [[ -z "$gpu_count" ]] && command -v nvidia-smi >/dev/null 2>&1; then
    gpu_count=$(nvidia-smi -L | wc -l)
  fi
  if [[ -n "$gpu_count" ]] && [[ "$gpu_count" -gt 1 ]]; then
    DEVICE=$(seq -s, 0 $((gpu_count - 1)))
  else
    DEVICE=0
  fi
fi

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
  --config "$CFG"
  --device "$DEVICE"
  --workers "$WORKERS"
)

srun "${cmd[@]}"
