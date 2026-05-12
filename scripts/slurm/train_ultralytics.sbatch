#!/usr/bin/env bash
#SBATCH --job-name=caries-ultra
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

ROOT_DIR="${SLURM_SUBMIT_DIR:-/home/junical/caries-detection}"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv-ultra}"
CFG="${CFG:-configs/models.yaml}"
FAMILY="${FAMILY:-v26}"
DEVICE="${DEVICE:-auto}"
ULTRALYTICS_DDP_BACKEND="${ULTRALYTICS_DDP_BACKEND:-gloo}"
MIN_CC_MAJOR="${MIN_CC_MAJOR:-7}"
MIN_CC_MINOR="${MIN_CC_MINOR:-0}"
export ULTRALYTICS_DDP_BACKEND
export MIN_CC_MAJOR MIN_CC_MINOR

DEVICE="$(slurm_resolve_device "$DEVICE")"
GPU_COUNT="$(slurm_gpu_count_from_device "$DEVICE")"

if [[ -z "${WORKERS:-}" ]]; then
  WORKERS="$(slurm_default_workers "$GPU_COUNT")"
fi

cd "$ROOT_DIR"
slurm_activate_venv "$VENV_DIR"
CUDA_ERROR_MESSAGE="CUDA is not available in the active environment. Use a CUDA-enabled local venv for GPU training."
export CUDA_ERROR_MESSAGE
slurm_check_cuda

cmd=(python -m scripts.train.train_ultralytics
  --config "$CFG"
  --family "$FAMILY"
  --device "$DEVICE"
  --workers "$WORKERS"
)
srun "${cmd[@]}"
