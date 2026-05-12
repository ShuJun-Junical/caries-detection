#!/usr/bin/env bash
#SBATCH --job-name=caries-v5-p100
#SBATCH --partition=hpcq
#SBATCH --gres=gpu:1
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
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv-v5-p100}"
YOLOV5_DIR="${YOLOV5_DIR:-$ROOT_DIR/third_party/yolov5}"
CFG="${CFG:-configs/models.yaml}"
DEVICE="${DEVICE:-0}"
MIN_CC_MAJOR="${MIN_CC_MAJOR:-6}"
MIN_CC_MINOR="${MIN_CC_MINOR:-0}"
export MIN_CC_MAJOR MIN_CC_MINOR

DEVICE="$(slurm_resolve_device "$DEVICE")"
GPU_COUNT="$(slurm_gpu_count_from_device "$DEVICE")"
if [[ -z "${WORKERS:-}" ]]; then
    WORKERS="$(slurm_default_workers "$GPU_COUNT")"
fi

cd "$ROOT_DIR"
slurm_activate_venv "$VENV_DIR"
CUDA_ERROR_MESSAGE="CUDA is not available in the active environment. Use a CUDA-enabled P100-compatible venv for GPU training."
export CUDA_ERROR_MESSAGE
slurm_check_cuda

cmd=(python -m scripts.train.train_yolov5
    --yolov5-dir "$YOLOV5_DIR"
    --config "$CFG"
    --device "$DEVICE"
    --workers "$WORKERS"
)

srun "${cmd[@]}"
