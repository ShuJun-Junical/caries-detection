#!/usr/bin/env bash
#SBATCH --job-name=caries-eval
#SBATCH --partition=dlq
#SBATCH --gres=gpu:1
#SBATCH --exclude=compute[02-03]
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/%j-%x.out
#SBATCH --error=logs/slurm/%j-%x.err
#SBATCH --chdir=/home/junical/caries-detection

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-/home/junical/caries-detection}"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv-ultra}"

# Supported values: v8 | v11 | latest
MODEL="${MODEL:-v8}"
DEVICE="${DEVICE:-0}"
SPLIT="${SPLIT:-test}"
IMGSZ="${IMGSZ:-640}"
BATCH="${BATCH:-16}"
DATA="${DATA:-configs/data.caries.yaml}"
OUT_ROOT="${OUT_ROOT:-runs/test_compare}"

# Use explicit best.pt paths to avoid picking stale runs.
V8_WEIGHTS="${V8_WEIGHTS:-runs/v8/caries-v8-20260421-042206/weights/best.pt}"
V11_WEIGHTS="${V11_WEIGHTS:-runs/v11/caries-v11-20260421-042206/weights/best.pt}"
LATEST_WEIGHTS="${LATEST_WEIGHTS:-runs/latest/caries-latest-20260421-042209/weights/best.pt}"

cd "$ROOT_DIR"
source "$VENV_DIR/bin/activate"

python - <<'PY'
import torch
print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available in the active environment.")
major, minor = torch.cuda.get_device_capability(0)
print(f"device_cc={major}.{minor}")
PY

cmd=(python -m scripts.eval.compare_test_models
  --models "$MODEL"
  --data "$DATA"
  --split "$SPLIT"
  --imgsz "$IMGSZ"
  --batch "$BATCH"
  --device "$DEVICE"
  --output-dir "$OUT_ROOT/$MODEL"
  --v8-weights "$V8_WEIGHTS"
  --v11-weights "$V11_WEIGHTS"
  --latest-weights "$LATEST_WEIGHTS"
)

echo "Running: ${cmd[*]}"
srun "${cmd[@]}"
