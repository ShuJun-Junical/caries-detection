#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

sbatch scripts/slurm/train_ultralytics.sbatch
sbatch scripts/slurm/train_yolov5.sbatch
sbatch scripts/slurm/train_ultralytics_p100.sbatch
sbatch scripts/slurm/train_yolov5_p100.sbatch
