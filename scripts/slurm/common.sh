#!/usr/bin/env bash

slurm_activate_venv() {
  local venv_dir="$1"
  source "$venv_dir/bin/activate"
}

slurm_resolve_device() {
  local device="${1:-auto}"
  local gpu_count=""

  if [[ "$device" == "auto" ]]; then
    gpu_count="${SLURM_GPUS_ON_NODE:-}"
    if [[ -z "$gpu_count" ]] && command -v nvidia-smi >/dev/null 2>&1; then
      gpu_count=$(nvidia-smi -L | wc -l)
    fi

    if [[ -n "$gpu_count" ]] && [[ "$gpu_count" -gt 1 ]]; then
      seq -s, 0 $((gpu_count - 1))
    else
      printf '0\n'
    fi
    return
  fi

  printf '%s\n' "$device"
}

slurm_gpu_count_from_device() {
  local device="$1"
  if [[ "$device" == *,* ]]; then
    awk -F',' '{print NF}' <<<"$device"
  else
    printf '1\n'
  fi
}

slurm_default_workers() {
  local gpu_count="${1:-1}"
  local cpus="${SLURM_CPUS_PER_TASK:-1}"
  local workers=$((cpus / gpu_count))
  if [[ "$workers" -lt 1 ]]; then
    workers=1
  fi
  printf '%s\n' "$workers"
}

slurm_check_cuda() {
  python - <<'PY'
import os
import torch

print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit(os.environ.get("CUDA_ERROR_MESSAGE", "CUDA is not available in the active environment."))

major, minor = torch.cuda.get_device_capability(0)
print(f"device_cc={major}.{minor}")

min_cc_major = os.environ.get("MIN_CC_MAJOR")
min_cc_minor = os.environ.get("MIN_CC_MINOR")
if min_cc_major is not None and min_cc_minor is not None:
    required = (int(min_cc_major), int(min_cc_minor))
    if (major, minor) < required:
        raise SystemExit(
            f"Allocated GPU has CC {major}.{minor}, which is below the required CC "
            f"{required[0]}.{required[1]} for this environment."
        )
PY
}
