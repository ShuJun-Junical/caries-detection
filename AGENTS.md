# Repository Conventions For AI Agents

This file defines project-specific rules for AI assistants working in this repository.

## 1) Project Purpose
- Train and evaluate dental caries detection models.
- Supported families: YOLOv5, YOLOv8, YOLOv11, and latest Ultralytics family.
- Dataset format: YOLO detection format.

## 2) Key Layout
- `configs/`: dataset/model/hyperparameter YAML files.
- `scripts/train/`: training entry points.
- `scripts/eval/`: model comparison and evaluation scripts.
- `scripts/slurm/`: Slurm submit scripts.
- `tools/`: dataset and distributed environment checks.
  - `prepare_caries_only_data.py`: build a temporary single-class (caries-only) dataset view; replaces `scripts/common/dataset_filters.py` and `scripts/train/prepare_caries_only_data.py`.
- `third_party/yolov5/`: upstream YOLOv5 codebase (vendor code).
- `runs/` and `logs/`: runtime artifacts (not source of truth).

## 3) Source-Of-Truth Rules
- Do not treat files under `runs/` as stable configuration.
- Prefer `configs/*.yaml` and scripts under `scripts/` as canonical behavior.
- Keep `third_party/yolov5/` untouched unless user explicitly asks to modify vendor code.

## 4) Environment Rules
- Use separate environments for Ultralytics and YOLOv5 to avoid dependency conflicts:
  - `.venv-ultra` for Ultralytics workflows.
  - `.venv-v5` for YOLOv5 workflows.
- Main dependency files:
  - `requirements-ultralytics.txt`
  - `requirements-v5.txt`

## 5) Training Entry Points
- Ultralytics training:
  - `python -m scripts.train.train_ultralytics --family v8|v11|latest --device 0 --workers 8`
- YOLOv5 training:
  - `python -m scripts.train.train_yolov5 --yolov5-dir third_party/yolov5 --device 0 --workers 8`
- Multi-target smoke run:
  - `python -m scripts.train.run_all --targets v5 v8 v11 latest --device 0 --workers 8`

## 6) Evaluation Entry Point
- Compare test metrics across families:
  - `python -m scripts.eval.compare_test_models --models v5 v8 v11 latest`

## 7) Slurm Rules
Training controls such as `use_attention`, `epochs`, `imgsz`, `batch`, `project`, and `name` should be set in the top-level of `configs/models.yaml` or the per-family section of `configs/models.yaml`. `device` and `workers` are supplied by the training CLI or Slurm wrapper.

### Parameter Control Matrix (Single Source of Truth)
- Config-only (do not expose as training CLI overrides):
  - `epochs`, `imgsz`, `batch`, `patience`, `optimizer`, `lr0`, `weight_decay`, `cache`, `use_attention`
  - per-family: `model`, `data`, `project`, `name`, `amp`
- CLI/Slurm runtime-only:
  - `device`, `workers`
  - `family` (Ultralytics selector), `yolov5-dir` (YOLOv5 repo path), `config` (`CFG` in Slurm)
- Enforcement rule:
  - Do not add the same parameter to both config and training CLI. If a new parameter is added, choose one control plane only.

## 8) Data Rules
- Dataset config: `dataset/data.caries.yaml`.
- Required split structure:
  - `dataset/train/images`, `dataset/train/labels`
  - `dataset/val/images`, `dataset/val/labels`
  - `dataset/test/images`, `dataset/test/labels`
- Validate data integrity before major training runs:
  - `python tools/check_dataset.py --data dataset/data.caries.yaml`

## 9) Editing Rules For Agents
- Keep changes minimal and scoped to the user request.
- Do not rename files/directories or refactor broad structure without explicit user confirmation.
- If requirement is ambiguous, ask the user before destructive operations.
- Preserve existing style and CLI interface in scripts unless user asks otherwise.

## 10) Artifact Hygiene
- `runs/` and `logs/` can be cleaned when user asks.
- Keep only explicitly requested model weights when pruning artifacts.
- Do not delete checkpoints in `checkpoints/` unless user explicitly requests it.

## 11) Strict Guardrails
- Never modify `third_party/yolov5/` unless the user explicitly asks.
- Never launch long training jobs automatically after code edits.
- Never delete dataset files or checkpoint files without explicit confirmation.
- For cleanup tasks, print the post-cleanup file list for verification.
- Keep outputs reproducible: prefer config-driven arguments over hardcoded values.

## 12) Recommended Execution Workflow
- Before Slurm submission, run a lightweight local validation:
  - `python tools/check_dataset.py --data dataset/data.caries.yaml`
- For script changes, run `--dry-run` when available.
- For Slurm, submit with explicit overrides when testing:
  - `FAMILY=v11 DEVICE=0 WORKERS=8 sbatch scripts/slurm/train_ultralytics.sbatch`
- Store final behavior in `configs/*.yaml`; keep ad-hoc experiments out of source files.
