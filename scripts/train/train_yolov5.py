from __future__ import annotations

"""Train YOLOv5 through the vendored upstream train.py entry point."""

import argparse
import subprocess
from pathlib import Path

from scripts.common.io_utils import ROOT, load_yaml, merge_dicts, now_tag
import os


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv5 via local third_party/yolov5/train.py")
    parser.add_argument("--yolov5-dir", default="third_party/yolov5")
    parser.add_argument("--data", default=None)
    parser.add_argument("--cfg", default="configs/models.yaml")
    parser.add_argument("--hyp", default=None, help="Optional hyperparameter YAML (defaults to top-level of configs/models.yaml)")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--img", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--project", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--cache", action="store_true")
    parser.add_argument(
        "--use-attention",
        action="store_true",
        help="Use YOLOv5 transformer model config (C3TR) to enable attention layers.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Load model/config file which may be the consolidated `configs/models.yaml`
    models_blob = load_yaml(args.cfg)
    FAMILY_KEYS = {"yolov5", "yolov8", "yolov11", "latest"}
    if "yolov5" in models_blob:
        # consolidated file: top-level keys are shared defaults + per-family sections
        model_cfg = models_blob.get("yolov5", {})
        base_cfg = load_yaml(args.hyp) if args.hyp else {k: v for k, v in models_blob.items() if k not in FAMILY_KEYS}
    else:
        # legacy per-model YAML
        model_cfg = models_blob
        if args.hyp:
            base_cfg = load_yaml(args.hyp)
        else:
            # prefer top-level defaults from consolidated models.yaml when available
            all_models = load_yaml("configs/models.yaml")
            FAMILY_KEYS = {"yolov5", "yolov8", "yolov11", "latest"}
            base_cfg = {k: v for k, v in all_models.items() if k not in FAMILY_KEYS}

    cli_cfg = {
        "model": args.weights,
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.img,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "project": args.project,
        "name": args.name,
        "cache": True if args.cache else None,
    }
    cfg = merge_dicts(base_cfg, model_cfg, cli_cfg)

    # Resolve workers with priority: CLI > env WORKERS > auto-calc from SLURM_CPUS_PER_TASK/GPUs > hyp/base default
    if cfg.get("workers") is None:
        env_workers = os.getenv("WORKERS")
        if env_workers:
            try:
                cfg["workers"] = int(env_workers)
            except ValueError:
                pass
        else:
            device_val = cfg.get("device")
            num_gpus = None
            if isinstance(device_val, str):
                if device_val == "auto":
                    slurm_gpus = os.getenv("SLURM_GPUS_ON_NODE")
                    if slurm_gpus and slurm_gpus.isdigit():
                        num_gpus = int(slurm_gpus)
                elif "," in device_val:
                    num_gpus = device_val.count(",") + 1
                elif device_val.isdigit():
                    num_gpus = 1
            if not num_gpus or num_gpus < 1:
                try:
                    import torch

                    num_gpus = torch.cuda.device_count()
                except Exception:
                    num_gpus = None
            if not num_gpus or num_gpus < 1:
                num_gpus = 1

            try:
                total_cpus = int(os.getenv("SLURM_CPUS_PER_TASK") or 0)
            except Exception:
                total_cpus = 0
            if not total_cpus:
                total_cpus = os.cpu_count() or 1

            cfg["workers"] = max(1, total_cpus // max(1, int(num_gpus)))

    data_value = cfg.get("data")
    if not data_value:
        raise ValueError("Missing training data config. Set data in the model YAML or pass --data.")

    yolov5_dir = Path(args.yolov5_dir)
    if not yolov5_dir.is_absolute():
        yolov5_dir = ROOT / yolov5_dir
    train_py = yolov5_dir / "train.py"
    if not train_py.exists():
        raise FileNotFoundError(
            f"Cannot find YOLOv5 train.py at {train_py}. Clone repo into third_party/yolov5 first."
        )

    data_path = Path(cfg["data"])
    if not data_path.is_absolute():
        data_path = ROOT / data_path

    weights = cfg.get("model", "yolov5s.pt")
    project = cfg.get("project", "runs/yolov5")
    if not Path(project).is_absolute():
        project = str(ROOT / project)

    name = cfg.get("name") or f"caries-yolov5-{now_tag()}"

    cmd = [
        "python",
        str(train_py),
        "--data",
        str(data_path),
        "--weights",
        str(weights),
        "--epochs",
        str(cfg.get("epochs", 100)),
        "--img",
        str(cfg.get("imgsz", 640)),
        "--batch",
        str(cfg.get("batch", 16)),
        "--device",
        str(cfg.get("device", "0")),
        "--workers",
        str(cfg.get("workers", 8)),
        "--project",
        str(project),
        "--name",
        str(name),
    ]

    if cfg.get("cache", False):
        cmd.append("--cache")

    if args.use_attention:
        transformer_cfg = yolov5_dir / "models" / "hub" / "yolov5s-transformer.yaml"
        if not transformer_cfg.exists():
            raise FileNotFoundError(
                f"Attention config not found at {transformer_cfg}. Ensure YOLOv5 repo is complete."
            )
        cmd.extend(["--cfg", str(transformer_cfg)])

    print(" ".join(cmd))
    if args.dry_run:
        return 0

    result = subprocess.run(cmd, cwd=str(yolov5_dir), check=False)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
