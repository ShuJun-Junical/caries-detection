from __future__ import annotations

"""Train YOLOv5 through the vendored upstream train.py entry point."""

import argparse
import subprocess
from pathlib import Path

from scripts.common.io_utils import ROOT, load_yaml, merge_dicts, now_tag

MODELS_CFG_PATH = "configs/models.yaml"
FAMILY_KEYS = {"yolov5", "yolov8", "yolov11", "latest"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv5 via local third_party/yolov5/train.py")
    parser.add_argument("--config", default=MODELS_CFG_PATH, help="Path to the consolidated model YAML")
    parser.add_argument("--yolov5-dir", default="third_party/yolov5")
    parser.add_argument("--device", required=True, help="Training device setting passed to YOLOv5")
    parser.add_argument("--workers", type=int, required=True, help="Dataloader worker count")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    models_blob = load_yaml(args.config)
    if "yolov5" in models_blob:
        model_cfg = models_blob.get("yolov5", {})
        base_cfg = {k: v for k, v in models_blob.items() if k not in FAMILY_KEYS}
    else:
        model_cfg = models_blob
        base_cfg = {k: v for k, v in load_yaml("configs/models.yaml").items() if k not in FAMILY_KEYS}

    cfg = merge_dicts(base_cfg, model_cfg)
    cfg["device"] = args.device
    cfg["workers"] = args.workers

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

    weights = Path(cfg.get("model", "yolov5s.pt"))
    if not weights.is_absolute():
        weights = ROOT / weights
    project = cfg.get("project", "runs/yolov5")
    if not Path(project).is_absolute():
        project = str(ROOT / project)

    name = cfg.get("name") or f"caries-yolov5-{now_tag()}"

    use_attention_flag = bool(cfg.pop("use_attention", False))

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
        str(cfg["device"]),
        "--workers",
        str(cfg["workers"]),
        "--project",
        str(project),
        "--name",
        str(name),
    ]

    if cfg.get("cache", False):
        cmd.append("--cache")

    if use_attention_flag:
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
