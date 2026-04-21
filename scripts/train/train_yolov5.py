from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from scripts.common.io_utils import ROOT, load_yaml, merge_dicts, now_tag


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv5 via local yolov5/train.py")
    parser.add_argument("--yolov5-dir", default="third_party/yolov5")
    parser.add_argument("--data", default="configs/data.caries.yaml")
    parser.add_argument("--cfg", default="configs/models.yolov5.yaml")
    parser.add_argument("--hyp", default="configs/hyp.base.yaml")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--img", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--project", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    base_cfg = load_yaml(args.hyp)
    model_cfg = load_yaml(args.cfg)

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

    yolov5_dir = Path(args.yolov5_dir)
    if not yolov5_dir.is_absolute():
        yolov5_dir = ROOT / yolov5_dir
    train_py = yolov5_dir / "train.py"
    if not train_py.exists():
        raise FileNotFoundError(
            f"Cannot find YOLOv5 train.py at {train_py}. Clone repo into third_party/yolov5 first."
        )

    data_path = Path(cfg.get("data", args.data))
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

    print(" ".join(cmd))
    if args.dry_run:
        return 0

    result = subprocess.run(cmd, cwd=str(yolov5_dir), check=False)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
