from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

from scripts.common.io_utils import ROOT, load_yaml, merge_dicts, now_tag

FAMILY_TO_CFG = {
    "v8": "configs/models.yolov8.yaml",
    "v11": "configs/models.yolov11.yaml",
    "latest": "configs/models.latest.yaml",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO (v8/v11/latest) with Ultralytics")
    parser.add_argument("--family", choices=["v8", "v11", "latest"], required=True)
    parser.add_argument("--model", default=None, help="Override model checkpoint, e.g. yolo11n.pt")
    parser.add_argument("--data", default="configs/data.caries.yaml")
    parser.add_argument("--hyp", default=None, help="Optional legacy hyperparameter YAML override")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--project", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    model_cfg = load_yaml(FAMILY_TO_CFG[args.family])
    base_cfg = load_yaml(args.hyp) if args.hyp else {}

    cli_cfg = {
        "model": args.model,
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "project": args.project,
        "name": args.name,
    }

    cfg = merge_dicts(base_cfg, model_cfg, cli_cfg)

    data_path = Path(cfg["data"])
    if not data_path.is_absolute():
        data_path = ROOT / data_path
    cfg["data"] = str(data_path)

    project = cfg.get("project", f"runs/{args.family}")
    if not Path(project).is_absolute():
        project = str(ROOT / project)
    cfg["project"] = project

    run_name = cfg.get("name") or f"caries-{args.family}-{now_tag()}"
    cfg["name"] = run_name

    if args.dry_run:
        print(cfg)
        return 0

    model_path = cfg.pop("model")
    model = YOLO(model_path)
    model.train(**cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
