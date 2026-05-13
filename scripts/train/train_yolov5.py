from __future__ import annotations

"""Train YOLOv5 through the vendored upstream train.py entry point."""

import argparse
import subprocess
from pathlib import Path

import yaml

from scripts.common.io_utils import ROOT, load_yaml, merge_dicts, now_tag

MODELS_CFG_PATH = "configs/models.yaml"
FAMILY_KEYS = {"v5", "v8", "v11", "v26"}


def _write_yolov5_data_yaml(data_path: Path, output_dir: Path) -> Path:
    data_cfg = load_yaml(data_path)
    dataset_root = Path(data_cfg.get("path", ""))
    if not dataset_root.is_absolute():
        dataset_root = ROOT / dataset_root
    data_cfg["path"] = str(dataset_root.resolve())

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{data_path.stem}.yolov5.yaml"
    with output_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data_cfg, f, sort_keys=False)
    return output_path


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
    base_cfg = {k: v for k, v in models_blob.items() if k not in FAMILY_KEYS}
    model_cfg = models_blob.get("v5")
    if model_cfg is None:
        raise KeyError("Missing family section in model config: v5")

    cfg = merge_dicts(base_cfg, model_cfg)
    cfg["device"] = args.device
    cfg["workers"] = args.workers

    data_value = cfg.get("data")
    if not data_value:
        raise ValueError("Missing training data config. Set data in the model YAML or pass --data.")

    data_path = Path(data_value)
    if not data_path.is_absolute():
        data_path = ROOT / data_path
    cfg["data"] = str(data_path)

    yolov5_dir = Path(args.yolov5_dir)
    if not yolov5_dir.is_absolute():
        yolov5_dir = ROOT / yolov5_dir
    train_py = yolov5_dir / "train.py"
    if not train_py.exists():
        raise FileNotFoundError(
            f"Cannot find YOLOv5 train.py at {train_py}. Clone repo into third_party/yolov5 first."
        )

    weights = Path(cfg.get("model", "yolov5s.pt"))
    if not weights.is_absolute():
        weights = ROOT / weights
    project_path = Path(cfg.get("project", Path("runs") / "v5"))
    if not project_path.is_absolute():
        project_path = ROOT / project_path
    project = str(project_path)
    name = now_tag()
    cfg["data"] = str(_write_yolov5_data_yaml(data_path, project_path))

    use_attention_flag = bool(cfg.pop("use_attention", False))
    attention_model = cfg.pop("attention_model", None)

    cmd = [
        "python",
        str(train_py),
        "--data",
        str(cfg["data"]),
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
        if not attention_model:
            raise ValueError("Missing attention_model for family v5 while use_attention=true")
        attention_cfg = Path(attention_model)
        if not attention_cfg.is_absolute():
            attention_cfg = ROOT / attention_cfg
        if not attention_cfg.exists():
            raise FileNotFoundError(
                f"Attention config not found at {attention_cfg}."
            )
        cmd.extend(["--cfg", str(attention_cfg)])

    print(" ".join(cmd))
    if args.dry_run:
        return 0

    result = subprocess.run(cmd, cwd=str(yolov5_dir), check=False)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
