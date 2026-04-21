from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

from scripts.common.io_utils import ROOT, ensure_dir


@dataclass(frozen=True)
class ModelSpec:
    label: str
    kind: str
    weights: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate YOLOv5/v8/v11/latest models on the test split and compare metrics"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["v5", "v8", "v11", "latest"],
        choices=["v5", "v8", "v11", "latest"],
        help="Model families to evaluate",
    )
    parser.add_argument("--data", default="configs/data.caries.yaml")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--yolov5-dir", default="third_party/yolov5")
    parser.add_argument("--output-dir", default="runs/test_compare")
    parser.add_argument("--v5-weights", default="runs/yolov5/weights/best.pt")
    parser.add_argument("--v8-weights", default="runs/v8/weights/best.pt")
    parser.add_argument("--v11-weights", default="runs/v11/weights/best.pt")
    parser.add_argument("--latest-weights", default="runs/latest/weights/best.pt")
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = ROOT / resolved
    return resolved


def to_float(value: Any) -> float:
    array = np.asarray(value)
    if array.ndim == 0:
        return float(array.item())
    return float(array.mean())


def load_yolov5_runner(yolov5_dir: Path):
    if str(yolov5_dir) not in sys.path:
        sys.path.insert(0, str(yolov5_dir))
    from val import run as yolov5_run  # type: ignore

    return yolov5_run


def evaluate_ultralytics(
    weights: Path,
    data: Path,
    split: str,
    imgsz: int,
    batch: int,
    device: str,
    project: Path,
    name: str,
) -> dict[str, float]:
    model = YOLO(str(weights))
    metrics = model.val(
        data=str(data),
        split=split,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(project),
        name=name,
        exist_ok=True,
        plots=False,
        verbose=False,
    )
    return {
        "precision": to_float(metrics.box.mp if hasattr(metrics.box, "mp") else metrics.box.p),
        "recall": to_float(metrics.box.mr if hasattr(metrics.box, "mr") else metrics.box.r),
        "mAP50": to_float(metrics.box.map50),
        "mAP50-95": to_float(metrics.box.map),
    }


def evaluate_yolov5(
    yolov5_run,
    weights: Path,
    data: Path,
    split: str,
    imgsz: int,
    batch: int,
    device: str,
    project: Path,
    name: str,
) -> dict[str, float]:
    results, _, _ = yolov5_run(
        data=str(data),
        weights=str(weights),
        batch_size=batch,
        imgsz=imgsz,
        device=device,
        task=split,
        project=str(project),
        name=name,
        exist_ok=True,
        plots=False,
        verbose=False,
    )
    precision, recall, map50, map_5095 = results[:4]
    return {
        "precision": float(precision),
        "recall": float(recall),
        "mAP50": float(map50),
        "mAP50-95": float(map_5095),
    }


def write_csv(rows: list[dict[str, Any]], csv_path: Path) -> None:
    fieldnames = ["model", "kind", "weights", "precision", "recall", "mAP50", "mAP50-95"]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_metrics(rows: list[dict[str, Any]], fig_path: Path, split: str) -> None:
    models = [row["model"] for row in rows]
    metrics = ["precision", "recall", "mAP50", "mAP50-95"]
    colors = ["#0f766e", "#2563eb", "#ca8a04", "#dc2626"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    fig.suptitle(f"Model comparison on {split} split", fontsize=14, fontweight="bold")

    for ax, metric in zip(axes.flat, metrics):
        values = [float(row[metric]) for row in rows]
        bars = ax.bar(models, values, color=colors[: len(models)], width=0.65)
        ax.set_title(metric)
        ax.set_ylim(0.0, 1.0)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", labelrotation=15)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.015,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.savefig(fig_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()

    data = resolve_path(args.data)
    output_dir = ensure_dir(args.output_dir)

    specs: list[ModelSpec] = []
    if "v5" in args.models:
        specs.append(ModelSpec("v5", "yolov5", resolve_path(args.v5_weights)))
    if "v8" in args.models:
        specs.append(ModelSpec("v8", "ultralytics", resolve_path(args.v8_weights)))
    if "v11" in args.models:
        specs.append(ModelSpec("v11", "ultralytics", resolve_path(args.v11_weights)))
    if "latest" in args.models:
        specs.append(ModelSpec("latest", "ultralytics", resolve_path(args.latest_weights)))

    yolov5_run = None
    if any(spec.kind == "yolov5" for spec in specs):
        yolov5_dir = resolve_path(args.yolov5_dir)
        yolov5_run = load_yolov5_runner(yolov5_dir)

    rows: list[dict[str, Any]] = []
    for spec in specs:
        if not spec.weights.exists():
            raise FileNotFoundError(
                f"Missing weights for {spec.label}: {spec.weights}. Pass an explicit path with --{spec.label}-weights."
            )

        model_dir = output_dir / spec.label
        model_dir.mkdir(parents=True, exist_ok=True)

        if spec.kind == "yolov5":
            if yolov5_run is None:
                raise RuntimeError("YOLOv5 runner was not loaded.")
            metrics = evaluate_yolov5(
                yolov5_run,
                spec.weights,
                data,
                args.split,
                args.imgsz,
                args.batch,
                args.device,
                model_dir,
                args.split,
            )
        else:
            metrics = evaluate_ultralytics(
                spec.weights,
                data,
                args.split,
                args.imgsz,
                args.batch,
                args.device,
                model_dir,
                args.split,
            )

        row: dict[str, Any] = {
            "model": spec.label,
            "kind": spec.kind,
            "weights": str(spec.weights),
        }
        row.update(metrics)
        rows.append(row)
        print(
            f"{spec.label}: precision={metrics['precision']:.4f} recall={metrics['recall']:.4f} "
            f"mAP50={metrics['mAP50']:.4f} mAP50-95={metrics['mAP50-95']:.4f}"
        )

    csv_path = output_dir / f"{args.split}_metrics.csv"
    fig_path = output_dir / f"{args.split}_comparison.png"
    write_csv(rows, csv_path)
    plot_metrics(rows, fig_path, args.split)

    print(f"Saved metrics to {csv_path}")
    print(f"Saved chart to {fig_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())