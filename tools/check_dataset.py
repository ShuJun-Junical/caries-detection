from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.common.dataset_utils import resolve_dataset_root, resolve_yolo_label_dir


def _load_data_cfg(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _image_stem_to_exts(images_dir: Path) -> dict[str, set[str]]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    mapping: dict[str, set[str]] = {}
    for p in images_dir.iterdir():
        if p.is_file() and p.suffix.lower() in exts:
            mapping.setdefault(p.stem, set()).add(p.suffix.lower())
    return mapping


def _label_stems(labels_dir: Path) -> set[str]:
    return {p.stem for p in labels_dir.glob("*.txt") if p.is_file()}


def inspect_split(root: Path, split: str, allowed_classes: set[int]) -> dict:
    images_dir = root / split / "images"
    if not images_dir.exists():
        return {
            "split": split,
            "error": f"Missing directory: {images_dir}",
        }

    try:
        labels_dir = resolve_yolo_label_dir(images_dir)
    except FileNotFoundError as exc:
        return {
            "split": split,
            "error": str(exc),
        }

    image_map = _image_stem_to_exts(images_dir)
    label_stems = _label_stems(labels_dir)

    image_stems = set(image_map.keys())
    missing_labels = sorted(image_stems - label_stems)
    missing_images = sorted(label_stems - image_stems)

    class_counter: Counter[int] = Counter()
    invalid_class_lines: list[dict] = []
    malformed_lines: list[dict] = []
    empty_labels: list[str] = []

    for label_path in sorted(labels_dir.glob("*.txt")):
        text = label_path.read_text(encoding="utf-8").strip()
        if not text:
            empty_labels.append(label_path.name)
            continue

        for line_no, line in enumerate(text.splitlines(), start=1):
            parts = line.split()
            if len(parts) != 5:
                malformed_lines.append(
                    {"file": label_path.name, "line": line_no, "content": line}
                )
                continue

            try:
                cls = int(float(parts[0]))
                _ = [float(x) for x in parts[1:]]
            except ValueError:
                malformed_lines.append(
                    {"file": label_path.name, "line": line_no, "content": line}
                )
                continue

            class_counter[cls] += 1
            if cls not in allowed_classes:
                invalid_class_lines.append(
                    {"file": label_path.name, "line": line_no, "class_id": cls}
                )

    return {
        "split": split,
        "images_count": len(image_stems),
        "labels_count": len(label_stems),
        "missing_labels_count": len(missing_labels),
        "missing_images_count": len(missing_images),
        "missing_labels_samples": missing_labels[:20],
        "missing_images_samples": missing_images[:20],
        "empty_labels_count": len(empty_labels),
        "empty_labels_samples": empty_labels[:20],
        "malformed_lines_count": len(malformed_lines),
        "malformed_lines_samples": malformed_lines[:20],
        "invalid_class_lines_count": len(invalid_class_lines),
        "invalid_class_lines_samples": invalid_class_lines[:20],
        "class_distribution": dict(sorted(class_counter.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Check YOLO dataset integrity")
    parser.add_argument(
        "--data",
        default="configs/data.caries.yaml",
        help="Path to data yaml",
    )
    parser.add_argument(
        "--report",
        default="logs/dataset_check_report.json",
        help="Output JSON report path",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    data_cfg_path = Path(args.data)
    if not data_cfg_path.is_absolute():
        data_cfg_path = repo_root / data_cfg_path

    cfg = _load_data_cfg(data_cfg_path)
    dataset_root = resolve_dataset_root(cfg)

    nc = int(cfg.get("nc", 0))
    allowed_classes = set(range(nc))

    results = {
        "dataset_root": str(dataset_root),
        "data_config": str(data_cfg_path),
        "nc": nc,
        "names": cfg.get("names", []),
        "splits": [],
    }

    for split in ("train", "valid", "test"):
        results["splits"].append(inspect_split(dataset_root, split, allowed_classes))

    report_path = Path(args.report)
    if not report_path.is_absolute():
        report_path = repo_root / report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(results, indent=2, ensure_ascii=True), encoding="utf-8")

    print(json.dumps(results, indent=2, ensure_ascii=True))

    has_blocker = False
    for split in results["splits"]:
        if split.get("error"):
            has_blocker = True
            continue
        if split["malformed_lines_count"] > 0 or split["invalid_class_lines_count"] > 0:
            has_blocker = True

    return 1 if has_blocker else 0


if __name__ == "__main__":
    raise SystemExit(main())
