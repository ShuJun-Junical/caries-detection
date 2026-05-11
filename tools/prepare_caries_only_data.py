from __future__ import annotations

import hashlib
import random
import argparse
from pathlib import Path
from typing import Any
import sys

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.common.dataset_utils import resolve_dataset_root, resolve_yolo_label_dir
from scripts.common.io_utils import ROOT, ensure_dir, load_yaml

dataset_version = "1.0.100"


def _resolve_data_yaml_path(data: str | Path) -> Path:
    p = Path(data)
    if not p.is_absolute():
        p = ROOT / p
    return p.resolve()


def _resolve_dataset_root(cfg: dict[str, Any], data_yaml_path: Path) -> Path:
    return resolve_dataset_root(cfg)


def _class_id_from_line(line: str) -> int | None:
    parts = line.split()
    if len(parts) != 5:
        return None
    try:
        return int(float(parts[0]))
    except ValueError:
        return None


def _parse_mix_spec(value: str) -> tuple[int, int]:
    try:
        parts = value.split(".")
        if len(parts) != 3:
            raise ValueError

        mix_flag = int(parts[0])
        ratio = int(parts[2])
        if mix_flag not in (0, 1) or ratio < 0:
            raise ValueError
    except ValueError as exc:
        raise ValueError(f"dataset_version value is invalid.") from exc

    return mix_flag, ratio


def _sample_image_names(image_names: list[str], sample_size: int, seed: int) -> set[str]:
    if sample_size <= 0 or not image_names:
        return set()
    sample_size = min(sample_size, len(image_names))
    rng = random.Random(seed)
    return set(rng.sample(image_names, sample_size))


def _write_split_view(
    src_images_dir: Path,
    src_labels_dir: Path,
    dst_images_dir: Path,
    dst_labels_dir: Path,
    *,
    keep_class_ids: set[int],
    image_prefix: str = "",
    clear_existing: bool = True,
    selected_image_stems: set[str] | None = None,
) -> None:
    if clear_existing:
        for old_image in dst_images_dir.iterdir():
            if old_image.is_symlink() or old_image.is_file():
                old_image.unlink()

        for old_label in dst_labels_dir.glob("*.txt"):
            old_label.unlink()

    for src_image in src_images_dir.iterdir():
        if not src_image.is_file():
            continue
        if selected_image_stems is not None and src_image.stem not in selected_image_stems:
            continue
        dst_image_name = f"{image_prefix}{src_image.name}"
        (dst_images_dir / dst_image_name).symlink_to(src_image)

    for src_label in src_labels_dir.glob("*.txt"):
        if selected_image_stems is not None and src_label.stem not in selected_image_stems:
            continue
        raw = src_label.read_text(encoding="utf-8")
        lines_out: list[str] = []

        for line in raw.splitlines():
            class_id = _class_id_from_line(line)
            if class_id is None or class_id not in keep_class_ids:
                continue

            parts = line.split()
            parts[0] = "0"
            lines_out.append(" ".join(parts))

        dst_label_name = f"{image_prefix}{src_label.name}"
        dst_label = dst_labels_dir / dst_label_name
        if lines_out:
            dst_label.write_text("\n".join(lines_out) + "\n", encoding="utf-8")
        else:
            dst_label.write_text("", encoding="utf-8")


def build_caries_only_data_yaml(
    data_yaml: str | Path,
    keep_class_ids: set[int] | None = None,
    out_dir: str | Path = "runs/tmp_data/caries_only",
) -> Path:
    """Create a temporary single-class dataset view for training.

    The original dataset is kept unchanged. This function writes filtered label
    files to an output directory and symlinks image directories.
    """

    data_yaml_path = _resolve_data_yaml_path(data_yaml)
    cfg = load_yaml(data_yaml_path)
    dataset_root = _resolve_dataset_root(cfg, data_yaml_path)

    if keep_class_ids is None:
        keep_class_ids = {0}

    mix_flag, ratio = _parse_mix_spec(dataset_version)

    target_root = ensure_dir(out_dir)
    out_yaml = target_root / "data.caries_only.generated.yaml"
    if out_yaml.exists():
        out_yaml.unlink()

    for split in ("train", "val", "test"):
        split_images = Path(cfg[split])
        src_images_dir = (dataset_root / split_images).resolve()
        src_labels_dir = resolve_yolo_label_dir(src_images_dir)

        dst_split_dir = target_root / split
        dst_images_dir = dst_split_dir / "images"
        dst_labels_dir = dst_split_dir / "labels"
        dst_images_dir.mkdir(parents=True, exist_ok=True)
        dst_labels_dir.mkdir(parents=True, exist_ok=True)

        # Keep images under target_root path so YOLOv5 resolves labels from target_root/labels.
        if dst_images_dir.is_symlink():
            dst_images_dir.unlink()
            dst_images_dir.mkdir(parents=True, exist_ok=True)

        _write_split_view(
            src_images_dir,
            src_labels_dir,
            dst_images_dir,
            dst_labels_dir,
            keep_class_ids=keep_class_ids,
        )

        if split == "train" and mix_flag == 1 and ratio > 0:
            test_images = Path(cfg["test"])
            mix_src_images_dir = (dataset_root / test_images).resolve()
            mix_src_labels_dir = resolve_yolo_label_dir(mix_src_images_dir)
            test_image_stems = sorted(
                src_image.stem for src_image in mix_src_images_dir.iterdir() if src_image.is_file()
            )
            digest = hashlib.sha256(
                f"{data_yaml_path.as_posix()}|{str(dataset_version)}".encode("utf-8")
            ).hexdigest()
            seed = int(digest[:16], 16)
            full_copies = ratio // 100
            remainder_pct = ratio % 100

            for copy_index in range(full_copies):
                copy_prefix = f"test{copy_index + 1}__"
                _write_split_view(
                    mix_src_images_dir,
                    mix_src_labels_dir,
                    dst_images_dir,
                    dst_labels_dir,
                    keep_class_ids=keep_class_ids,
                    image_prefix=copy_prefix,
                    clear_existing=False,
                )

            if remainder_pct > 0:
                sample_size = round(len(test_image_stems) * remainder_pct / 100)
                if remainder_pct > 0 and len(test_image_stems) > 0:
                    sample_size = max(1, sample_size)
                sampled_names = _sample_image_names(
                    test_image_stems,
                    sample_size,
                    seed=seed,
                )
                _write_split_view(
                    mix_src_images_dir,
                    mix_src_labels_dir,
                    dst_images_dir,
                    dst_labels_dir,
                    keep_class_ids=keep_class_ids,
                    image_prefix="testp__",
                    clear_existing=False,
                    selected_image_stems=sampled_names,
                )

    out_cfg = {
        "path": str(target_root),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": 1,
        "names": ["caries"],
        "dataset_version": str(dataset_version),
    }
    out_yaml.write_text(yaml.safe_dump(out_cfg, sort_keys=False), encoding="utf-8")
    return out_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build caries-only dataset view before training"
    )
    parser.add_argument("--data", default="configs/data.caries.yaml")
    parser.add_argument("--out-dir", default="runs/tmp_data/caries_only")
    parser.add_argument("--keep-class-id", type=int, default=0)
    parser.add_argument(
        "--merge-class-ids",
        default="",
        help="comma-separated class ids to merge into the single target class, e.g. 0,1",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = ROOT / data_path

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir

    merge_class_ids = {
        int(item)
        for item in args.merge_class_ids.split(",")
        if item.strip() != ""
    }
    if not merge_class_ids:
        merge_class_ids = {args.keep_class_id}

    if args.dry_run:
        print(
            {
                "data": str(data_path),
                "out_dir": str(out_dir),
                "keep_class_ids": sorted(merge_class_ids),
            }
        )
        return 0

    out_yaml = build_caries_only_data_yaml(
        data_yaml=data_path,
        keep_class_ids=merge_class_ids,
        out_dir=out_dir,
    )
    print(str(out_yaml))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
