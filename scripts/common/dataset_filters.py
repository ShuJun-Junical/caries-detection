from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Any

import yaml

from scripts.common.io_utils import ROOT, ensure_dir, load_yaml

dataset_version = "1.0.100"

def _resolve_data_yaml_path(data: str | Path) -> Path:
    p = Path(data)
    if not p.is_absolute():
        p = ROOT / p
    return p.resolve()


def _resolve_dataset_root(cfg: dict[str, Any], data_yaml_path: Path) -> Path:
    dataset_root = Path(cfg["path"])
    if dataset_root.is_absolute():
        return dataset_root.resolve()
    return (data_yaml_path.parent / dataset_root).resolve()


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
    keep_class_id: int,
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
            if class_id is None or class_id != keep_class_id:
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
    keep_class_id: int = 0,
    out_dir: str | Path = "runs/tmp_data/caries_only",
) -> Path:
    """Create a temporary single-class dataset view for training.

    The original dataset is kept unchanged. This function writes filtered label
    files to an output directory and symlinks image directories.
    """

    data_yaml_path = _resolve_data_yaml_path(data_yaml)
    cfg = load_yaml(data_yaml_path)
    dataset_root = _resolve_dataset_root(cfg, data_yaml_path)

    mix_flag, ratio = _parse_mix_spec(dataset_version)

    target_root = ensure_dir(out_dir)
    out_yaml = target_root / "data.caries_only.generated.yaml"
    if out_yaml.exists():
        out_yaml.unlink()

    for split in ("train", "val", "test"):
        split_images = Path(cfg[split])
        src_images_dir = (dataset_root / split_images).resolve()
        src_labels_dir = src_images_dir.parent / "labels"

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
            keep_class_id=keep_class_id,
        )

        if split == "train" and mix_flag == 1 and ratio > 0:
            test_images = Path(cfg["test"])
            mix_src_images_dir = (dataset_root / test_images).resolve()
            mix_src_labels_dir = mix_src_images_dir.parent / "labels"
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
                    keep_class_id=keep_class_id,
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
                    keep_class_id=keep_class_id,
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
