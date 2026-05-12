from __future__ import annotations

import hashlib
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

import yaml

from scripts.common.io_utils import ROOT, ensure_dir, load_yaml


def _resolve_path(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = ROOT / resolved
    return resolved


def resolve_dataset_root(cfg: dict[str, Any]) -> Path:
    dataset_root = Path(cfg["path"])
    if dataset_root.is_absolute():
        return dataset_root.resolve()
    return (ROOT / dataset_root).resolve()


def resolve_yolo_label_dir(images_dir: Path) -> Path:
    labels_dir = images_dir.parent / "yolo"
    if labels_dir.exists():
        return labels_dir
    raise FileNotFoundError(f"Missing yolo directory for images dir: {images_dir}")


def _sync_tree(src_dir: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for old_item in dst_dir.iterdir():
        if old_item.is_symlink() or old_item.is_file():
            old_item.unlink()
        elif old_item.is_dir():
            shutil.rmtree(old_item)

    for item in src_dir.iterdir():
        if item.is_file():
            dst_item = dst_dir / item.name
            if dst_item.exists() or dst_item.is_symlink():
                if dst_item.is_dir() and not dst_item.is_symlink():
                    shutil.rmtree(dst_item)
                else:
                    dst_item.unlink()
            dst_item.symlink_to(item)


def ensure_standard_dataset_yaml(data_yaml: str | Path, out_root: str | Path = "runs/tmp_data/datasets") -> Path:
    data_yaml_path = _resolve_path(data_yaml)
    cfg = load_yaml(data_yaml_path)
    dataset_root = resolve_dataset_root(cfg)

    split_sources: list[tuple[Path, Path]] = []
    for split in ("train", "val", "test"):
        split_path = Path(cfg[split])
        src_images_dir = (dataset_root / split_path).resolve()
        src_labels_dir = resolve_yolo_label_dir(src_images_dir)
        split_sources.append((src_images_dir, src_labels_dir))

    digest = hashlib.sha256(str(data_yaml_path).encode("utf-8")).hexdigest()[:12]
    run_token = os.environ.get("SLURM_JOB_ID") or f"pid{os.getpid()}-{uuid.uuid4().hex[:8]}"
    mirror_root = ensure_dir(out_root) / f"{data_yaml_path.stem}-{digest}-{run_token}"
    mirror_root.mkdir(parents=True, exist_ok=True)

    for split, (src_images_dir, src_labels_dir) in zip(("train", "val", "test"), split_sources, strict=True):
        dst_split_dir = mirror_root / split
        dst_images_dir = dst_split_dir / "images"
        dst_labels_dir = dst_split_dir / "labels"
        _sync_tree(src_images_dir, dst_images_dir)
        _sync_tree(src_labels_dir, dst_labels_dir)

    out_cfg = {
        "path": str(mirror_root),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": int(cfg.get("nc", 0)),
        "names": cfg.get("names", []),
    }
    out_yaml = mirror_root / "data.generated.yaml"
    out_yaml.write_text(yaml.safe_dump(out_cfg, sort_keys=False), encoding="utf-8")
    return out_yaml