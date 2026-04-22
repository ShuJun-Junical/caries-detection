from __future__ import annotations

"""Prepare single-class (caries-only) dataset view before training jobs."""

import argparse
from pathlib import Path

from scripts.common.dataset_filters import build_caries_only_data_yaml
from scripts.common.io_utils import ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build caries-only dataset view on login node before training"
    )
    parser.add_argument("--data", default="dataset/data.caries.yaml")
    parser.add_argument("--out-dir", default="dataset/caries_only")
    parser.add_argument("--keep-class-id", type=int, default=0)
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

    if args.dry_run:
        print(
            {
                "data": str(data_path),
                "out_dir": str(out_dir),
                "keep_class_id": args.keep_class_id,
            }
        )
        return 0

    out_yaml = build_caries_only_data_yaml(
        data_yaml=data_path,
        keep_class_id=args.keep_class_id,
        out_dir=out_dir,
    )
    print(str(out_yaml))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
