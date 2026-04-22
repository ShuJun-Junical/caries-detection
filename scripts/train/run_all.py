from __future__ import annotations

"""Run selected training families sequentially for smoke testing."""

import argparse
import subprocess
from pathlib import Path

from scripts.common.io_utils import ROOT


def run_cmd(cmd: list[str], dry_run: bool) -> int:
    print(" ".join(cmd))
    if dry_run:
        return 0
    return subprocess.run(cmd, cwd=str(ROOT), check=False).returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLOv5/v8/v11/latest training targets in sequence")
    parser.add_argument(
        "--targets",
        nargs="+",
        default=["v5", "v8", "v11", "latest"],
        choices=["v5", "v8", "v11", "latest"],
    )
    parser.add_argument("--epochs", type=int, default=1, help="Use small value for smoke tests")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--yolov5-dir", default="third_party/yolov5")
    parser.add_argument(
        "--use-attention",
        action="store_true",
        help="Enable attention mechanism for supported training targets.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    failures: list[tuple[str, int]] = []

    for target in args.targets:
        if target == "v5":
            cmd = [
                "python",
                "-m",
                "scripts.train.train_yolov5",
                "--yolov5-dir",
                args.yolov5_dir,
                "--epochs",
                str(args.epochs),
            ]
        else:
            cmd = [
                "python",
                "-m",
                "scripts.train.train_ultralytics",
                "--family",
                target,
                "--epochs",
                str(args.epochs),
            ]

        if args.use_attention:
            cmd.append("--use-attention")

        if args.dry_run:
            cmd.append("--dry-run") if target != "v5" else None

        code = run_cmd(cmd, args.dry_run)
        if code != 0:
            failures.append((target, code))

    if failures:
        print("Failed targets:")
        for target, code in failures:
            print(f"- {target}: exit code {code}")
        return 1

    print("All selected targets completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
