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
    parser.add_argument("--config", default="configs/models.yaml", help="Path to the consolidated model YAML")
    parser.add_argument("--device", required=True, help="Training device setting passed to the target scripts")
    parser.add_argument("--workers", type=int, required=True, help="Dataloader worker count passed to the target scripts")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--yolov5-dir", default="third_party/yolov5")
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
                "--config",
                args.config,
                "--device",
                args.device,
                "--workers",
                str(args.workers),
            ]
        else:
            cmd = [
                "python",
                "-m",
                "scripts.train.train_ultralytics",
                "--family",
                target,
                "--config",
                args.config,
                "--device",
                args.device,
                "--workers",
                str(args.workers),
            ]

        if args.dry_run:
            cmd.append("--dry-run")

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
