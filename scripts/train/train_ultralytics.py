from __future__ import annotations

"""Train Ultralytics detector families (v8/v11/latest) from unified YAML+CLI config."""

import argparse
import os
from pathlib import Path

from ultralytics import YOLO

from scripts.common.attention_utils import inject_cbam_attention
from scripts.common.io_utils import ROOT, load_yaml, merge_dicts, now_tag

MODELS_CFG_PATH = "configs/models.yaml"
FAMILY_KEYS = {"yolov5", "yolov8", "yolov11", "latest"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Ultralytics family models (v8/v11/latest)")
    parser.add_argument("--family", choices=["v8", "v11", "latest"], required=True)
    parser.add_argument("--model", default=None, help="Override model checkpoint, e.g. yolo11n.pt")
    parser.add_argument(
        "--data",
        default=None,
        help="Path to prebuilt dataset YAML. Filtering is no longer done during training.",
    )
    parser.add_argument("--hyp", default=None, help="Optional legacy hyperparameter YAML override")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--project", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument(
        "--use-attention",
        action="store_true",
        help="Inject lightweight CBAM attention blocks into the training model.",
    )
    parser.add_argument(
        "--attention-max-blocks",
        type=int,
        default=4,
        help="Maximum number of feature blocks to patch when --use-attention is enabled.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    all_models = load_yaml(MODELS_CFG_PATH)
    # base config is the top-level shared keys in the consolidated models.yaml
    if args.hyp:
        base_cfg = load_yaml(args.hyp)
    else:
        base_cfg = {k: v for k, v in all_models.items() if k not in FAMILY_KEYS}
    # Map CLI family names like 'v8'/'v11' to keys in `configs/models.yaml`
    # which use names like 'yolov8'/'yolov11'. Leave 'latest' as-is.
    fam_key = args.family
    if isinstance(fam_key, str) and fam_key.startswith("v"):
        fam_key = f"yolo{fam_key[1:]}"
    model_cfg = all_models.get(fam_key, {})

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

    # Resolve workers with priority: CLI > env WORKERS > auto-calc from SLURM_CPUS_PER_TASK/GPUs > hyp/base default
    if cfg.get("workers") is None:
        # 1) env override
        env_workers = os.getenv("WORKERS")
        if env_workers:
            try:
                cfg["workers"] = int(env_workers)
            except ValueError:
                pass
        else:
            # 2) attempt auto-calc
            # determine number of GPUs requested/available
            device_val = cfg.get("device")
            num_gpus = None
            if isinstance(device_val, str):
                if device_val == "auto":
                    # try SLURM hint
                    slurm_gpus = os.getenv("SLURM_GPUS_ON_NODE")
                    if slurm_gpus and slurm_gpus.isdigit():
                        num_gpus = int(slurm_gpus)
                elif "," in device_val:
                    num_gpus = device_val.count(",") + 1
                elif device_val.isdigit():
                    num_gpus = 1
            # fallback to torch if available
            if not num_gpus or num_gpus < 1:
                try:
                    import torch

                    num_gpus = torch.cuda.device_count()
                except Exception:
                    num_gpus = None
            if not num_gpus or num_gpus < 1:
                num_gpus = 1

            total_cpus = None
            try:
                total_cpus = int(os.getenv("SLURM_CPUS_PER_TASK") or 0)
            except Exception:
                total_cpus = 0
            if not total_cpus:
                try:
                    total_cpus = os.cpu_count() or 1
                except Exception:
                    total_cpus = 1

            cfg["workers"] = max(1, total_cpus // max(1, int(num_gpus)))

    data_value = cfg.get("data")
    if not data_value:
        raise ValueError("Missing training data config. Set data in the model YAML or pass --data.")

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

    # Determine whether to inject attention. Prefer explicit CLI flag, otherwise
    # fall back to the `use_attention` key from config. Remove the key from
    # `cfg` so it is not passed to Ultralytics' `model.train` (Ultralytics
    # rejects unknown kwargs).
    use_attention_cfg = cfg.pop("use_attention", False)
    use_attention_flag = args.use_attention or bool(use_attention_cfg)

    model_path = cfg.pop("model")
    model = YOLO(model_path)
    if use_attention_flag:
        injected = inject_cbam_attention(model.model, max_blocks=args.attention_max_blocks)
        if injected < 1:
            raise RuntimeError(
                "Failed to inject attention blocks. Check model structure or disable --use-attention."
            )
        print(f"Injected CBAM attention blocks: {injected}")

    model.train(**cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
