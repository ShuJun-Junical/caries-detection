from __future__ import annotations

"""Train Ultralytics detector families (v8/v11/latest) from unified YAML config."""

import argparse
from pathlib import Path

from ultralytics import YOLO

from scripts.common.attention_utils import inject_cbam_attention
from scripts.common.dataset_utils import ensure_standard_dataset_yaml
from scripts.common.io_utils import ROOT, load_yaml, merge_dicts, now_tag

MODELS_CFG_PATH = "configs/models.yaml"
FAMILY_KEYS = {"yolov5", "yolov8", "yolov11", "latest"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Ultralytics family models (v8/v11/latest)")
    parser.add_argument("--config", default=MODELS_CFG_PATH, help="Path to the consolidated model YAML")
    parser.add_argument("--family", choices=["v8", "v11", "latest"], required=True)
    parser.add_argument("--device", required=True, help="Training device setting passed to Ultralytics")
    parser.add_argument("--workers", type=int, required=True, help="Dataloader worker count")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    all_models = load_yaml(args.config)
    base_cfg = {k: v for k, v in all_models.items() if k not in FAMILY_KEYS}
    fam_key = f"yolov{args.family[1:]}" if args.family.startswith("v") else args.family
    model_cfg = all_models.get(fam_key)
    if model_cfg is None:
        raise KeyError(f"Missing family section in model config: {fam_key}")

    cfg = merge_dicts(base_cfg, model_cfg)
    cfg["device"] = args.device
    cfg["workers"] = args.workers

    data_value = cfg.get("data")
    if not data_value:
        raise ValueError("Missing training data config. Set data in the model YAML or pass --data.")

    cfg["data"] = str(ensure_standard_dataset_yaml(cfg["data"]))

    project = cfg.get("project", f"runs/{args.family}")
    if not Path(project).is_absolute():
        project = str(ROOT / project)
    cfg["project"] = project

    run_name = cfg.get("name") or f"caries-{args.family}-{now_tag()}"
    cfg["name"] = run_name

    if args.dry_run:
        print(cfg)
        return 0

    use_attention_flag = bool(cfg.pop("use_attention", False))

    model_path = Path(cfg.pop("model"))
    if not model_path.is_absolute():
        model_path = ROOT / model_path
    model = YOLO(model_path)
    if use_attention_flag:
        injected = inject_cbam_attention(model.model)
        if injected < 1:
            raise RuntimeError(
                "Failed to inject attention blocks. Check model structure or disable use_attention in YAML."
            )
        print(f"Injected CBAM attention blocks: {injected}")

    model.train(**cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
