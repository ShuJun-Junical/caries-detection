from __future__ import annotations

"""Train Ultralytics detector families (v8/v11/v26) from unified YAML config."""

import argparse
import os
from pathlib import Path

from ultralytics import YOLO
from ultralytics.utils import LOGGER

from scripts.common.attention_utils import (
    count_attention_state_keys,
    count_cbam_modules,
    register_ultralytics_cbam,
)
from scripts.common.io_utils import ROOT, load_yaml, merge_dicts, now_tag

MODELS_CFG_PATH = "configs/models.yaml"
FAMILY_KEYS = {"v5", "v8", "v11", "v26"}


def _resolve_path(value: str | os.PathLike[str]) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return path


def _verify_attention_model(model: YOLO) -> None:
    cbam_blocks = count_cbam_modules(model.model)
    attn_keys = count_attention_state_keys(model.model)
    if cbam_blocks < 1 or attn_keys < 1:
        raise RuntimeError(
            "CBAM verification failed for YAML-declared attention model: "
            f"cbam_blocks={cbam_blocks}, attention_keys={attn_keys}"
        )
    LOGGER.info(
        "Verified YAML-declared CBAM model: "
        f"cbam_blocks={cbam_blocks}, attention_keys={attn_keys}"
    )


def _load_attention_weights(model: YOLO, base_weights_path: Path) -> None:
    base_model = YOLO(base_weights_path)
    target_layers = [layer for layer in model.model.model if layer.__class__.__name__ != "CBAM"]
    source_layers = list(base_model.model.model)

    if len(target_layers) != len(source_layers):
        raise RuntimeError(
            "Attention weight transfer failed: non-CBAM layer count does not match base model "
            f"({len(target_layers)} != {len(source_layers)})"
        )

    loaded_tensors = 0
    total_tensors = 0
    for target_layer, source_layer in zip(target_layers, source_layers, strict=True):
        source_state = source_layer.state_dict()
        incompatible = target_layer.load_state_dict(source_state, strict=False)
        if incompatible.unexpected_keys:
            raise RuntimeError(
                "Attention weight transfer hit unexpected keys: "
                f"{incompatible.unexpected_keys}"
            )
        total_tensors += len(source_state)
        loaded_tensors += len(source_state) - len(incompatible.missing_keys)

    LOGGER.info(
        "Transferred pretrained tensors into YAML-declared CBAM model: "
        f"{loaded_tensors}/{total_tensors}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Ultralytics family models (v8/v11/v26)")
    parser.add_argument("--config", default=MODELS_CFG_PATH, help="Path to the consolidated model YAML")
    parser.add_argument("--family", choices=["v8", "v11", "v26"], required=True)
    parser.add_argument("--device", required=True, help="Training device setting passed to Ultralytics")
    parser.add_argument("--workers", type=int, required=True, help="Dataloader worker count")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    all_models = load_yaml(args.config)
    base_cfg = {k: v for k, v in all_models.items() if k not in FAMILY_KEYS}
    model_cfg = all_models.get(args.family)
    if model_cfg is None:
        raise KeyError(f"Missing family section in model config: {args.family}")

    cfg = merge_dicts(base_cfg, model_cfg)
    cfg["device"] = args.device
    cfg["workers"] = args.workers

    data_value = cfg.get("data")
    if not data_value:
        raise ValueError("Missing training data config. Set data in the model YAML or pass --data.")

    data_path = Path(data_value)
    if not data_path.is_absolute():
        data_path = ROOT / data_path
    cfg["data"] = str(data_path)

    project_path = Path(cfg.get("project", Path("runs") / args.family))
    if not project_path.is_absolute():
        project_path = ROOT / project_path
    cfg["project"] = str(project_path)
    cfg["name"] = now_tag()

    root_path = str(ROOT)
    current_pythonpath = os.environ.get("PYTHONPATH")
    if current_pythonpath:
        paths = current_pythonpath.split(os.pathsep)
        if root_path not in paths:
            os.environ["PYTHONPATH"] = os.pathsep.join([root_path, current_pythonpath])
    else:
        os.environ["PYTHONPATH"] = root_path

    if args.dry_run:
        print(cfg)
        return 0

    use_attention_flag = bool(cfg.pop("use_attention", False))
    configured_model = cfg.pop("model")
    attention_model = cfg.pop("attention_model", None)
    resume_from = os.environ.get("RESUME_FROM")

    if use_attention_flag:
        register_ultralytics_cbam()

    model_path = _resolve_path(resume_from if resume_from else configured_model)
    if resume_from:
        cfg["resume"] = True
        model = YOLO(model_path)
        if use_attention_flag:
            _verify_attention_model(model)
    elif use_attention_flag:
        if not attention_model:
            raise ValueError(f"Missing attention_model for family {args.family} while use_attention=true")
        attention_model_path = _resolve_path(attention_model)
        model = YOLO(attention_model_path)
        _load_attention_weights(model, model_path)
        _verify_attention_model(model)
    else:
        model = YOLO(model_path)

    model.train(**cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
