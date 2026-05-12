from __future__ import annotations

"""Train Ultralytics detector families (v8/v11/v26) from unified YAML config."""

import argparse
from pathlib import Path

from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import LOGGER

from scripts.common.attention_utils import inject_cbam_attention
from scripts.common.io_utils import ROOT, load_yaml, merge_dicts, now_tag

MODELS_CFG_PATH = "configs/models.yaml"
FAMILY_KEYS = {"v5", "v8", "v11", "v26"}


def _count_cbam_modules(module) -> int:
    return sum(1 for m in module.modules() if m.__class__.__name__ == "CBAM")


def _count_attention_state_keys(module) -> int:
    keys = module.state_dict().keys()
    return sum(1 for k in keys if "channel_attention" in k or "spatial_attention" in k)


def _build_attention_trainer(enable_attention: bool):
    class AttentionTrainer(DetectionTrainer):
        def get_model(self, cfg: str | None = None, weights=None, verbose: bool = True):
            model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
            if not enable_attention:
                return model

            if _count_cbam_modules(model) > 0:
                return model

            params_before = sum(p.numel() for p in model.parameters())
            injected = inject_cbam_attention(model)
            params_after = sum(p.numel() for p in model.parameters())
            cbam_blocks = _count_cbam_modules(model)
            attn_keys = _count_attention_state_keys(model)

            if injected < 1 or cbam_blocks < 1 or attn_keys < 1 or params_after <= params_before:
                raise RuntimeError(
                    "CBAM injection validation failed in trainer.get_model(). "
                    "Expected positive injected blocks, attention keys, and parameter delta."
                )

            LOGGER.info(
                "Injected CBAM in trainer.get_model(): "
                f"injected={injected}, cbam_blocks={cbam_blocks}, "
                f"attention_keys={attn_keys}, delta_params={params_after - params_before}"
            )
            return model

    return AttentionTrainer


def _verify_attention_on_train_start(trainer) -> None:
    model = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
    cbam_blocks = _count_cbam_modules(model)
    attn_keys = _count_attention_state_keys(model)
    if cbam_blocks < 1 or attn_keys < 1:
        raise RuntimeError(
            "CBAM verification failed at on_train_start: "
            f"cbam_blocks={cbam_blocks}, attention_keys={attn_keys}"
        )
    LOGGER.info(
        "Verified CBAM at on_train_start: "
        f"cbam_blocks={cbam_blocks}, attention_keys={attn_keys}"
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

    if args.dry_run:
        print(cfg)
        return 0

    use_attention_flag = bool(cfg.pop("use_attention", False))

    model_path = Path(cfg.pop("model"))
    if not model_path.is_absolute():
        model_path = ROOT / model_path
    model = YOLO(model_path)
    trainer_cls = _build_attention_trainer(use_attention_flag)
    if use_attention_flag:
        model.add_callback("on_train_start", _verify_attention_on_train_start)

    model.train(trainer=trainer_cls, **cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
