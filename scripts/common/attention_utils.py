from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        return self.act(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        if kernel_size not in (3, 7):
            raise ValueError("SpatialAttention kernel_size must be 3 or 7")
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        features = torch.cat([avg_out, max_out], dim=1)
        return self.act(self.conv(features))


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7) -> None:
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction=reduction)
        self.spatial_attention = SpatialAttention(kernel_size=spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x) * x
        x = self.spatial_attention(x) * x
        return x


def _iter_named_children(module: nn.Module) -> Iterable[tuple[str, nn.Module]]:
    for name, child in module.named_children():
        yield name, child


def _infer_out_channels(module: nn.Module) -> int | None:
    if hasattr(module, "c2") and isinstance(module.c2, int):
        return int(module.c2)

    out_channels = getattr(module, "out_channels", None)
    if isinstance(out_channels, int):
        return out_channels

    for attr_name in ("cv2", "cv3", "conv"):
        attr = getattr(module, attr_name, None)
        if attr is None:
            continue
        nested_out = getattr(attr, "out_channels", None)
        if isinstance(nested_out, int):
            return nested_out
        if hasattr(attr, "conv"):
            nested_conv_out = getattr(attr.conv, "out_channels", None)
            if isinstance(nested_conv_out, int):
                return nested_conv_out

    return None


def inject_cbam_attention(model: nn.Module, max_blocks: int = 4) -> int:
    """Inject CBAM after feature blocks and return how many blocks were patched."""
    if max_blocks < 1:
        raise ValueError("max_blocks must be >= 1")

    target_types = {"C2f", "C3", "C3k2", "BottleneckCSP"}
    injected = 0

    def patch(module: nn.Module) -> None:
        nonlocal injected
        if injected >= max_blocks:
            return

        for name, child in list(_iter_named_children(module)):
            if injected >= max_blocks:
                return

            child_type = child.__class__.__name__
            if child_type in target_types and not isinstance(child, nn.Sequential):
                channels = _infer_out_channels(child)
                if channels and channels > 0:
                    setattr(module, name, nn.Sequential(child, CBAM(channels)))
                    injected += 1
                    continue

            patch(child)

    patch(model)
    return injected
