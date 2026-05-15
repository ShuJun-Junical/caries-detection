"""Project-local Python startup hook.

This keeps project-local modules such as CBAM visible inside subprocesses
spawned by Ultralytics DDP, as long as the repository root is on PYTHONPATH.
"""

from __future__ import annotations

try:
    from scripts.common.attention_utils import register_ultralytics_cbam
except Exception:
    pass
else:
    register_ultralytics_cbam()
