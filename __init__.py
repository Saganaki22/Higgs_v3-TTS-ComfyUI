"""ComfyUI custom nodes for Higgs Audio v3 TTS."""

from __future__ import annotations

__version__ = "v0.1.71"

import importlib.metadata as _metadata
import importlib.util
import logging
import sys
import types
from typing import Any

logger = logging.getLogger("Higgs_v3-TTS-ComfyUI")
logger.propagate = False
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[Higgs_v3-TTS-ComfyUI] %(message)s"))
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)


def _block_broken_torchcodec() -> None:
    """Prevent incompatible torchcodec wheels from killing Transformers audio imports."""
    broken = False
    if "torchcodec" not in sys.modules:
        try:
            import torchcodec  # noqa: F401
        except Exception:
            broken = True

    tc = sys.modules.get("torchcodec")
    if not broken and tc is not None and getattr(tc, "__spec__", None) is not None:
        return

    stub = types.ModuleType("torchcodec")
    stub.__path__ = []
    stub.__package__ = "torchcodec"
    stub.__spec__ = importlib.util.spec_from_loader("torchcodec", loader=None, origin="torchcodec")
    for sub in ("decoders", "encoders", "samplers", "transforms", "_core"):
        sub_mod = types.ModuleType(f"torchcodec.{sub}")
        sub_mod.__spec__ = importlib.util.spec_from_loader(f"torchcodec.{sub}", loader=None)
        if sub == "decoders":
            class _AudioDecoder:
                pass

            sub_mod.AudioDecoder = _AudioDecoder
        setattr(stub, sub, sub_mod)
        sys.modules[f"torchcodec.{sub}"] = sub_mod
    sys.modules["torchcodec"] = stub

    original_version = _metadata.version

    def patched_version(name: str) -> str:
        if name == "torchcodec":
            return "0.0.0"
        return original_version(name)

    _metadata.version = patched_version
    logger.info("torchcodec is unavailable or incompatible; using audio fallback paths.")


_block_broken_torchcodec()

NODE_CLASS_MAPPINGS: dict[str, Any] = {}
NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {}

try:
    from .nodes import NODE_CLASS_MAPPINGS as _NODE_CLASS_MAPPINGS
    from .nodes import NODE_DISPLAY_NAME_MAPPINGS as _NODE_DISPLAY_NAME_MAPPINGS

    NODE_CLASS_MAPPINGS.update(_NODE_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(_NODE_DISPLAY_NAME_MAPPINGS)
    logger.info("Registered %d node(s).", len(NODE_CLASS_MAPPINGS))
except Exception as exc:
    logger.error("Failed to register Higgs v3 TTS nodes: %s", exc, exc_info=True)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "__version__"]
