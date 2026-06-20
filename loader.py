"""Higgs v3 model loading, assets, and ComfyUI memory registration."""

from __future__ import annotations

import gc
import importlib.util
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from .native import (
    HiggsAudioCodec,
    build_native_model,
    convert_modules_for_comfy,
    load_native_weights,
    load_tokenizer,
    read_config,
    set_runtime_dtype,
)

logger = logging.getLogger("Higgs_v3-TTS-ComfyUI")

MODEL_FOLDER_NAME = "higgsv3tts"
OFFICIAL_REPO_ID = "bosonai/higgs-audio-v3-tts-4b"
OFFICIAL_MODEL_NAME = "Higgs Audio v3 TTS 4B - bosonai (auto-download)"
HF_ENDPOINT = "https://huggingface.co"
DTYPE_OPTIONS = ["auto", "bf16"]
DEVICE_OPTIONS = ["auto", "cuda", "cpu"]
ATTENTION_OPTIONS = ["auto", "sdpa", "flash_attention", "sageattention"]
SMALL_ASSET_PATTERNS = [
    ".gitattributes",
    "LICENSE",
    "README.md",
    "chat_template.jinja",
    "config.json",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "assets/model_architecture.png",
]

_ACTIVE_BUNDLE: "HiggsV3Bundle | None" = None
_ACTIVE_LOAD_KEY: tuple[Any, ...] | None = None


@dataclass
class HiggsV3Bundle:
    model: torch.nn.Module
    codec: HiggsAudioCodec
    tokenizer: Any
    model_dir: Path
    device: torch.device
    torch_dtype: torch.dtype
    dtype_name: str
    attention: str
    patchers: list[Any] = field(default_factory=list)


try:
    import comfy.model_patcher as _model_patcher

    _ComfyCorePatcher = _model_patcher.CoreModelPatcher
    del _model_patcher
except Exception:
    _ComfyCorePatcher = None


def _empty_accelerator_cache() -> None:
    try:
        import comfy.model_management as mm

        mm.soft_empty_cache()
        return
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.empty_cache()


def model_dir() -> Path:
    try:
        import folder_paths

        base = Path(folder_paths.models_dir) / MODEL_FOLDER_NAME
    except Exception:
        base = Path(__file__).resolve().parent / "models" / MODEL_FOLDER_NAME
    base.mkdir(parents=True, exist_ok=True)
    return base


def register_model_folder() -> None:
    try:
        import folder_paths

        base = str(model_dir())
        if MODEL_FOLDER_NAME not in folder_paths.folder_names_and_paths:
            folder_paths.add_model_folder_path(MODEL_FOLDER_NAME, base)
        logger.info("Higgs v3 model folder registered: %s", base)
    except Exception:
        pass


def assets_dir() -> Path:
    path = Path(__file__).resolve().parent / "assets" / "higgs-audio-v3-tts-4b"
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_small_assets(download_if_missing: bool) -> Path:
    dest = assets_dir()
    required = [dest / "config.json", dest / "tokenizer.json", dest / "model.safetensors.index.json"]
    if all(path.is_file() for path in required):
        return dest
    if not download_if_missing:
        missing = [str(path) for path in required if not path.is_file()]
        raise FileNotFoundError(f"Missing Higgs v3 small assets: {missing}. Enable download_if_missing.")

    from huggingface_hub import snapshot_download

    logger.info("Downloading Higgs v3 small assets to %s", dest)
    kwargs = {
        "repo_id": OFFICIAL_REPO_ID,
        "local_dir": str(dest),
        "allow_patterns": SMALL_ASSET_PATTERNS,
        "ignore_patterns": ["model.safetensors"],
        "endpoint": HF_ENDPOINT,
    }
    snapshot_download(**kwargs)
    return dest


def _copy_small_assets_to_model_dir(runtime_dir: Path, asset_dir: Path) -> None:
    for rel in SMALL_ASSET_PATTERNS:
        src = asset_dir / rel
        if not src.is_file():
            continue
        dst = runtime_dir / rel
        if dst.is_file():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _has_model_file(path: Path) -> bool:
    return (path / "model.safetensors").is_file()


def get_model_choices() -> list[str]:
    base = model_dir()
    choices = [OFFICIAL_MODEL_NAME]
    if _has_model_file(base):
        choices.append("local root: ComfyUI/models/higgsv3tts")
    try:
        for entry in sorted(base.iterdir()):
            if entry.is_dir() and _has_model_file(entry):
                choices.append(entry.name)
    except OSError:
        pass
    return choices


def _download_full_model(target_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download

    logger.info("Downloading Higgs v3 model.safetensors to %s. This is a large download.", target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    hf_hub_download(
        repo_id=OFFICIAL_REPO_ID,
        filename="model.safetensors",
        local_dir=str(target_dir),
        endpoint=HF_ENDPOINT,
    )
    return target_dir


def resolve_model_dir(model_choice: str, download_if_missing: bool) -> Path:
    base = model_dir()
    if model_choice == "local root: ComfyUI/models/higgsv3tts" and _has_model_file(base):
        path = base
    elif model_choice == OFFICIAL_MODEL_NAME:
        official_dir = base / "higgs-audio-v3-tts-4b"
        path = official_dir if _has_model_file(official_dir) else base if _has_model_file(base) else official_dir
        if not _has_model_file(path):
            if not download_if_missing:
                raise FileNotFoundError(
                    f"Missing model.safetensors. Put it in {official_dir} or {base}, or enable download_if_missing."
                )
            path = _download_full_model(official_dir)
    else:
        path = base / model_choice
        if not _has_model_file(path):
            raise FileNotFoundError(f"Higgs model folder does not contain model.safetensors: {path}")

    asset_dir = ensure_small_assets(download_if_missing)
    _copy_small_assets_to_model_dir(path, asset_dir)
    return path


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        try:
            import comfy.model_management as mm

            return torch.device(mm.get_torch_device())
        except Exception:
            pass
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was selected, but torch.cuda is not available.")
    return device


def resolve_dtype(dtype_name: str, device: torch.device) -> torch.dtype:
    if dtype_name == "auto":
        if device.type == "cuda" and torch.cuda.is_available():
            try:
                return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
            except Exception:
                return torch.float32
        return torch.float32
    if dtype_name == "bf16":
        if device.type == "cuda" and torch.cuda.is_available():
            try:
                if not torch.cuda.is_bf16_supported():
                    raise RuntimeError("bf16 was selected, but this CUDA device does not report bf16 support. Use dtype=auto to fall back to fp32.")
            except RuntimeError:
                raise
            except Exception:
                pass
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def resolve_attention(attention: str) -> tuple[str, str | None]:
    if attention in {"auto", "sdpa"}:
        return "sdpa", "sdpa"
    if attention == "flash_attention":
        if importlib.util.find_spec("flash_attn") is None:
            raise ImportError("flash_attention was selected, but flash_attn is not installed.")
        return "flash_attention", "flash_attention_2"
    if attention == "sageattention":
        if importlib.util.find_spec("sageattention") is None:
            raise ImportError("sageattention was selected, but sageattention is not installed.")
        return "sageattention", "sdpa"
    raise ValueError(f"Unsupported attention mode: {attention}")


def dynamic_vram_active(device: torch.device) -> bool:
    if torch.device(device).type == "cpu":
        return False
    try:
        import comfy.memory_management

        if not bool(comfy.memory_management.aimdo_enabled):
            return False
        try:
            import comfy_aimdo.control
            import comfy_aimdo.host_buffer
            import comfy_aimdo.model_vbar

            return (
                comfy_aimdo.control.lib is not None
                and comfy_aimdo.host_buffer.lib is not None
                and comfy_aimdo.model_vbar.lib is not None
            )
        except Exception:
            return False
    except Exception:
        return False


def _register_many_with_comfy(patchers: list[Any]) -> None:
    patchers = [
        patcher
        for patcher in patchers
        if patcher is not None and patcher.load_device.type != "cpu"
    ]
    if not patchers:
        return
    try:
        import comfy.model_management as mm

        already_loaded = {
            id(loaded.model)
            for loaded in mm.current_loaded_models
            if loaded.model is not None
        }
        to_load = [patcher for patcher in patchers if id(patcher) not in already_loaded]
        if not to_load:
            return
        mm.load_models_gpu(to_load)
        for patcher in to_load:
            logger.info(
                "Loaded %s through ComfyUI%s memory management.",
                patcher.model.__class__.__name__,
                "/AIMDO" if patcher.is_dynamic() else "",
            )
    except Exception as exc:
        raise RuntimeError("Could not load model through ComfyUI memory management.") from exc


def _unregister_from_comfy(patcher: Any) -> None:
    try:
        import comfy.model_management as mm

        survivors = []
        for loaded in mm.current_loaded_models:
            if loaded.model is patcher:
                try:
                    if loaded.model_finalizer is not None:
                        loaded.model_finalizer.detach()
                    loaded.model_finalizer = None
                    loaded.real_model = None
                except Exception:
                    pass
                try:
                    finalizer = getattr(loaded, "_patcher_finalizer", None)
                    if finalizer is not None:
                        finalizer.detach()
                    loaded._patcher_finalizer = None
                except Exception:
                    pass
                continue
            survivors.append(loaded)
        mm.current_loaded_models[:] = survivors
    except Exception:
        pass


def _set_module_device_if_writable(module: torch.nn.Module, device: torch.device) -> None:
    try:
        module.device = torch.device(device)
    except (AttributeError, RuntimeError, TypeError):
        pass


def _ensure_writable_device_property(module: torch.nn.Module) -> None:
    cls = module.__class__
    prop = getattr(cls, "device", None)
    if not isinstance(prop, property) or prop.fset is not None:
        return
    if getattr(module, "_higgs_writable_device_property", False):
        return

    def _get_device(self):
        runtime_device = self.__dict__.get("_higgs_runtime_device")
        if runtime_device is not None:
            return runtime_device
        return prop.fget(self)

    def _set_device(self, value):
        self.__dict__["_higgs_runtime_device"] = torch.device(value)

    writable_cls = type(
        cls.__name__,
        (cls,),
        {
            "device": property(_get_device, _set_device),
            "_higgs_device_base_class": cls,
            "__module__": cls.__module__,
        },
    )
    module.__class__ = writable_cls
    module._higgs_writable_device_property = True


def register_runtime_module(module: torch.nn.Module, device: torch.device, *, dynamic: bool | None = None) -> Any:
    device = torch.device(device)
    module._higgs_runtime_device = torch.device(device)
    _ensure_writable_device_property(module)
    if _ComfyCorePatcher is None or device.type == "cpu":
        module.to(device)
        return None

    import comfy.model_patcher as model_patcher

    use_dynamic = dynamic_vram_active(device) and dynamic is not False
    patcher_class = (
        model_patcher.ModelPatcherDynamic
        if use_dynamic
        else model_patcher.ModelPatcher
    )
    patcher = patcher_class(module, load_device=device, offload_device=torch.device("cpu"))
    module.model_loaded_weight_memory = 0
    _register_many_with_comfy([patcher])
    if not patcher.is_dynamic():
        _set_module_device_if_writable(module, device)
    logger.info(
        "Registered %s with ComfyUI%s memory management.",
        module.__class__.__name__,
        "/AIMDO" if patcher.is_dynamic() else "",
    )
    return patcher


def resume_runtime_module(patcher: Any, device: torch.device) -> None:
    del device
    if patcher is not None:
        _register_many_with_comfy([patcher])


def unload_runtime_module(patcher: Any, *, hard: bool = True) -> None:
    if patcher is None:
        return
    _unregister_from_comfy(patcher)
    try:
        patcher.detach()
    except Exception:
        pass


def resume_bundle_to_device(bundle: HiggsV3Bundle) -> None:
    for patcher in bundle.patchers:
        resume_runtime_module(patcher, bundle.device)


def unload_higgs_bundle(bundle: HiggsV3Bundle | None, reason: str = "manual unload", hard: bool = True) -> None:
    global _ACTIVE_BUNDLE, _ACTIVE_LOAD_KEY
    if bundle is None:
        return
    logger.info("Unloading Higgs v3 bundle (%s).", reason)
    for patcher in list(bundle.patchers):
        unload_runtime_module(patcher, hard=hard)
    if not hard:
        try:
            bundle.model.to("cpu")
        except Exception:
            pass
        try:
            bundle.codec.model.to("cpu")
        except Exception:
            pass
    for module in (bundle.model, bundle.codec.model):
        try:
            module.model_loaded_weight_memory = 0
            if hasattr(module, "dynamic_vbars"):
                module.dynamic_vbars.clear()
            if hard and hasattr(module, "to_empty"):
                module.to_empty(device=torch.device("meta"))
        except Exception:
            pass
    bundle.patchers.clear()
    if hard:
        try:
            bundle.model = None
            bundle.codec = None
            bundle.tokenizer = None
        except Exception:
            pass
    gc.collect()
    _empty_accelerator_cache()
    if _ACTIVE_BUNDLE is bundle:
        _ACTIVE_BUNDLE = None
        _ACTIVE_LOAD_KEY = None


def load_higgs_bundle(
    model_choice: str,
    dtype_name: str,
    device_name: str,
    attention: str,
    download_if_missing: bool,
) -> HiggsV3Bundle:
    global _ACTIVE_BUNDLE, _ACTIVE_LOAD_KEY

    register_model_folder()
    runtime_dir = resolve_model_dir(model_choice, download_if_missing)
    device = resolve_device(device_name)
    torch_dtype = resolve_dtype(dtype_name, device)
    runtime_attention, hf_attention = resolve_attention(attention)

    model_file = runtime_dir / "model.safetensors"
    load_key = (
        str(runtime_dir.resolve()),
        model_file.stat().st_mtime_ns,
        str(device),
        str(torch_dtype),
        runtime_attention,
    )
    if _ACTIVE_BUNDLE is not None and _ACTIVE_LOAD_KEY == load_key:
        resume_bundle_to_device(_ACTIVE_BUNDLE)
        return _ACTIVE_BUNDLE
    if _ACTIVE_BUNDLE is not None:
        unload_higgs_bundle(_ACTIVE_BUNDLE, reason="load settings changed")

    config = read_config(runtime_dir)
    logger.info("Loading Higgs v3 from %s on %s with dtype=%s attention=%s", runtime_dir, device, torch_dtype, runtime_attention)
    weight_device = torch.device("cpu") if device.type != "cpu" else device
    model = build_native_model(config, torch_dtype, hf_attention)
    load_native_weights(model, runtime_dir, weight_device, torch_dtype)
    codec = HiggsAudioCodec.from_pretrained(runtime_dir, device=weight_device, dtype=torch_dtype, runtime_device=device)
    convert_modules_for_comfy(model)
    convert_modules_for_comfy(codec.model)
    set_runtime_dtype(model, torch_dtype)
    set_runtime_dtype(codec.model, torch_dtype)
    tokenizer = load_tokenizer(runtime_dir)

    patchers: list[Any] = []
    try:
        use_dynamic = dynamic_vram_active(device)
        if use_dynamic:
            logger.info("AIMDO DynamicVRAM is active; using dynamic patchers for Higgs v3 model and codec.")
        else:
            logger.info("AIMDO not active; using static ComfyUI memory management.")
        model_patcher = register_runtime_module(model, device, dynamic=use_dynamic)
        if model_patcher is not None:
            patchers.append(model_patcher)
        codec_patcher = register_runtime_module(codec.model, device, dynamic=use_dynamic)
        if codec_patcher is not None:
            patchers.append(codec_patcher)
    except Exception:
        for patcher in list(patchers):
            unload_runtime_module(patcher, hard=True)
        for module in (model, codec.model):
            try:
                module.model_loaded_weight_memory = 0
                if hasattr(module, "dynamic_vbars"):
                    module.dynamic_vbars.clear()
                if hasattr(module, "to_empty"):
                    module.to_empty(device=torch.device("meta"))
            except Exception:
                pass
        gc.collect()
        _empty_accelerator_cache()
        raise

    bundle = HiggsV3Bundle(
        model=model,
        codec=codec,
        tokenizer=tokenizer,
        model_dir=runtime_dir,
        device=device,
        torch_dtype=torch_dtype,
        dtype_name=dtype_name,
        attention=runtime_attention,
        patchers=patchers,
    )
    _ACTIVE_BUNDLE = bundle
    _ACTIVE_LOAD_KEY = load_key
    install_comfy_unload_hook()
    _empty_accelerator_cache()
    return bundle


def unload_active_bundle() -> None:
    unload_higgs_bundle(_ACTIVE_BUNDLE, reason="active unload")


def install_comfy_unload_hook() -> None:
    """Patch ComfyUI unload calls so the active native Higgs bundle hard-releases."""
    try:
        import comfy.model_management as mm
    except Exception:
        return

    if getattr(mm, "_higgsv3_unload_hook_installed", False):
        return

    original_unload_all_models = mm.unload_all_models

    def unload_all_models_with_higgsv3(*args, **kwargs):
        try:
            return original_unload_all_models(*args, **kwargs)
        finally:
            unload_higgs_bundle(_ACTIVE_BUNDLE, reason="ComfyUI unload_all_models")

    mm.unload_all_models = unload_all_models_with_higgsv3

    original_unload_model_and_clones = getattr(mm, "unload_model_and_clones", None)
    if original_unload_model_and_clones is not None:
        def unload_model_and_clones_with_higgsv3(model, *args, **kwargs):
            try:
                return original_unload_model_and_clones(model, *args, **kwargs)
            finally:
                if _ACTIVE_BUNDLE is not None and model is not None:
                    owned = list(_ACTIVE_BUNDLE.patchers) + [_ACTIVE_BUNDLE.model, _ACTIVE_BUNDLE.codec.model]
                    if any(existing is model or existing is getattr(model, "model", None) for existing in owned if existing is not None):
                        unload_higgs_bundle(_ACTIVE_BUNDLE, reason="ComfyUI unload_model_and_clones")

        mm.unload_model_and_clones = unload_model_and_clones_with_higgsv3

    mm._higgsv3_unload_hook_installed = True
    logger.debug("Installed Higgs v3 unload hook for ComfyUI native unload.")
