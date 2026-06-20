"""Native in-process Higgs Audio v3 TTS runtime."""

from __future__ import annotations

import gc
import json
import logging
import math
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from safetensors import safe_open
from tokenizers import Tokenizer
from torch import nn
try:
    from comfy.ops import cast_bias_weight, uncast_bias_weight
except Exception:
    def cast_bias_weight(model, input=None, **kwargs):
        return model.weight, getattr(model, "bias", None), None

    def uncast_bias_weight(model, weight, bias, offload_stream):
        return None

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None
from transformers import (
    HiggsAudioV2TokenizerConfig,
    HiggsAudioV2TokenizerModel,
    PreTrainedTokenizerFast,
    Qwen3Config,
    Qwen3ForCausalLM,
)

logger = logging.getLogger("Higgs_v3-TTS-ComfyUI")

BOC_ID = 1024
EOC_ID = 1025
STOP_CODE = -1
K_MAX = 1026
AUDIO_PLACEHOLDER_ID = -100
SAMPLE_RATE = 24_000
CODEC_PREFIX = "tied.embedding.modality_embeddings.0.model."
CODEC_CONFIG_PATH = Path(__file__).resolve().parent / "assets" / "higgs_audio_v2_tokenizer_config.json"


class HiggsFusedMultiTextEmbedding(nn.Module):
    def __init__(self, num_codebooks: int, vocab_size: int, hidden_size: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_codebooks * vocab_size, hidden_size))
        self.num_codebooks = int(num_codebooks)
        self.vocab_size = int(vocab_size)

    def forward(self, codes_l_n: torch.Tensor) -> torch.Tensor:
        offsets = torch.arange(
            self.num_codebooks,
            device=codes_l_n.device,
            dtype=codes_l_n.dtype,
        ) * self.vocab_size
        return F.embedding(codes_l_n + offsets, self.weight).sum(dim=-2)


class HiggsFusedMultiTextHead(nn.Module):
    def __init__(self, num_codebooks: int, vocab_size: int, hidden_size: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_codebooks * vocab_size, hidden_size))
        self.num_codebooks = int(num_codebooks)
        self.vocab_size = int(vocab_size)

    def generate(self, hidden_l_d: torch.Tensor) -> torch.Tensor:
        logits = F.linear(hidden_l_d, self.weight)
        return logits.reshape(hidden_l_d.shape[0], self.num_codebooks, self.vocab_size)


@dataclass
class HiggsSamplerState:
    num_codebooks: int
    delay_count: int = 0
    eoc_countdown: int | None = None
    generation_done: bool = False
    last_codes: torch.Tensor | None = None


def _sample_independent(
    logits_n_v: torch.Tensor,
    *,
    temperature: float,
    top_p: float | None,
    top_k: int | None,
) -> torch.Tensor:
    if temperature <= 1e-5:
        return logits_n_v.argmax(dim=-1)
    logits = logits_n_v / float(temperature)
    if top_k is not None and top_k > 0:
        k = min(int(top_k), logits.size(-1))
        kth = logits.topk(k, dim=-1).values[:, -1:]
        logits = torch.where(logits < kth, float("-inf"), logits)
    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cum_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        remove = cum_probs > float(top_p)
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        scatter = torch.zeros_like(remove)
        scatter.scatter_(-1, sorted_indices, remove)
        logits = torch.where(scatter, float("-inf"), logits)
    probs = logits.softmax(dim=-1)
    return probs.multinomial(num_samples=1).squeeze(-1)


def sampler_step(
    logits_n_v: torch.Tensor,
    state: HiggsSamplerState,
    *,
    temperature: float,
    top_p: float | None,
    top_k: int | None,
) -> torch.Tensor:
    n = state.num_codebooks
    if state.generation_done:
        return torch.full((n,), STOP_CODE, dtype=torch.long, device=logits_n_v.device)

    codes_n = _sample_independent(
        logits_n_v,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    ).to(torch.long)

    if state.delay_count < n:
        next_cb = state.delay_count + 1
        if next_cb < n:
            codes_n[next_cb:] = BOC_ID
        state.delay_count += 1
    elif state.eoc_countdown is not None:
        state.eoc_countdown -= 1
        if state.eoc_countdown <= 0:
            state.generation_done = True
    elif int(codes_n[0].item()) == EOC_ID:
        state.eoc_countdown = n - 2 if n > 2 else 0
        if n <= 2:
            state.generation_done = True

    if not state.generation_done:
        state.last_codes = codes_n.clone()
    return codes_n


def apply_delay_pattern(codes_t_n: torch.Tensor) -> torch.Tensor:
    if codes_t_n.ndim != 2:
        raise ValueError(f"codes must be [T, N], got {tuple(codes_t_n.shape)}")
    t, n = codes_t_n.shape
    out = torch.full((t + n - 1, n), EOC_ID, device=codes_t_n.device, dtype=codes_t_n.dtype)
    t_idx = torch.arange(t + n - 1, device=codes_t_n.device)
    for c in range(n):
        out[t_idx < c, c] = BOC_ID
        out[c : c + t, c] = codes_t_n[:, c]
    return out


def reverse_delay_pattern(delayed_l_n: torch.Tensor) -> torch.Tensor:
    if delayed_l_n.ndim != 2:
        raise ValueError(f"delayed codes must be [L, N], got {tuple(delayed_l_n.shape)}")
    l, n = delayed_l_n.shape
    t = l - (n - 1)
    if t <= 0:
        raise ValueError(f"Need at least {n} delayed rows, got {l}.")
    out = torch.empty((t, n), device=delayed_l_n.device, dtype=delayed_l_n.dtype)
    for c in range(n):
        out[:, c] = delayed_l_n[c : c + t, c]
    return out


class HiggsTokenizerAdapter:
    def __init__(self, tokenizer: Any) -> None:
        self._tok = tokenizer
        vocab = dict(tokenizer.get_added_vocab())
        missing = [token for token in ("<|tts|>", "<|ref_audio|>", "<|text|>", "<|audio|>") if token not in vocab]
        if missing:
            raise ValueError(f"Tokenizer is missing Higgs TTS specials: {missing}")
        self.tts_id = vocab["<|tts|>"]
        self.ref_audio_id = vocab["<|ref_audio|>"]
        self.text_id = vocab["<|text|>"]
        self.audio_id = vocab["<|audio|>"]
        self.ref_text_id = vocab.get("<|ref_text|>")

    def build_prompt(self, prompt_text: str, *, num_ref_tokens: int = 0, reference_text: str | None = None) -> list[int]:
        ids = [self.tts_id]
        if reference_text and num_ref_tokens > 0 and self.ref_text_id is not None:
            ids.append(self.ref_text_id)
            ids.extend(self._tok.encode(reference_text, add_special_tokens=False))
        if num_ref_tokens > 0:
            ids.append(self.ref_audio_id)
            ids.extend([AUDIO_PLACEHOLDER_ID] * int(num_ref_tokens))
        ids.append(self.text_id)
        ids.extend(self._tok.encode(prompt_text, add_special_tokens=False))
        ids.append(self.audio_id)
        return ids


def load_tokenizer(model_dir: Path) -> HiggsTokenizerAdapter:
    raw = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    return HiggsTokenizerAdapter(PreTrainedTokenizerFast(tokenizer_object=raw))


class HiggsNativeTTS(nn.Module):
    def __init__(self, config: dict[str, Any], torch_dtype: torch.dtype, attn_implementation: str | None):
        super().__init__()
        text_config_dict = dict(config["text_config"])
        text_config = Qwen3Config(**text_config_dict)
        if attn_implementation is not None:
            text_config._attn_implementation = attn_implementation
        self.backbone = Qwen3ForCausalLM(text_config)
        self.backbone.to(dtype=torch_dtype)

        enc_cfg = dict(config.get("audio_encoder_config") or {})
        if enc_cfg.get("encoder_type", "discrete") != "discrete":
            raise NotImplementedError("Only the Higgs v3 discrete TTS path is supported.")
        self.num_codebooks = int(enc_cfg.get("num_codebooks", 8))
        self.codebook_vocab_size = int(enc_cfg.get("vocab_size", 1026))
        hidden_size = int(enc_cfg.get("out_dim", text_config.hidden_size))
        self.modality_embedding = HiggsFusedMultiTextEmbedding(
            self.num_codebooks,
            self.codebook_vocab_size,
            hidden_size,
        ).to(dtype=torch_dtype)
        self.modality_head = HiggsFusedMultiTextHead(
            self.num_codebooks,
            self.codebook_vocab_size,
            hidden_size,
        ).to(dtype=torch_dtype)
        self.tie_modality = bool(enc_cfg.get("tie_word_embeddings", True))
        self.retie_weights()

    def retie_weights(self) -> None:
        if self.tie_modality:
            self.modality_head.weight = self.modality_embedding.weight
        try:
            self.backbone.tie_weights()
        except Exception:
            pass

    def _prompt_embeds(
        self,
        prompt_ids: list[int],
        reference_codes_delayed: torch.Tensor | None,
        device: torch.device,
    ) -> torch.Tensor:
        ids = torch.tensor(prompt_ids, dtype=torch.long, device=device)
        mask = ids == AUDIO_PLACEHOLDER_ID
        safe_ids = ids.clamp_min(0).view(1, -1)
        embeds = self.backbone.model.embed_tokens(safe_ids)
        if mask.any():
            if reference_codes_delayed is None:
                raise ValueError("Prompt contains audio placeholders but no reference codes were supplied.")
            ref_codes = reference_codes_delayed.to(device=device, dtype=torch.long)
            if int(mask.sum().item()) != ref_codes.shape[0]:
                raise ValueError(
                    f"Reference placeholder count ({int(mask.sum().item())}) does not match delayed codes ({ref_codes.shape[0]})."
                )
            embeds[0, mask] = self.modality_embedding(ref_codes).to(embeds.dtype)
        return embeds

    @torch.no_grad()
    def generate_codes(
        self,
        prompt_ids: list[int],
        reference_codes_delayed: torch.Tensor | None,
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float | None,
        top_k: int | None,
        progress_callback=None,
    ) -> torch.Tensor:
        device = torch.device(getattr(self, "_higgs_runtime_device", next(self.parameters()).device))
        prompt_embeds = self._prompt_embeds(prompt_ids, reference_codes_delayed, device)
        out = self.backbone.model(inputs_embeds=prompt_embeds, use_cache=True)
        past = out.past_key_values
        hidden = out.last_hidden_state[:, -1, :]
        state = HiggsSamplerState(num_codebooks=self.num_codebooks)
        rows: list[torch.Tensor] = []
        terminal_progress = (
            tqdm(
                total=int(max_new_tokens),
                desc="Higgs v3 audio tokens",
                unit="tok",
                ascii=False,
                dynamic_ncols=True,
                leave=True,
            )
            if tqdm is not None
            else None
        )

        try:
            for i in range(int(max_new_tokens)):
                logits = self.modality_head.generate(hidden)[0].to(torch.float32)
                codes = sampler_step(
                    logits,
                    state,
                    temperature=float(temperature),
                    top_p=top_p,
                    top_k=top_k,
                )
                if int(codes[0].item()) != STOP_CODE:
                    rows.append(codes.detach().to("cpu", torch.long))
                if terminal_progress is not None:
                    terminal_progress.update(1)
                if progress_callback is not None and (i == 0 or (i + 1) % 8 == 0 or state.generation_done):
                    progress_callback(i + 1, int(max_new_tokens))
                if tqdm is None and (i == 0 or (i + 1) % 128 == 0 or state.generation_done):
                    logger.info("Higgs v3 audio tokens %d/%d", i + 1, int(max_new_tokens))
                if state.generation_done:
                    if terminal_progress is not None and terminal_progress.n < terminal_progress.total:
                        terminal_progress.total = terminal_progress.n
                    break
                if state.last_codes is None:
                    break
                next_embed = self.modality_embedding(state.last_codes.view(1, -1)).view(1, 1, -1)
                out = self.backbone.model(
                    inputs_embeds=next_embed.to(device=device, dtype=prompt_embeds.dtype),
                    past_key_values=past,
                    use_cache=True,
                )
                past = out.past_key_values
                hidden = out.last_hidden_state[:, -1, :]
        finally:
            if terminal_progress is not None:
                terminal_progress.close()

        if len(rows) < self.num_codebooks:
            raise RuntimeError(f"Higgs generated too few audio token rows ({len(rows)}).")
        if not state.generation_done and len(rows) >= int(max_new_tokens):
            logger.warning(
                "Higgs v3 reached max_new_tokens=%d before the model emitted a stop token. "
                "If text is missing, enable longform chunking or raise max_new_tokens.",
                int(max_new_tokens),
            )
        return torch.stack(rows, dim=0)


def _codec_config() -> HiggsAudioV2TokenizerConfig:
    if CODEC_CONFIG_PATH.is_file():
        cfg = json.loads(CODEC_CONFIG_PATH.read_text(encoding="utf-8"))
        for key in ("architectures", "torch_dtype", "transformers_version"):
            cfg.pop(key, None)
        return HiggsAudioV2TokenizerConfig(**cfg)
    return HiggsAudioV2TokenizerConfig()


def _codec_state_dict(model_dir: Path) -> dict[str, torch.Tensor]:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
        shards: dict[str, list[str]] = {}
        for full_name, shard in weight_map.items():
            if full_name.startswith(CODEC_PREFIX):
                shards.setdefault(shard, []).append(full_name)
    else:
        shards = {"model.safetensors": []}

    state: dict[str, torch.Tensor] = {}
    for shard, names in shards.items():
        shard_path = model_dir / shard
        with safe_open(str(shard_path), framework="pt", device="cpu") as f:
            keys = names or [key for key in f.keys() if key.startswith(CODEC_PREFIX)]
            for full_name in keys:
                state[full_name[len(CODEC_PREFIX) :]] = f.get_tensor(full_name)
    if not state:
        raise FileNotFoundError(f"No bundled Higgs audio codec weights were found in {model_dir}.")
    return state


class HiggsAudioCodec:
    def __init__(self, model: HiggsAudioV2TokenizerModel, device: torch.device) -> None:
        self.model = model
        self.device = torch.device(device)
        self.dtype = next(model.parameters()).dtype

    @classmethod
    def from_pretrained(
        cls,
        model_dir: Path,
        *,
        device: torch.device,
        dtype: torch.dtype,
        runtime_device: torch.device | None = None,
    ) -> "HiggsAudioCodec":
        model = HiggsAudioV2TokenizerModel(_codec_config()).to(dtype=dtype).eval()
        state = _codec_state_dict(model_dir)
        missing, _unexpected = model.load_state_dict(state, strict=False)
        if len(missing) > len(state) // 2:
            raise RuntimeError(
                f"Codec weight load is too sparse: {len(missing)} missing / {len(state)} loaded."
            )
        model = model.to(device=device)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return cls(model, runtime_device or device)

    @torch.no_grad()
    def encode_reference(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        wav = waveform.detach().float().cpu()
        if wav.ndim == 1:
            wav = wav.view(1, 1, -1)
        elif wav.ndim == 2:
            wav = wav[:1].unsqueeze(0)
        elif wav.ndim == 3:
            wav = wav[:, :1, :]
        else:
            raise ValueError(f"Unsupported reference audio shape: {tuple(wav.shape)}")
        if int(sample_rate) != SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, int(sample_rate), SAMPLE_RATE)
        if wav.shape[-1] < SAMPLE_RATE:
            wav = F.pad(wav, (0, SAMPLE_RATE - wav.shape[-1]))
        wav = wav.to(device=self.device, dtype=self.dtype)
        codes_b_n_t = self.model.encode(wav).audio_codes
        return codes_b_n_t.squeeze(0).transpose(0, 1).to(torch.long).cpu()

    @torch.no_grad()
    def decode(self, codes_t_n: torch.Tensor) -> torch.Tensor:
        codes_b_n_t = codes_t_n.transpose(0, 1).unsqueeze(0).to(device=self.device, dtype=torch.long)
        return self.model.decode(codes_b_n_t).audio_values.squeeze(0).squeeze(0).detach().float().cpu()


class _ComfyLinear(nn.Linear):
    comfy_cast_weights = True
    weight_function = []
    bias_function = []

    def forward(self, x):
        if not hasattr(self, "_v") and self.weight.device == x.device:
            return F.linear(x, self.weight, self.bias)
        weight, bias, stream = cast_bias_weight(self, x, offloadable=True)
        try:
            return F.linear(x, weight, bias)
        finally:
            uncast_bias_weight(self, weight, bias, stream)


class _ComfyEmbedding(nn.Embedding):
    comfy_cast_weights = True
    weight_function = []
    bias_function = []
    bias = None

    def _weight_dtype(self):
        return getattr(self, "weight_comfy_model_dtype", None) or self.weight.dtype

    def forward(self, input):
        if not hasattr(self, "_v") and self.weight.device == input.device:
            return F.embedding(
                input, self.weight, self.padding_idx, self.max_norm,
                self.norm_type, self.scale_grad_by_freq, self.sparse)
        weight, bias, stream = cast_bias_weight(
            self,
            dtype=self._weight_dtype(),
            device=input.device,
            offloadable=True,
        )
        try:
            return F.embedding(
                input, weight, self.padding_idx, self.max_norm,
                self.norm_type, self.scale_grad_by_freq, self.sparse)
        finally:
            uncast_bias_weight(self, weight, bias, stream)


class _ComfyConv1d(nn.Conv1d):
    comfy_cast_weights = True
    weight_function = []
    bias_function = []

    def forward(self, input):
        if (
            not hasattr(self, "_v")
            and self.weight.device == input.device
            and (self.bias is None or self.bias.device == input.device)
        ):
            return self._conv_forward(input, self.weight, self.bias)
        weight, bias, stream = cast_bias_weight(self, input, offloadable=True)
        try:
            return self._conv_forward(input, weight, bias)
        finally:
            uncast_bias_weight(self, weight, bias, stream)


class _ComfyConvTranspose1d(nn.ConvTranspose1d):
    comfy_cast_weights = True
    weight_function = []
    bias_function = []

    def forward(self, input, output_size=None):
        num_spatial_dims = 1
        output_padding = self._output_padding(
            input,
            output_size,
            self.stride,
            self.padding,
            self.kernel_size,
            num_spatial_dims,
            self.dilation,
        )
        if (
            not hasattr(self, "_v")
            and self.weight.device == input.device
            and (self.bias is None or self.bias.device == input.device)
        ):
            return F.conv_transpose1d(
                input,
                self.weight,
                self.bias,
                self.stride,
                self.padding,
                output_padding,
                self.groups,
                self.dilation,
            )
        weight, bias, stream = cast_bias_weight(self, input, offloadable=True)
        try:
            return F.conv_transpose1d(
                input,
                weight,
                bias,
                self.stride,
                self.padding,
                output_padding,
                self.groups,
                self.dilation,
            )
        finally:
            uncast_bias_weight(self, weight, bias, stream)


class _ComfyHiggsFusedMultiTextEmbedding(HiggsFusedMultiTextEmbedding):
    comfy_cast_weights = True
    weight_function = []
    bias_function = []
    bias = None

    def _weight_dtype(self):
        return getattr(self, "weight_comfy_model_dtype", None) or self.weight.dtype

    def forward(self, codes_l_n: torch.Tensor) -> torch.Tensor:
        offsets = torch.arange(
            self.num_codebooks,
            device=codes_l_n.device,
            dtype=codes_l_n.dtype,
        ) * self.vocab_size
        if not hasattr(self, "_v") and self.weight.device == codes_l_n.device:
            weight = self.weight
            stream = None
            bias = None
        else:
            weight, bias, stream = cast_bias_weight(
                self,
                dtype=self._weight_dtype(),
                device=codes_l_n.device,
                offloadable=True,
            )
        try:
            return F.embedding(codes_l_n + offsets, weight).sum(dim=-2)
        finally:
            uncast_bias_weight(self, weight, bias, stream)


class _ComfyHiggsFusedMultiTextHead(HiggsFusedMultiTextHead):
    comfy_cast_weights = True
    weight_function = []
    bias_function = []
    bias = None

    def generate(self, hidden_l_d: torch.Tensor) -> torch.Tensor:
        if not hasattr(self, "_v") and self.weight.device == hidden_l_d.device:
            weight = self.weight
            stream = None
            bias = None
        else:
            weight, bias, stream = cast_bias_weight(self, hidden_l_d, offloadable=True)
        try:
            logits = F.linear(hidden_l_d, weight)
            return logits.reshape(hidden_l_d.shape[0], self.num_codebooks, self.vocab_size)
        finally:
            uncast_bias_weight(self, weight, bias, stream)


def _comfy_qwen3_rmsnorm_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    if not hasattr(self, "_v") and self.weight.device == hidden_states.device:
        weight = self.weight
        stream = None
        bias = None
    else:
        weight, bias, stream = cast_bias_weight(self, hidden_states, offloadable=True)
    try:
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return weight * hidden_states.to(input_dtype)
    finally:
        uncast_bias_weight(self, weight, bias, stream)


def _patch_qwen3_rmsnorm(module: nn.Module) -> None:
    if getattr(module, "_higgs_comfy_cast_rmsnorm", False):
        return
    module.bias = None
    module.comfy_cast_weights = True
    module.weight_function = []
    module.bias_function = []
    module.forward = _comfy_qwen3_rmsnorm_forward.__get__(module, module.__class__)
    module._higgs_comfy_cast_rmsnorm = True


def set_runtime_dtype(module: nn.Module, dtype: torch.dtype) -> None:
    """Tag floating tensors with the dtype Comfy/AIMDO should materialize."""
    for sub in module.modules():
        for name, value in sub.named_parameters(recurse=False):
            if value is not None and value.is_floating_point():
                setattr(sub, f"{name}_comfy_model_dtype", dtype)
        for name, value in sub.named_buffers(recurse=False):
            if value is not None and value.is_floating_point():
                setattr(sub, f"{name}_comfy_model_dtype", dtype)


def convert_modules_for_comfy(model: nn.Module) -> None:
    """Patch castable modules in-place so DynamicVRAM can page their weights."""
    for module in model.modules():
        if isinstance(module, (_ComfyLinear, _ComfyEmbedding, _ComfyConv1d, _ComfyConvTranspose1d)):
            continue
        if isinstance(module, nn.Linear):
            module.__class__ = _ComfyLinear
        elif type(module) is nn.Embedding:
            module.__class__ = _ComfyEmbedding
        elif isinstance(module, nn.Conv1d) and not hasattr(module, "parametrizations"):
            module.__class__ = _ComfyConv1d
        elif isinstance(module, nn.ConvTranspose1d) and not hasattr(module, "parametrizations"):
            module.__class__ = _ComfyConvTranspose1d
        elif isinstance(module, HiggsFusedMultiTextEmbedding):
            module.__class__ = _ComfyHiggsFusedMultiTextEmbedding
        elif isinstance(module, HiggsFusedMultiTextHead):
            module.__class__ = _ComfyHiggsFusedMultiTextHead
        elif module.__class__.__name__ == "Qwen3RMSNorm" and hasattr(module, "variance_epsilon"):
            _patch_qwen3_rmsnorm(module)


def map_higgs_weight_name(name: str) -> str | None:
    if name.startswith(CODEC_PREFIX):
        return None
    prefix_map = {
        "tied.embedding.text_embedding.": "backbone.model.embed_tokens.",
        "body.layers.": "backbone.model.layers.",
        "body.norm.": "backbone.model.norm.",
        "tied.head.text_head.": "backbone.lm_head.",
        "tied.embedding.modality_embeddings.0.embedding.": "modality_embedding.",
        "tied.head.modality_heads.0.": "modality_head.",
    }
    for source, target in prefix_map.items():
        if name.startswith(source):
            return target + name[len(source) :]
    return name


def iter_safetensor_items(model_dir: Path) -> Iterable[tuple[str, torch.Tensor]]:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
        shards = sorted(set(weight_map.values()))
    else:
        shards = ["model.safetensors"]
    for shard in shards:
        shard_path = model_dir / shard
        with safe_open(str(shard_path), framework="pt", device="cpu") as f:
            for key in f.keys():
                yield key, f.get_tensor(key)


def _set_tensor(module: nn.Module, name: str, tensor: torch.Tensor, device: torch.device, dtype: torch.dtype) -> None:
    if tensor.is_floating_point():
        tensor = tensor.to(dtype=dtype)
    try:
        from accelerate.utils.modeling import set_module_tensor_to_device

        set_module_tensor_to_device(module, name, device=device, value=tensor.contiguous())
        return
    except Exception:
        pass

    target = dict(module.named_parameters(remove_duplicate=False)).get(name)
    if target is None:
        target = dict(module.named_buffers(remove_duplicate=False)).get(name)
    if target is None:
        raise KeyError(name)
    if target.shape != tensor.shape:
        raise ValueError(f"Shape mismatch for {name}: expected {tuple(target.shape)}, got {tuple(tensor.shape)}")
    target.data = tensor.to(device=device, dtype=target.dtype if target.is_floating_point() else target.dtype).contiguous()


def build_native_model(config: dict[str, Any], dtype: torch.dtype, attention_impl: str | None) -> HiggsNativeTTS:
    try:
        from accelerate import init_empty_weights

        with init_empty_weights():
            return HiggsNativeTTS(config, dtype, attention_impl)
    except Exception:
        return HiggsNativeTTS(config, dtype, attention_impl)


def load_native_weights(model: HiggsNativeTTS, model_dir: Path, device: torch.device, dtype: torch.dtype) -> None:
    param_names = set(dict(model.named_parameters(remove_duplicate=False)))
    loaded: set[str] = set()
    skipped = 0
    for source_name, tensor in iter_safetensor_items(model_dir):
        mapped = map_higgs_weight_name(source_name)
        if mapped is None:
            skipped += 1
            continue
        if mapped not in param_names:
            if mapped in {"backbone.lm_head.weight", "modality_head.weight"}:
                continue
            logger.debug("Skipping unmapped Higgs tensor: %s -> %s", source_name, mapped)
            skipped += 1
            continue
        _set_tensor(model, mapped, tensor, device, dtype)
        loaded.add(mapped)
    model.retie_weights()
    model.to(device=device)
    model.eval()
    missing = [name for name in param_names if "lm_head.weight" not in name and "modality_head.weight" not in name and name not in loaded]
    if missing:
        raise RuntimeError(f"Higgs native load missed {len(missing)} parameter(s), first: {missing[:8]}")
    logger.info("Loaded Higgs native weights: %d tensors, skipped %d codec/tied tensors.", len(loaded), skipped)


def read_config(model_dir: Path) -> dict[str, Any]:
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing Higgs config.json in {model_dir}.")
    return json.loads(config_path.read_text(encoding="utf-8"))


def comfy_audio_to_tensor(audio: dict) -> tuple[torch.Tensor, int]:
    waveform = audio["waveform"]
    sample_rate = int(audio["sample_rate"])
    if not isinstance(waveform, torch.Tensor):
        waveform = torch.as_tensor(waveform)
    wav = waveform[0].detach().float().cpu()
    if wav.ndim == 2 and wav.shape[0] > 1:
        wav = wav.mean(dim=0)
    elif wav.ndim == 2:
        wav = wav.squeeze(0)
    return wav.contiguous(), sample_rate


def tensor_audio_to_comfy(audio: torch.Tensor, sample_rate: int = SAMPLE_RATE) -> dict:
    audio = audio.detach().float().cpu().clamp(-1.0, 1.0)
    return {"waveform": audio.view(1, 1, -1).contiguous(), "sample_rate": int(sample_rate)}


def trim_silence_edges(audio: torch.Tensor, sample_rate: int, threshold_db: float = -42.0) -> torch.Tensor:
    if audio.numel() == 0:
        return audio
    threshold = 10 ** (threshold_db / 20.0)
    frame = max(1, int(sample_rate * 0.01))
    padded = F.pad(audio.abs(), (0, (frame - audio.numel() % frame) % frame))
    rms = padded.view(-1, frame).pow(2).mean(dim=1).sqrt()
    active = torch.nonzero(rms > threshold, as_tuple=False).flatten()
    if active.numel() == 0:
        return audio
    start = max(0, int(active[0].item()) * frame)
    end = min(audio.numel(), (int(active[-1].item()) + 1) * frame)
    pad = int(sample_rate * 0.05)
    return audio[max(0, start - pad) : min(audio.numel(), end + pad)].contiguous()


def load_audio_path(path: str) -> tuple[torch.Tensor, int]:
    value = path.strip()
    if not value:
        raise ValueError("reference_audio_path is empty.")
    audio_path = Path(value).expanduser()
    if not audio_path.is_file():
        raise FileNotFoundError(f"Reference audio not found: {audio_path}")
    wav, sample_rate = torchaudio.load(str(audio_path))
    if wav.ndim == 2 and wav.shape[0] > 1:
        wav = wav.mean(dim=0)
    else:
        wav = wav.squeeze(0)
    return wav.detach().float().cpu().contiguous(), int(sample_rate)


@contextmanager
def attention_runtime(attention: str):
    if attention != "sageattention":
        yield
        return
    import torch.nn.functional as torch_f
    from sageattention import sageattn

    original_sdpa = torch_f.scaled_dot_product_attention

    def sage_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kwargs):
        if (
            attn_mask is not None
            or dropout_p not in (0, 0.0)
            or query.device.type != "cuda"
            or query.dtype not in (torch.float16, torch.bfloat16)
        ):
            return original_sdpa(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, **kwargs)
        try:
            output = sageattn(query, key, value, tensor_layout="HND", is_causal=is_causal, sm_scale=scale)
            return output[0] if isinstance(output, tuple) else output
        except Exception:
            return original_sdpa(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, **kwargs)

    torch_f.scaled_dot_product_attention = sage_sdpa
    try:
        yield
    finally:
        torch_f.scaled_dot_product_attention = original_sdpa


def manual_seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.manual_seed(seed)


def generate_higgs_audio(
    bundle: Any,
    *,
    text: str,
    reference_audio: dict | None,
    reference_audio_path: str,
    reference_text: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int,
    trim_reference_audio: bool,
    silence_threshold_db: float,
    max_reference_seconds: float,
    progress_callback=None,
) -> dict:
    if not text.strip():
        raise ValueError("Text cannot be empty.")
    try:
        from .loader import resume_bundle_to_device

        resume_bundle_to_device(bundle)
    except Exception:
        pass
    if seed:
        manual_seed_all(int(seed))

    ref_delayed = None
    if reference_audio is not None or reference_audio_path.strip():
        if reference_audio is not None:
            wav, sr = comfy_audio_to_tensor(reference_audio)
        else:
            wav, sr = load_audio_path(reference_audio_path)
        if trim_reference_audio:
            wav = trim_silence_edges(wav, sr, silence_threshold_db)
        max_samples = int(max_reference_seconds * sr)
        if max_samples > 0 and wav.numel() > max_samples:
            wav = wav[:max_samples].contiguous()
        raw_codes = bundle.codec.encode_reference(wav, sr)
        ref_delayed = apply_delay_pattern(raw_codes)

    prompt_ids = bundle.tokenizer.build_prompt(
        text.strip(),
        num_ref_tokens=0 if ref_delayed is None else int(ref_delayed.shape[0]),
        reference_text=reference_text.strip() or None,
    )
    with torch.inference_mode(), attention_runtime(bundle.attention):
        delayed = bundle.model.generate_codes(
            prompt_ids,
            ref_delayed,
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            top_p=None if top_p <= 0 or top_p >= 1 else float(top_p),
            top_k=None if top_k <= 0 else int(top_k),
            progress_callback=progress_callback,
        )
        raw = reverse_delay_pattern(delayed)
        audio = bundle.codec.decode(raw)
    if not torch.isfinite(audio).all():
        raise RuntimeError("Higgs generated non-finite audio samples.")
    gc.collect()
    return tensor_audio_to_comfy(audio, SAMPLE_RATE)
