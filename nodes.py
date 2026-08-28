"""ComfyUI node definitions for Higgs Audio v3 TTS."""

from __future__ import annotations

import logging
import re

import torch

from .loader import ATTENTION_OPTIONS, DEVICE_OPTIONS, DTYPE_OPTIONS, bundle_state_token, get_model_choices, load_higgs_bundle, reload_dead_bundle
from .native import generate_higgs_audio
from .whisper import HiggsV3WhisperTranscribe

logger = logging.getLogger("Higgs_v3-TTS-ComfyUI")

try:
    from comfy.utils import ProgressBar
except Exception:
    ProgressBar = None

try:
    from comfy_api.latest import IO

    _HAS_DYNAMIC_COMBO = True
except Exception:
    IO = None
    _HAS_DYNAMIC_COMBO = False


MAX_SPEAKERS = 6
PROGRESS_UNITS_PER_SEGMENT = 1000
DELIVERY_PROSODY_VALUES = {
    "speed_very_slow",
    "speed_slow",
    "speed_fast",
    "speed_very_fast",
    "pitch_low",
    "pitch_high",
    "expressive_high",
    "expressive_low",
}


def _delivery_state_from_prefix(prefix: str) -> dict[str, str]:
    state: dict[str, str] = {}
    for kind, value in re.findall(r"<\|(emotion|style|prosody):([^|]+)\|>", prefix):
        if kind == "prosody" and value not in DELIVERY_PROSODY_VALUES:
            continue
        state[kind] = value
    return state


def _delivery_state_prefix(state: dict[str, str], skip_categories: set[str] | None = None) -> str:
    skip_categories = skip_categories or set()
    parts: list[str] = []
    if "emotion" not in skip_categories and state.get("emotion"):
        parts.append(f"<|emotion:{state['emotion']}|>")
    if "style" not in skip_categories and state.get("style"):
        parts.append(f"<|style:{state['style']}|>")
    if "prosody" not in skip_categories and state.get("prosody"):
        parts.append(f"<|prosody:{state['prosody']}|>")
    return "".join(parts)


def _update_delivery_state_from_text(state: dict[str, str], text: str) -> None:
    for kind, value in re.findall(r"<\|(emotion|style|prosody):([^|]+)\|>", text):
        if kind == "prosody" and value not in DELIVERY_PROSODY_VALUES:
            continue
        state[kind] = value


def _initial_delivery_categories(text: str) -> set[str]:
    categories: set[str] = set()
    pos = 0
    pattern = re.compile(r"\s*<\|(emotion|style|prosody):([^|]+)\|>")
    for match in pattern.finditer(text):
        if match.start() != pos:
            break
        kind, value = match.group(1), match.group(2)
        if kind != "prosody" or value in DELIVERY_PROSODY_VALUES:
            categories.add(kind)
        pos = match.end()
    return categories


def _bare_tag_start(text: str) -> int | None:
    match = re.search(r"<\|(?:emotion|style|prosody|sfx):[^|]+\|>\s*$", text)
    return None if match is None else match.start()


def _text_input() -> tuple:
    return (
        "STRING",
        {
            "multiline": True,
            "default": "Hello! This is Higgs Audio v3 running natively inside ComfyUI.",
            "tooltip": "Text to synthesize. Inline tags work anywhere, for example <|emotion:relief|>, <|prosody:pause|>, or <|sfx:laughter|>Haha at the exact moment it should happen.",
        },
    )


def _generation_controls() -> dict:
    return {
        "max_new_tokens": (
            "INT",
            {
                "default": 2048,
                "min": 32,
                "max": 8192,
                "step": 8,
                "tooltip": "Maximum audio-code tokens per single pass. 2048 is roughly 25-30 seconds; raise it or enable chunking if speech cuts off.",
            },
        ),
        "temperature": (
            "FLOAT",
            {
                "default": 1.0,
                "min": 0.0,
                "max": 2.0,
                "step": 0.05,
                "tooltip": "Sampling variety. 0 is greedy and repeatable; around 0.8-1.1 is usually natural.",
            },
        ),
        "top_p": (
            "FLOAT",
            {
                "default": 0.95,
                "min": 0.0,
                "max": 1.0,
                "step": 0.01,
                "tooltip": "Nucleus sampling cutoff. 1.0 disables it; 0.9-0.98 keeps speech expressive without wandering too much.",
            },
        ),
        "top_k": (
            "INT",
            {
                "default": 50,
                "min": 0,
                "max": 1026,
                "step": 1,
                "tooltip": "Limits each codebook sample to the top K choices. 0 disables; 50 is a steady default.",
            },
        ),
        "seed": (
            "INT",
            {
                "default": 0,
                "min": 0,
                "max": 2**31 - 1,
                "tooltip": "0 uses the current random state. A positive value is repeatable and is reused unchanged for every longform chunk.",
            },
        ),
        "longform_chunking": (
            "BOOLEAN",
            {
                "default": True,
                "tooltip": "Split long text at sentence or pause-tag boundaries. Turn this on for narration; off is one direct pass and may stop early on long text.",
            },
        ),
        "words_per_chunk": (
            "INT",
            {
                "default": 45,
                "min": 20,
                "max": 300,
                "step": 5,
                "tooltip": "Target words per chunk. Around 35-55 fits the 2048-token default better; raise with max_new_tokens for longer chunks.",
            },
        ),
        "tag_chunk": (
            "BOOLEAN",
            {
                "default": False,
                "tooltip": "Cut chunks at every <|...|> tag instead of only at sentence breaks. Oversized tag sections are still split by words_per_chunk, with the active tag re-inserted at the start of each new piece so it keeps the tone/voice.",
            },
        ),
        "pause_between_chunks": (
            "FLOAT",
            {
                "default": 0.15,
                "min": 0.0,
                "max": 2.0,
                "step": 0.05,
                "tooltip": "Seconds of silence inserted between longform chunks. Does not replace inline pause tags.",
            },
        ),
    }


def _common_generation_inputs() -> dict:
    inputs = {"text": _text_input()}
    inputs.update(_generation_controls())
    return inputs


def _is_cjk(char: str) -> bool:
    cp = ord(char)
    return (
        0x4E00 <= cp <= 0x9FFF
        or 0x3400 <= cp <= 0x4DBF
        or 0x20000 <= cp <= 0x2A6DF
        or 0x3040 <= cp <= 0x30FF
        or 0x30A0 <= cp <= 0x30FF
        or 0xAC00 <= cp <= 0xD7AF
        or 0x0E00 <= cp <= 0x0E7F
        or 0x1000 <= cp <= 0x109F
        or 0x1780 <= cp <= 0x17FF
    )


def _tag_safe_boundary(segment: str) -> int | None:
    """Return a good split point, never inside a Higgs <|...|> tag."""
    boundary = re.compile(
        r"(<\|prosody:(?:pause|long_pause)\|>|[.!?]+(?:\s|$)|[。？！\u0964\u0965\u061F\u104B\u0F0D]+)"
    )
    tag_ranges = [(m.start(), m.end()) for m in re.finditer(r"<\|[^|]*\|>", segment)]
    last_end = None
    for match in boundary.finditer(segment):
        end = match.end()
        if any(start < end < stop for start, stop in tag_ranges):
            continue
        last_end = end
    return last_end


def _chunk_by_characters(text: str, chars_per_chunk: int) -> list[str]:
    if len(text) <= chars_per_chunk:
        return [text]
    chunks: list[str] = []
    pos = 0
    while pos < len(text):
        while pos < len(text) and text[pos].isspace():
            pos += 1
        target = min(pos + chars_per_chunk, len(text))
        if target >= len(text):
            tail = text[pos:].strip()
            if tail:
                chunks.append(tail)
            break
        segment = text[pos:target]
        split = _tag_safe_boundary(segment)
        if split is None or split < max(20, chars_per_chunk // 3):
            split = target - pos
            next_tag = segment.rfind("<|", 0, split)
            next_close = segment.rfind("|>", 0, split)
            if next_tag > next_close:
                split = next_tag
            else:
                bare_tag = _bare_tag_start(segment[:split])
                if bare_tag is not None and bare_tag > 0:
                    split = bare_tag
        chunk = text[pos : pos + split].strip()
        if chunk:
            chunks.append(chunk)
        pos += max(split, 1)
    return chunks or [text]


def _smart_chunk_text(text: str, words_per_chunk: int, enabled: bool) -> list[str]:
    if not enabled or words_per_chunk <= 0:
        return [text.strip()]
    text = text.strip()
    if not text:
        return []

    cjk_count = sum(1 for ch in text if _is_cjk(ch))
    alpha_count = sum(1 for ch in text if ch.isalpha() or _is_cjk(ch))
    if alpha_count > 0 and cjk_count / alpha_count > 0.3:
        return _chunk_by_characters(text, words_per_chunk)

    words = text.split()
    if len(words) <= words_per_chunk:
        return [text]
    chunks: list[str] = []
    current: list[str] = []
    for word in words:
        current.append(word)
        if len(current) >= words_per_chunk:
            candidate = " ".join(current)
            split = _tag_safe_boundary(candidate)
            if split is not None and split >= max(20, len(candidate) // 3):
                final = candidate[:split].strip()
                rest = candidate[split:].strip()
                if final:
                    chunks.append(final)
                current = rest.split() if rest else []
            else:
                bare_tag = _bare_tag_start(candidate)
                if bare_tag is not None and bare_tag > 0:
                    final = candidate[:bare_tag].strip()
                    rest = candidate[bare_tag:].strip()
                    if final:
                        chunks.append(final)
                    current = rest.split() if rest else []
                else:
                    chunks.append(candidate.strip())
                    current = []
    if current:
        chunks.append(" ".join(current).strip())
    return [chunk for chunk in chunks if chunk]


def _split_content_by_words(content: str, words_per_chunk: int) -> list[str]:
    """Word/character split for a tag-free span of text (used by _tag_chunk_text)."""
    content = content.strip()
    if not content:
        return []
    if words_per_chunk <= 0:
        return [content]

    cjk_count = sum(1 for ch in content if _is_cjk(ch))
    alpha_count = sum(1 for ch in content if ch.isalpha() or _is_cjk(ch))
    if alpha_count > 0 and cjk_count / alpha_count > 0.3:
        return _chunk_by_characters(content, words_per_chunk)

    words = content.split()
    if len(words) <= words_per_chunk:
        return [content]

    chunks: list[str] = []
    current: list[str] = []
    for word in words:
        current.append(word)
        if len(current) >= words_per_chunk:
            candidate = " ".join(current)
            split = _tag_safe_boundary(candidate)
            if split is not None and split >= max(20, len(candidate) // 3):
                final = candidate[:split].strip()
                rest = candidate[split:].strip()
                if final:
                    chunks.append(final)
                current = rest.split() if rest else []
            else:
                chunks.append(candidate.strip())
                current = []
    if current:
        chunks.append(" ".join(current).strip())
    return [chunk for chunk in chunks if chunk]


def _tag_chunk_text(text: str, words_per_chunk: int) -> list[str]:
    """Chunk text by cutting right before every <|...|> tag.

    Any text before the first tag becomes its own leading chunk. From there,
    each tag (or run of back-to-back tags) opens a new chunk that runs until
    the next tag. If that tag's section is longer than words_per_chunk, it is
    split further on word boundaries, and the section's tag(s) are
    automatically re-inserted at the front of every extra piece so later
    pieces still inherit the tone/voice the tag set.
    """
    text = text.strip()
    if not text:
        return []

    tag_re = re.compile(r"<\|[^|]*\|>")
    matches = list(tag_re.finditer(text))
    if not matches:
        return _split_content_by_words(text, words_per_chunk) or [text]

    # Group the text into segments, one per tag (or per run of adjacent tags
    # separated only by whitespace), each holding its tag(s) plus the content
    # that follows until the next tag.
    segments: list[str] = []
    if matches[0].start() > 0:
        segments.append(text[: matches[0].start()])

    i = 0
    while i < len(matches):
        block_start = matches[i].start()
        j = i
        while j + 1 < len(matches):
            gap = text[matches[j].end() : matches[j + 1].start()]
            if gap.strip() == "":
                j += 1
            else:
                break
        seg_end = matches[j + 1].start() if j + 1 < len(matches) else len(text)
        segments.append(text[block_start:seg_end])
        i = j + 1

    leading_tags_re = re.compile(r"^(?:\s*<\|[^|]*\|>)+")
    chunks: list[str] = []
    for segment in segments:
        leading = leading_tags_re.match(segment)
        if not leading:
            # Text before the very first tag in the whole string.
            chunks.extend(_split_content_by_words(segment, words_per_chunk))
            continue

        tag_prefix = segment[: leading.end()].strip()
        content = segment[leading.end() :].strip()

        if not content:
            # A tag with nothing after it (e.g. a trailing sfx tag); keep it
            # with whatever came before instead of emitting an empty chunk.
            if chunks:
                chunks[-1] = f"{chunks[-1]} {tag_prefix}".strip()
            else:
                chunks.append(tag_prefix)
            continue

        for piece in _split_content_by_words(content, words_per_chunk):
            chunks.append(f"{tag_prefix}{piece}")

    return [chunk for chunk in chunks if chunk.strip()]


def _concat_audio_segments(segments: list[dict], pause_seconds: float) -> dict:
    if not segments:
        raise RuntimeError("No audio segments were generated.")
    sample_rate = int(segments[0]["sample_rate"])
    parts: list[torch.Tensor] = []
    pause_samples = int(max(0.0, float(pause_seconds)) * sample_rate)
    silence = None
    if pause_samples > 0:
        silence = torch.zeros((1, 1, pause_samples), dtype=torch.float32)
    for segment in segments:
        if int(segment["sample_rate"]) != sample_rate:
            raise RuntimeError("Generated chunks have mismatched sample rates.")
        if parts and silence is not None:
            parts.append(silence)
        waveform = segment["waveform"]
        if not isinstance(waveform, torch.Tensor):
            waveform = torch.as_tensor(waveform)
        parts.append(waveform.detach().float().cpu())
    return {"waveform": torch.cat(parts, dim=-1).contiguous(), "sample_rate": sample_rate}


def _ensure_bundle_is_loaded(higgs_model):
    """Reload the bundle in place if it was unloaded elsewhere in the graph
    (e.g. a Clear VRAM node) since Load Model produced it, instead of just
    erroring out. This does not depend on ComfyUI re-running Load Model."""
    if higgs_model is None:
        raise RuntimeError("No Higgs v3 model connected. Add a Higgs v3 Load Model node before this one.")
    if getattr(higgs_model, "codec", None) is None or getattr(higgs_model, "model", None) is None:
        reload_dead_bundle(higgs_model)

def _generate_chunked_audio(
    higgs_model,
    *,
    text: str,
    control_prefix: str = "",
    reference_audio: dict | None,
    reference_text: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int,
    longform_chunking: bool,
    words_per_chunk: int,
    tag_chunk: bool = False,
    pause_between_chunks: float,
    progress_callback=None,
    use_first_chunk_as_reference: bool = False,
) -> dict:
    _ensure_bundle_is_loaded(higgs_model)
    if not bool(longform_chunking):
        prompt_text = control_prefix + text
        logger.info("Higgs v3 generating single pass: %s", text[:90])

        def update_single_pass(current: int, total: int) -> None:
            if progress_callback is None:
                return
            fraction = min(1.0, max(0.0, float(current) / max(1, int(total))))
            progress_callback(round(fraction * PROGRESS_UNITS_PER_SEGMENT), PROGRESS_UNITS_PER_SEGMENT)

        audio = generate_higgs_audio(
            higgs_model,
            text=prompt_text,
            reference_audio=reference_audio,
            reference_audio_path="",
            reference_text=reference_text,
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=int(top_k),
            seed=int(seed),
            trim_reference_audio=True,
            silence_threshold_db=-42.0,
            max_reference_seconds=100.0,
            progress_callback=update_single_pass,
        )
        if progress_callback is not None:
            progress_callback(PROGRESS_UNITS_PER_SEGMENT, PROGRESS_UNITS_PER_SEGMENT)
        return audio

    if bool(tag_chunk):
        chunks = _tag_chunk_text(text, int(words_per_chunk))
    else:
        chunks = _smart_chunk_text(text, int(words_per_chunk), bool(longform_chunking))
    if not chunks:
        raise ValueError("Text cannot be empty.")
    if len(chunks) > 1:
        logger.info(
            "Higgs v3 longform chunking: %d chunks, target=%d words/chars, tag_chunk=%s.",
            len(chunks),
            int(words_per_chunk),
            bool(tag_chunk),
        )
    segments: list[dict] = []
    delivery_state = _delivery_state_from_prefix(control_prefix)
    active_reference_audio = reference_audio
    active_reference_text = reference_text
    progress_total = len(chunks) * PROGRESS_UNITS_PER_SEGMENT
    for index, chunk in enumerate(chunks):
        local_seed = int(seed) if seed else 0
        skip_categories = _initial_delivery_categories(chunk)
        if bool(tag_chunk):
            # tag_chunk already carries the right <|...|> tag(s) in front of
            # every chunk it produces, so only the caller-provided prefix (if
            # any) is added, and only to the very first chunk.
            active_prefix = control_prefix if index == 0 else ""
        elif index == 0 and control_prefix:
            active_prefix = control_prefix
        else:
            # Strong emotions can overpower clone conditioning when automatically
            # inserted at the start of every later chunk, so emotions stay local.
            active_prefix = _delivery_state_prefix(delivery_state, skip_categories | {"emotion"})
        prompt_text = active_prefix + chunk
        logger.info("Higgs v3 generating chunk %d/%d: %s", index + 1, len(chunks), chunk[:90])

        def update_chunk(current: int, total: int, chunk_index: int = index) -> None:
            if progress_callback is None:
                return
            fraction = min(1.0, max(0.0, float(current) / max(1, int(total))))
            value = chunk_index * PROGRESS_UNITS_PER_SEGMENT + round(
                fraction * PROGRESS_UNITS_PER_SEGMENT
            )
            progress_callback(value, progress_total)

        segment = generate_higgs_audio(
            higgs_model,
            text=prompt_text,
            reference_audio=active_reference_audio,
            reference_audio_path="",
            reference_text=active_reference_text,
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=int(top_k),
            seed=local_seed,
            trim_reference_audio=True,
            silence_threshold_db=-42.0,
            max_reference_seconds=100.0,
            progress_callback=update_chunk,
        )
        segments.append(segment)
        if (
            use_first_chunk_as_reference
            and reference_audio is None
            and len(chunks) > 1
            and index == 0
        ):
            active_reference_audio = segment
            active_reference_text = prompt_text
            logger.info("Higgs v3 longform voice anchor: using chunk 1 as reference for later chunks.")
        _update_delivery_state_from_text(delivery_state, chunk)
        if progress_callback is not None:
            progress_callback((index + 1) * PROGRESS_UNITS_PER_SEGMENT, progress_total)
    return _concat_audio_segments(segments, pause_between_chunks if len(segments) > 1 else 0.0)


def _parse_dialogue_lines(text: str) -> list[tuple[int, str]]:
    tag_re = re.compile(r"\[speaker[_\s-]*(\d+)\]\s*:\s*(.*)", re.IGNORECASE)
    turns: list[tuple[int, str]] = []
    current_speaker: int | None = None
    current_parts: list[str] = []
    for raw in text.strip().splitlines():
        match = tag_re.match(raw.strip())
        if match:
            if current_speaker is not None and current_parts:
                turns.append((current_speaker, " ".join(current_parts).strip()))
            current_speaker = int(match.group(1)) - 1
            current_parts = [match.group(2).strip()] if match.group(2).strip() else []
        elif raw.strip() and current_speaker is not None:
            current_parts.append(raw.strip())
    if current_speaker is not None and current_parts:
        turns.append((current_speaker, " ".join(current_parts).strip()))
    return [(speaker, line) for speaker, line in turns if line]


class HiggsV3LoadModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (
                    get_model_choices(),
                    {
                        "default": "Higgs Audio v3 TTS 4B - bosonai (auto-download)",
                        "tooltip": "Model folder under ComfyUI/models/higgsv3tts. Put model.safetensors in higgs-audio-v3-tts-4b or the root higgsv3tts folder.",
                    },
                ),
                "dtype": (
                    DTYPE_OPTIONS,
                    {
                        "default": "auto",
                        "tooltip": "Weight dtype for Higgs and its audio codec. auto uses bf16 on supported CUDA and fp32 otherwise. fp16 is hidden because it can produce non-finite audio.",
                    },
                ),
                "device": (
                    DEVICE_OPTIONS,
                    {
                        "default": "auto",
                        "tooltip": "Device for native inference. auto follows ComfyUI's current torch device; cuda is fastest; cpu is fallback only and very slow.",
                    },
                ),
                "attention": (
                    ATTENTION_OPTIONS,
                    {
                        "default": "auto",
                        "tooltip": "Attention backend. auto/sdpa are usually fastest here; flash_attention needs flash_attn; sageattention may be slower for token-by-token TTS.",
                    },
                ),
                "download_if_missing": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "If files are missing, downloads small assets plus the large model.safetensors into ComfyUI/models/higgsv3tts/higgs-audio-v3-tts-4b.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("HIGGSV3TTS_MODEL",)
    RETURN_NAMES = ("higgs_model",)
    FUNCTION = "load"
    CATEGORY = "Higgs v3 TTS"
    DESCRIPTION = "Load Higgs Audio v3 TTS natively with ComfyUI/AIMDO memory registration."

    def load(self, model: str, dtype: str, device: str, attention: str, download_if_missing: bool):
        bundle = load_higgs_bundle(
            model_choice=model,
            dtype_name=dtype,
            device_name=device,
            attention=attention,
            download_if_missing=bool(download_if_missing),
        )
        return (bundle,)

    @classmethod
    def IS_CHANGED(cls, model: str, dtype: str, device: str, attention: str, download_if_missing: bool):
        # If nothing about the widgets changed, ComfyUI would normally reuse
        # its cached bundle output untouched. bundle_state_token() flips to
        # "unloaded" the moment something (a Clear VRAM node, ComfyUI's own
        # memory manager, etc.) has torn down the active bundle in the
        # background, which forces a fresh load() instead of handing a dead
        # bundle to the next node in the graph.
        return bundle_state_token()


class HiggsV3Generate:
    @classmethod
    def INPUT_TYPES(cls):
        required = {"higgs_model": ("HIGGSV3TTS_MODEL",)}
        required.update(_common_generation_inputs())
        return {"required": required}

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "Higgs v3 TTS"
    DESCRIPTION = "Generate Higgs Audio v3 speech without reference audio."

    def generate(
        self,
        higgs_model,
        text: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        seed: int,
        longform_chunking: bool,
        words_per_chunk: int,
        tag_chunk: bool,
        pause_between_chunks: float,
    ) -> tuple[dict]:
        pbar = ProgressBar(PROGRESS_UNITS_PER_SEGMENT) if ProgressBar is not None else None

        def update_progress(current: int, total: int) -> None:
            if pbar is not None:
                pbar.update_absolute(current, total)

        audio = _generate_chunked_audio(
            higgs_model,
            text=text,
            control_prefix="",
            reference_audio=None,
            reference_text="",
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=int(top_k),
            seed=int(seed),
            longform_chunking=bool(longform_chunking),
            words_per_chunk=int(words_per_chunk),
            tag_chunk=bool(tag_chunk),
            pause_between_chunks=float(pause_between_chunks),
            progress_callback=update_progress,
            use_first_chunk_as_reference=True,
        )
        return (audio,)


class HiggsV3VoiceClone:
    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "higgs_model": ("HIGGSV3TTS_MODEL",),
            "text": _text_input(),
            "reference_audio": (
                "AUDIO",
                {
                    "tooltip": "Reference voice clip for cloning. Use clean speech with little music/noise; the same clip is reused for every longform chunk.",
                },
            ),
            "reference_text": (
                "STRING",
                {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Exact transcript of the reference clip. This strongly improves cloning and is reused for every chunk; Whisper output should be corrected if needed.",
                },
            ),
        }
        required.update(_generation_controls())
        return {"required": required}

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "clone"
    CATEGORY = "Higgs v3 TTS"
    DESCRIPTION = "Generate Higgs Audio v3 speech with zero-shot reference voice cloning."

    def clone(
        self,
        higgs_model,
        text: str,
        reference_audio: dict,
        reference_text: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        seed: int,
        longform_chunking: bool,
        words_per_chunk: int,
        tag_chunk: bool,
        pause_between_chunks: float,
    ) -> tuple[dict]:
        pbar = ProgressBar(PROGRESS_UNITS_PER_SEGMENT) if ProgressBar is not None else None

        def update_progress(current: int, total: int) -> None:
            if pbar is not None:
                pbar.update_absolute(current, total)

        audio = _generate_chunked_audio(
            higgs_model,
            text=text,
            control_prefix="",
            reference_audio=reference_audio,
            reference_text=reference_text,
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=int(top_k),
            seed=int(seed),
            longform_chunking=bool(longform_chunking),
            words_per_chunk=int(words_per_chunk),
            tag_chunk=bool(tag_chunk),
            pause_between_chunks=float(pause_between_chunks),
            progress_callback=update_progress,
        )
        return (audio,)


MULTI_SPEAKER_DEFAULT_TEXT = (
    "[Speaker_1]: Hello, I am speaker one.\n"
    "[Speaker_2]: And I am speaker two. <|sfx:laughter|>Haha, nice to meet you."
)


def _speaker_dynamic_inputs(count: int) -> list:
    inputs = []
    for speaker_index in range(1, count + 1):
        inputs.append(
            IO.Audio.Input(
                f"speaker_{speaker_index}_audio",
                optional=True,
                tooltip=(
                    f"Reference audio for Speaker_{speaker_index}. Use a clean clip; "
                    "matching reference text improves cloning."
                ),
            )
        )
        inputs.append(
            IO.String.Input(
                f"speaker_{speaker_index}_reference_text",
                multiline=True,
                default="",
                optional=True,
                tooltip=(
                    f"Transcript for Speaker_{speaker_index} reference audio. "
                    "Leave empty only if you do not have it."
                ),
            )
        )
    return inputs


def _io_generation_inputs() -> list:
    return [
        IO.Int.Input(
            "max_new_tokens",
            default=2048,
            min=32,
            max=8192,
            step=8,
            tooltip="Maximum audio-code tokens per single pass. 2048 is roughly 25-30 seconds; raise it or enable chunking if speech cuts off.",
        ),
        IO.Float.Input(
            "temperature",
            default=1.0,
            min=0.0,
            max=2.0,
            step=0.05,
            tooltip="Sampling variety. 0 is greedy; around 0.8-1.1 is usually natural.",
        ),
        IO.Float.Input(
            "top_p",
            default=0.95,
            min=0.0,
            max=1.0,
            step=0.01,
            tooltip="Nucleus sampling cutoff. 1.0 disables it; 0.9-0.98 keeps speech expressive.",
        ),
        IO.Int.Input(
            "top_k",
            default=50,
            min=0,
            max=1026,
            step=1,
            tooltip="Limits each codebook sample to the top K choices. 0 disables it.",
        ),
        IO.Int.Input(
            "seed",
            default=0,
            min=0,
            max=2**31 - 1,
            tooltip="0 uses the current random state. A positive value is repeatable and is reused unchanged for every longform chunk.",
        ),
        IO.Boolean.Input(
            "longform_chunking",
            default=True,
            tooltip="Split long text at sentence or pause-tag boundaries. Off is one direct pass and may stop early on long text.",
        ),
        IO.Int.Input(
            "words_per_chunk",
            default=45,
            min=20,
            max=300,
            step=5,
            tooltip="Target words per chunk. Around 35-55 fits the 2048-token default better; raise with max_new_tokens for longer chunks.",
        ),
        IO.Boolean.Input(
            "tag_chunk",
            default=False,
            tooltip="Cut chunks at every <|...|> tag instead of only at sentence breaks. Oversized tag sections are still split by words_per_chunk, with the active tag re-inserted at the start of each new piece so it keeps the tone/voice.",
        ),
        IO.Float.Input(
            "pause_between_chunks",
            default=0.15,
            min=0.0,
            max=2.0,
            step=0.05,
            tooltip="Seconds of silence inserted between longform chunks. Does not replace inline pause tags.",
        ),
    ]


def _generate_multi_speaker_audio(
    higgs_model,
    *,
    text: str,
    num_speakers: int,
    speaker_audio: dict[int, dict | None],
    speaker_ref_text: dict[int, str],
    pause_between_speakers: float,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int,
    longform_chunking: bool,
    words_per_chunk: int,
    tag_chunk: bool = False,
    pause_between_chunks: float,
) -> dict:
    turns = _parse_dialogue_lines(text)
    if not turns:
        raise ValueError("No speaker lines found. Use [Speaker_1]: text format.")

    num_speakers = max(2, min(MAX_SPEAKERS, int(num_speakers)))
    for speaker_idx, _line in turns:
        if speaker_idx < 0 or speaker_idx >= num_speakers:
            raise ValueError(f"Script uses Speaker_{speaker_idx + 1}, but num_speakers is {num_speakers}.")
        if speaker_audio.get(speaker_idx) is None:
            raise ValueError(f"Missing reference audio for Speaker_{speaker_idx + 1}.")

    pbar = ProgressBar(len(turns) * PROGRESS_UNITS_PER_SEGMENT) if ProgressBar is not None else None
    segments: list[dict] = []
    logger.info("Higgs v3 multi-speaker generation: %d turns, %d speakers.", len(turns), num_speakers)
    for index, (speaker_idx, line_text) in enumerate(turns):
        local_seed = int(seed) + index if seed else 0
        logger.info(
            "Higgs v3 speaker turn %d/%d [Speaker_%d]: %s",
            index + 1,
            len(turns),
            speaker_idx + 1,
            line_text[:90],
        )

        def update_turn(current: int, total: int, turn_index: int = index) -> None:
            if pbar is None:
                return
            fraction = min(1.0, max(0.0, float(current) / max(1, int(total))))
            value = turn_index * PROGRESS_UNITS_PER_SEGMENT + round(
                fraction * PROGRESS_UNITS_PER_SEGMENT
            )
            pbar.update_absolute(value, len(turns) * PROGRESS_UNITS_PER_SEGMENT)

        segments.append(
            _generate_chunked_audio(
                higgs_model,
                text=line_text,
                control_prefix="",
                reference_audio=speaker_audio[speaker_idx],
                reference_text=speaker_ref_text.get(speaker_idx, ""),
                max_new_tokens=int(max_new_tokens),
                temperature=float(temperature),
                top_p=float(top_p),
                top_k=int(top_k),
                seed=local_seed,
                longform_chunking=bool(longform_chunking),
                words_per_chunk=int(words_per_chunk),
                tag_chunk=bool(tag_chunk),
                pause_between_chunks=float(pause_between_chunks),
                progress_callback=update_turn,
            )
        )
        if pbar is not None:
            pbar.update_absolute(
                (index + 1) * PROGRESS_UNITS_PER_SEGMENT,
                len(turns) * PROGRESS_UNITS_PER_SEGMENT,
            )

    return _concat_audio_segments(segments, float(pause_between_speakers))


if _HAS_DYNAMIC_COMBO:

    class HiggsV3MultiSpeaker(IO.ComfyNode):
        @classmethod
        def define_schema(cls) -> IO.Schema:
            speaker_options = [
                IO.DynamicCombo.Option(str(count), _speaker_dynamic_inputs(count))
                for count in range(2, MAX_SPEAKERS + 1)
            ]
            return IO.Schema(
                node_id="HiggsV3MultiSpeaker",
                display_name="Higgs v3 Multi-Speaker",
                category="Higgs v3 TTS",
                description="Generate dialogue with multiple cloned Higgs v3 voices using [Speaker_N]: tags.",
                inputs=[
                    IO.Custom("HIGGSV3TTS_MODEL").Input("higgs_model"),
                    IO.String.Input(
                        "text",
                        multiline=True,
                        default=MULTI_SPEAKER_DEFAULT_TEXT,
                        tooltip=(
                            "Dialogue script. Use [Speaker_1]:, [Speaker_2]:, etc. "
                            "Lines without a speaker tag continue the previous speaker."
                        ),
                    ),
                    IO.DynamicCombo.Input(
                        "num_speakers",
                        options=speaker_options,
                        display_name="Number of Speakers",
                        tooltip=(
                            f"Number of active speakers (2-{MAX_SPEAKERS}). "
                            "Changing this adds or removes speaker audio/reference text inputs."
                        ),
                    ),
                    IO.Float.Input(
                        "pause_between_speakers",
                        default=0.3,
                        min=0.0,
                        max=3.0,
                        step=0.05,
                        tooltip="Seconds of silence inserted when moving from one speaker turn to the next.",
                    ),
                    *_io_generation_inputs(),
                ],
                outputs=[IO.Audio.Output(display_name="audio")],
            )

        @classmethod
        def execute(
            cls,
            higgs_model,
            text: str,
            num_speakers: dict,
            pause_between_speakers: float,
            max_new_tokens: int,
            temperature: float,
            top_p: float,
            top_k: int,
            seed: int,
            longform_chunking: bool,
            words_per_chunk: int,
            tag_chunk: bool,
            pause_between_chunks: float,
        ) -> IO.NodeOutput:
            speaker_count = int(num_speakers.get("num_speakers", 2))
            speaker_audio = {
                index - 1: num_speakers.get(f"speaker_{index}_audio")
                for index in range(1, speaker_count + 1)
            }
            speaker_ref_text = {
                index - 1: str(num_speakers.get(f"speaker_{index}_reference_text") or "")
                for index in range(1, speaker_count + 1)
            }
            audio = _generate_multi_speaker_audio(
                higgs_model,
                text=text,
                num_speakers=speaker_count,
                speaker_audio=speaker_audio,
                speaker_ref_text=speaker_ref_text,
                pause_between_speakers=float(pause_between_speakers),
                max_new_tokens=int(max_new_tokens),
                temperature=float(temperature),
                top_p=float(top_p),
                top_k=int(top_k),
                seed=int(seed),
                longform_chunking=bool(longform_chunking),
                words_per_chunk=int(words_per_chunk),
                tag_chunk=bool(tag_chunk),
                pause_between_chunks=float(pause_between_chunks),
            )
            return IO.NodeOutput(audio)

else:

    class HiggsV3MultiSpeaker:
        @classmethod
        def INPUT_TYPES(cls):
            required = {
                "higgs_model": ("HIGGSV3TTS_MODEL",),
                "text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": MULTI_SPEAKER_DEFAULT_TEXT,
                        "tooltip": "Dialogue script. Use [Speaker_1]:, [Speaker_2]:, etc. Inline SFX and pause tags are kept in-place.",
                    },
                ),
                "num_speakers": (
                    "INT",
                    {
                        "default": 2,
                        "min": 2,
                        "max": MAX_SPEAKERS,
                        "step": 1,
                        "tooltip": f"Number of active speakers (2-{MAX_SPEAKERS}). Upgrade ComfyUI for dynamic speaker inputs.",
                    },
                ),
                "speaker_1_audio": (
                    "AUDIO",
                    {"tooltip": "Reference audio for Speaker_1. Use a clean clip; matching transcript improves cloning."},
                ),
                "speaker_1_reference_text": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Transcript for Speaker_1 reference audio. Leave empty only if you do not have it.",
                    },
                ),
                "speaker_2_audio": (
                    "AUDIO",
                    {"tooltip": "Reference audio for Speaker_2. This voice is used by [Speaker_2]: lines."},
                ),
                "speaker_2_reference_text": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Transcript for Speaker_2 reference audio. A correct transcript improves voice match.",
                    },
                ),
                "pause_between_speakers": (
                    "FLOAT",
                    {
                        "default": 0.3,
                        "min": 0.0,
                        "max": 3.0,
                        "step": 0.05,
                        "tooltip": "Seconds of silence inserted when moving from one generated speaker turn to the next.",
                    },
                ),
            }
            required.update(_generation_controls())
            optional = {}
            for speaker_index in range(3, MAX_SPEAKERS + 1):
                optional[f"speaker_{speaker_index}_audio"] = (
                    "AUDIO",
                    {
                        "tooltip": (
                            f"Optional reference audio for Speaker_{speaker_index}. "
                            f"Required if the script uses [Speaker_{speaker_index}]:."
                        )
                    },
                )
                optional[f"speaker_{speaker_index}_reference_text"] = (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": f"Optional transcript for Speaker_{speaker_index} reference audio.",
                    },
                )
            return {"required": required, "optional": optional}

        RETURN_TYPES = ("AUDIO",)
        RETURN_NAMES = ("audio",)
        FUNCTION = "generate"
        CATEGORY = "Higgs v3 TTS"
        DESCRIPTION = "Generate dialogue with multiple cloned Higgs v3 voices using [Speaker_N]: tags."

        def generate(
            self,
            higgs_model,
            text: str,
            num_speakers: int,
            speaker_1_audio: dict,
            speaker_1_reference_text: str,
            speaker_2_audio: dict,
            speaker_2_reference_text: str,
            pause_between_speakers: float,
            max_new_tokens: int,
            temperature: float,
            top_p: float,
            top_k: int,
            seed: int,
            longform_chunking: bool,
            words_per_chunk: int,
            tag_chunk: bool,
            pause_between_chunks: float,
            **kwargs,
        ) -> tuple[dict]:
            speaker_audio: dict[int, dict | None] = {
                0: speaker_1_audio,
                1: speaker_2_audio,
            }
            speaker_ref_text: dict[int, str] = {
                0: speaker_1_reference_text,
                1: speaker_2_reference_text,
            }
            for speaker_index in range(3, MAX_SPEAKERS + 1):
                speaker_audio[speaker_index - 1] = kwargs.get(f"speaker_{speaker_index}_audio")
                speaker_ref_text[speaker_index - 1] = str(
                    kwargs.get(f"speaker_{speaker_index}_reference_text") or ""
                )
            audio = _generate_multi_speaker_audio(
                higgs_model,
                text=text,
                num_speakers=int(num_speakers),
                speaker_audio=speaker_audio,
                speaker_ref_text=speaker_ref_text,
                pause_between_speakers=float(pause_between_speakers),
                max_new_tokens=int(max_new_tokens),
                temperature=float(temperature),
                top_p=float(top_p),
                top_k=int(top_k),
                seed=int(seed),
                longform_chunking=bool(longform_chunking),
                words_per_chunk=int(words_per_chunk),
                tag_chunk=bool(tag_chunk),
                pause_between_chunks=float(pause_between_chunks),
            )
            return (audio,)


NODE_CLASS_MAPPINGS = {
    "HiggsV3LoadModel": HiggsV3LoadModel,
    "HiggsV3Generate": HiggsV3Generate,
    "HiggsV3VoiceClone": HiggsV3VoiceClone,
    "HiggsV3MultiSpeaker": HiggsV3MultiSpeaker,
    "HiggsV3WhisperTranscribe": HiggsV3WhisperTranscribe,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HiggsV3LoadModel": "Higgs v3 Load Model",
    "HiggsV3Generate": "Higgs v3 Generate",
    "HiggsV3VoiceClone": "Higgs v3 Voice Clone",
    "HiggsV3MultiSpeaker": "Higgs v3 Multi-Speaker",
    "HiggsV3WhisperTranscribe": "Higgs v3 Whisper Transcribe",
}
