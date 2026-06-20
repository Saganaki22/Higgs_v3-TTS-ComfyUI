<img width="1006" height="78" alt="logo" src="https://github.com/user-attachments/assets/02fde121-2666-4119-83dd-0931e40610b8" />

# Higgs_v3-TTS-ComfyUI

**English** | **[中文](./README_zh.md)**

**Version: v0.1.71**

ComfyUI nodes for [bosonai/higgs-audio-v3-tts-4b](https://huggingface.co/bosonai/higgs-audio-v3-tts-4b): multilingual conversational TTS, zero-shot voice cloning, inline emotion/style/prosody/SFX tags, longform chunking, multi-speaker dialogue, Whisper reference transcription, and ComfyUI/AIMDO DynamicVRAM support.

[![ComfyUI](https://img.shields.io/badge/ComfyUI-Custom%20Node-orange)](https://github.com/comfyanonymous/ComfyUI)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-bosonai%2Fhiggs--audio--v3--tts--4b-blue)](https://huggingface.co/bosonai/higgs-audio-v3-tts-4b)

> License note: Higgs Audio v3 TTS is released by Boson AI for research and non-commercial use. Do not use voice cloning without consent.

<img width="1077" height="1115" alt="Screenshot 2026-06-05 041705" src="https://github.com/user-attachments/assets/64d17c30-80c5-42b1-8d5a-2d98ebbc45ee" />


## Features

- **Native in-process inference** - Uses the local Transformers Qwen3 backbone plus Higgs audio-token embedding/head logic inside ComfyUI.
- **ComfyUI AUDIO in/out** - Reference voices and generated audio use standard ComfyUI `AUDIO`.
- **Voice cloning** - Reference audio plus optional transcript. A correct transcript materially improves cloning.
- **Multi-speaker dialogue** - Use `[Speaker_1]:`, `[Speaker_2]:`, etc. with separate reference voices.
- **Inline controls** - Emotion, style, prosody, pauses, and sound effects can be typed directly in the prompt.
- **Longform chunking** - Splits long text at sentence/pause boundaries and avoids cutting through `<|...|>` tags.
- **AIMDO DynamicVRAM support** - Higgs model/codec weights load CPU-first and use ComfyUI/AIMDO paging when DynamicVRAM is active.
- **Managed model folder** - Model files live under `ComfyUI/models/higgsv3tts/`.
- **No keep-loaded toggle, no unload node** - The loader handles model-switch cleanup internally.

## Installation

### Manual Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Saganaki22/Higgs_v3-TTS-ComfyUI.git
cd Higgs_v3-TTS-ComfyUI
python install.py
```

For this local Windows setup:

```powershell
...\venv\Scripts\python.exe ...\ComfyUI\custom_nodes\Higgs_v3-TTS-ComfyUI\install.py
```

Restart ComfyUI after installing or updating.

`install.py` does not modify `torch`, `torchaudio`, or `transformers`. The nodepack is built to work with the Qwen3 and Higgs Audio V2 tokenizer modules already present in this ComfyUI environment.

## Transformers Compatibility

This nodepack is built for Transformers **5.3.0 through 5.5.0**, with **5.5.0 recommended**.

It works across that range by loading Higgs v3 natively instead of relying on a remote-code `AutoModel` path. The runtime builds the Qwen3 backbone plus the Higgs audio-code embedding/head modules directly, maps `model.safetensors` weights into those modules, and normalizes the bundled Higgs Audio V2 tokenizer config so Transformers 5.3.0 can instantiate the codec without choking on newer metadata keys.

## Model Files

Place the large checkpoint here:

```text
ComfyUI/models/higgsv3tts/higgs-audio-v3-tts-4b/model.safetensors
```

This root-folder layout is also accepted:

```text
ComfyUI/models/higgsv3tts/model.safetensors
```

The nodepack includes/downloads the small Hugging Face assets into:

```text
ComfyUI/custom_nodes/Higgs_v3-TTS-ComfyUI/assets/higgs-audio-v3-tts-4b/
```

On load, small files such as `config.json`, `tokenizer.json`, `tokenizer_config.json`, and `model.safetensors.index.json` are copied beside `model.safetensors`.

If `download_if_missing` is enabled and `model.safetensors` is absent, the loader downloads the single large file from Hugging Face into:

```text
ComfyUI/models/higgsv3tts/higgs-audio-v3-tts-4b/
```

The checkpoint is about **9.31 GB** on disk. Expect roughly **11 GB VRAM** for the Higgs model/codec path with `bf16` on CUDA, plus extra headroom for ComfyUI and any other loaded models. AIMDO DynamicVRAM can reduce live VRAM pressure by paging castable weights, but 11 GB+ is still the recommended target until lower-VRAM workflows are tested thoroughly.

## Nodes

<details>
<summary><strong>1. Higgs v3 Load Model</strong> - Load the native Higgs v3 bundle</summary>

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | COMBO | `Higgs Audio v3 TTS 4B - bosonai (auto-download)` | Managed model choice from `ComfyUI/models/higgsv3tts/`. |
| `dtype` | COMBO | `auto` | `auto`, `bf16`. `auto` uses bf16 on supported CUDA and fp32 otherwise. `fp16` is hidden because it can produce non-finite audio. |
| `device` | COMBO | `auto` | `auto`, `cuda`, `cpu`. `auto` follows ComfyUI's current torch device. |
| `attention` | COMBO | `auto` | `auto`, `sdpa`, `flash_attention`, `sageattention`. |
| `download_if_missing` | BOOLEAN | `True` | Download missing small assets and the large model file if needed. |

**Output:** `higgs_model` (`HIGGSV3TTS_MODEL`)

</details>

<details>
<summary><strong>2. Higgs v3 Generate</strong> - Text to speech without a reference voice</summary>

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `higgs_model` | HIGGSV3TTS_MODEL | required | Output from Load Model. |
| `text` | STRING | example text | Text to synthesize. Inline control tags are allowed anywhere. |
| `max_new_tokens` | INT | `2048` | Maximum audio-token steps per single pass. `2048` is roughly 25-30 seconds of audio. |
| `temperature` | FLOAT | `1.0` | Sampling variety. `0` is greedy; `0.8-1.1` is usually natural. |
| `top_p` | FLOAT | `0.95` | Nucleus sampling. `1.0` disables it. |
| `top_k` | INT | `50` | Top-K codebook sampling. `0` disables it. |
| `seed` | INT | `0` | `0` uses the current random state; a positive seed is reused unchanged for every longform chunk. |
| `longform_chunking` | BOOLEAN | `True` | Split long text safely at sentence/pause boundaries. When off, the node makes one direct generation call. |
| `words_per_chunk` | INT | `45` | Target chunk size. Around 35-55 fits the 2048-token default better; CJK-like scripts use character-style splitting. |
| `pause_between_chunks` | FLOAT | `0.15` | Silence inserted between generated chunks. |

**Output:** `audio` (`AUDIO`)

</details>

<details>
<summary><strong>3. Higgs v3 Voice Clone</strong> - Text to speech using one reference voice</summary>

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `higgs_model` | HIGGSV3TTS_MODEL | required | Output from Load Model. |
| `text` | STRING | example text | Text to synthesize in the reference voice. |
| `reference_audio` | AUDIO | required | Clean speaker reference audio. |
| `reference_text` | STRING | empty | Transcript of the reference audio. Strongly recommended. |
| generation controls | same as Generate | | Same controls and longform chunking behavior as Generate. |

Reference cleanup is internal: trim enabled, silence threshold `-42 dB`, max reference length `100s`.

When longform chunking is on, every clone chunk uses the same original `reference_audio` and `reference_text`. When chunking is off, the clone node does one direct pass and does not call the chunk splitter.

**Output:** `audio` (`AUDIO`)

</details>

<details>
<summary><strong>4. Higgs v3 Multi-Speaker</strong> - Dialogue with multiple cloned voices</summary>

Use a tagged script:

```text
[Speaker_1]: Hello there.
[Speaker_2]: Hi. <|sfx:laughter|>Haha, I heard you.
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `higgs_model` | HIGGSV3TTS_MODEL | required | Output from Load Model. |
| `text` | STRING | example script | Multi-speaker script using `[Speaker_N]:` tags. |
| `num_speakers` | DYNAMIC | `2` | Number of speaker slots to use, from 2 to 6. Adds/removes speaker inputs in newer ComfyUI. |
| `speaker_N_audio` | AUDIO | required for active speakers | Reference voice for `[Speaker_N]:`. |
| `speaker_N_reference_text` | STRING | empty | Transcript for that speaker's reference audio. |
| `pause_between_speakers` | FLOAT | `0.3` | Silence inserted between turns. |
| generation controls | same as Generate | | Applied to every speaker turn/chunk. |

Speaker inputs are paired in order: `speaker_1_audio`, `speaker_1_reference_text`, then `speaker_2_audio`, `speaker_2_reference_text`, and so on up to 6. On older ComfyUI builds without dynamic inputs, extra speaker slots are shown as optional fallback inputs.

**Output:** `audio` (`AUDIO`)

</details>

<details>
<summary><strong>5. Higgs v3 Whisper Transcribe</strong> - Reference audio to transcript</summary>

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `audio` | AUDIO | required | Reference audio to transcribe. |
| `model` | COMBO | `whisper-large-v3-turbo (auto-download)` | Whisper model stored under `ComfyUI/models/audio_encoders/`. |
| `dtype` | COMBO | `auto` | `auto`, `bf16`, `fp32`. |
| `language` | COMBO | `auto` | Optional language hint. |
| `task` | COMBO | `transcribe` | `transcribe` keeps source language; `translate` outputs English. |
| `chunk_length_s` | INT | `30` | Whisper chunk length. `0` lets Transformers choose. |
| `download_if_missing` | BOOLEAN | `True` | Download selected Whisper model if missing. |

**Output:** `transcript` (`STRING`)

</details>

## Supported Languages

The upstream Higgs Audio v3 TTS model reports single-digit WER/CER across 100 languages. Boson splits them into two tiers.

### Polished, Production-Quality Tier

WER/CER under 5, 83 languages:

Afrikaans, Arabic, Armenian, Assamese, Asturian, Azerbaijani, Bashkir, Basque, Belarusian, Bengali, Bosnian, Bulgarian, Catalan, Cebuano, Central Kurdish, Chinese, Croatian, Czech, Danish, Dutch, Eastern Mari, English, Esperanto, Estonian, Finnish, French, Galician, Georgian, German, Greek, Gujarati, Haitian Creole, Hausa, Hebrew, Hindi, Hungarian, Indonesian, Italian, Javanese, Kannada, Kazakh, Kinyarwanda, Kyrgyz, Latvian, Lingala, Lithuanian, Luo, Macedonian, Malay, Malayalam, Maltese, Maori, Marathi, Mongolian, Nepali, Norwegian, Occitan, Persian, Polish, Portuguese, Romanian, Russian, Sepedi, Serbian, Shona, Slovak, Slovene, Spanish, Swahili, Swedish, Tagalog, Tajik, Tamil, Telugu, Turkish, Ukrainian, Urdu, Uyghur, Uzbek, Vietnamese, Xhosa, Zulu, Korean.

### Usable, Less Polished Tier

WER/CER between 5 and 10, 17 languages:

Albanian, Chichewa/Nyanja, Eastern Punjabi, Ganda, Icelandic, Irish, Kabyle, Kabuverdianu, Kamba, Latin, Luxembourgish, Oromo, Pashto, Sindhi, Somali, Umbundu, Welsh.

## Inline Control Tags

Inline tags typed directly into `text` are the control path for emotion, style, speed, pitch, expressiveness, pauses, and sound effects. The nodes do not add separate delivery dropdowns.

Use inline tags when you want changes at specific moments:

```text
<|emotion:anger|>I told you to wait.
<|prosody:pause|>
<|emotion:relief|>Okay. We can fix this.
<|sfx:sigh|>Ahh, let's start over.
```

Longform chunking preserves tags and avoids cutting inside `<|...|>`. Style and delivery-prosody tags such as speed, pitch, and expressiveness carry into later chunks. Emotion tags stay local to the chunk where they were written so a strong emotion is not automatically inserted at the beginning of every later chunk.

### Emotion

`elation`, `amusement`, `enthusiasm`, `determination`, `pride`, `contentment`, `affection`, `relief`, `contemplation`, `confusion`, `surprise`, `awe`, `longing`, `arousal`, `anger`, `fear`, `disgust`, `bitterness`, `sadness`, `shame`, `helplessness`

Example:

```text
<|emotion:amusement|>Wait, that was actually funny.
```

#### Preserving the cloned voice with strong emotions

Strong emotion tags can overpower the reference-speaker conditioning when they are the first token, causing the cloned voice to drift. This has been observed with `<|emotion:sadness|>`, while milder tags such as `<|emotion:amusement|>` may preserve the speaker correctly at the start.

For stronger emotions, let Higgs establish the cloned speaker with at least one word before the tag:

```text
This <|emotion:sadness|>is a short test sentence to test the text to speech.
```

Put the next word directly after the tag, without a space:

```text
Recommended: <|emotion:sadness|>is
Avoid:       <|emotion:sadness|> is
```

This is a known model limitation rather than the Voice Clone node selecting a random speaker. The reference audio and `reference_text` are still supplied to the model.

### Style

`singing`, `shouting`, `whispering`

Example:

```text
<|style:whispering|>Keep your voice down.
```

### Sound Effects

Sound effects are positional. Put them exactly where the sound should happen, and pair the tag with written sound text.

- `<|sfx:cough|>` - Cough. Suggested text: `Ahem`.
- `<|sfx:laughter|>` - Laugh. Suggested text: `Haha`, `Hehe`.
- `<|sfx:crying|>` - Cry. Suggested text: `Boohoo`, `Sob`.
- `<|sfx:screaming|>` - Scream. Suggested text: `Ahh`, `Aaah`.
- `<|sfx:burping|>` - Burp. Suggested text: `Burp`.
- `<|sfx:humming|>` - Hum. Suggested text: `Hmm`, `Mmm`.
- `<|sfx:sigh|>` - Sigh. Suggested text: `Ahh`, `Uh`.
- `<|sfx:sniff|>` - Sniff. Suggested text: `Sff`.
- `<|sfx:sneeze|>` - Sneeze. Suggested text: `Achoo`.

Example:

```text
That was perfect. <|sfx:laughter|>Haha, absolutely perfect.
```

### Prosody

- `<|prosody:speed_very_slow|>` - About 0.65x speed.
- `<|prosody:speed_slow|>` - About 0.85x speed.
- `<|prosody:speed_fast|>` - About 1.2x speed.
- `<|prosody:speed_very_fast|>` - About 1.4x speed.
- `<|prosody:pitch_low|>` - Lower pitch.
- `<|prosody:pitch_high|>` - Higher pitch.
- `<|prosody:pause|>` - Short pause, about 400-700 ms.
- `<|prosody:long_pause|>` - Longer pause, about 700-1500 ms.
- `<|prosody:expressive_high|>` - More expressive delivery.
- `<|prosody:expressive_low|>` - Flatter delivery.

#### Verifying speed controls

For a controlled comparison, use an exact `reference_text`, clean single-speaker reference audio, fixed seed `12345`, `temperature=0.8`, `top_p=1.0`, `top_k=50`, `max_new_tokens=1024`, and disable longform chunking. Generate both prompts with the same settings:

```text
<|prosody:speed_very_slow|>This is a short test sentence to test the text to speech of Higgs audio version 3 text to speech voice clone node by Saganaki 22

<|prosody:speed_very_fast|>This is a short test sentence to test the text to speech of Higgs audio version 3 text to speech voice clone node by Saganaki 22
```

In one verified test, `speed_very_slow` produced about 9 seconds of audio and `speed_very_fast` produced about 7 seconds. Exact durations depend on the reference voice and sampling, so compare them using the same fixed seed rather than expecting an exact duration.

## Longform Chunking

Higgs v3 has a finite context length, and long text can also hit `max_new_tokens`. Chunking is useful for long narration and dialogue.

The audio codec is roughly 75 audio tokens per second, so `max_new_tokens=2048` is not enough for very long text in one pass. If chunking is off, the model may stop before the whole prompt is spoken. Use chunking for long clone/generation prompts, or raise `max_new_tokens` for longer single passes.

The chunker:

- splits at sentence endings and `<|prosody:pause|>` / `<|prosody:long_pause|>`;
- avoids cutting through `<|...|>` control tags;
- avoids ending a chunk with a bare SFX/control tag;
- carries active style and delivery-prosody tags into later chunks;
- keeps emotion tags local to the chunk where they appear;
- reuses the same positive seed unchanged for every chunk;
- inserts `pause_between_chunks` seconds of silence between chunks.

Voice consistency:

- Generate uses chunk 1 as an internal voice reference for later chunks when no external reference audio is connected.
- Voice Clone uses the same user-provided reference audio and reference text for every chunk.
- Multi-Speaker uses each speaker's reference audio and reference text for every chunk in that speaker's turn.

For very controlled acting, write short turns manually or use Multi-Speaker lines as natural chunk boundaries.

## Console Progress

During generation, the node logs progress in the ComfyUI terminal:

- longform chunk count;
- chunk number and preview text;
- multi-speaker turn number and speaker id;
- an in-place `tqdm` audio-token bar with percentage, elapsed time, ETA, and token rate:

```text
Higgs v3 audio tokens: 64%|████████████████████████▋             | 1310/2048 [00:39<00:21, 34.52tok/s]
```

If the model emits its natural stop token before `max_new_tokens`, the completed bar adjusts to the actual generated token count and finishes at 100%.

The ComfyUI node progress bar is updated continuously from native audio-token progress. For longform and multi-speaker generation, token progress is mapped across the full set of chunks and turns so the bar advances smoothly without resetting between segments.

## Attention Backends

| Option | Behavior |
|--------|----------|
| `auto` | Uses PyTorch SDPA. |
| `sdpa` | Explicit PyTorch scaled-dot-product attention. |
| `flash_attention` | Uses Transformers FlashAttention 2 path when `flash_attn` is installed. |
| `sageattention` | Uses SDPA config plus a runtime SageAttention patch for CUDA BF16 tensors. It can be slower than SDPA/FlashAttention for this token-by-token generation path, so benchmark it on your GPU. |

If an optional attention package is not installed, selecting it raises a clear error.

## Memory Behavior

Higgs v3 loads weights on CPU first, patches the Qwen/Higgs/codec torch modules into Comfy-castable modules, and registers the model plus codec through ComfyUI model management. When AIMDO DynamicVRAM is active, castable weights are VBAR-backed and paged into VRAM during forward passes; without AIMDO, ComfyUI falls back to normal static model loading. Whisper is also registered with ComfyUI model management.

There is no dedicated unload node and no keep-loaded toggle. Changing model, dtype, device, or attention settings hard-unloads the previous active Higgs bundle before loading the new one: it unregisters the Comfy model patchers, clears AIMDO state, moves weights to meta, breaks bundle references, runs Python GC, and asks Comfy/PyTorch to empty accelerator caches.

ComfyUI offload is different from hard unload. Offload frees VRAM by moving weights to CPU RAM so the same loaded node can run again without re-reading the checkpoint. That CPU RAM residency is expected until the active bundle is hard-unloaded.

## Troubleshooting

### Download says internet is missing

If the log mentions `hf-mirror.com` or Hugging Face metadata/HEAD failures, update to this nodepack version and retry. Downloads are forced through `https://huggingface.co`.

The large file is downloaded as only:

```text
model.safetensors
```

Small config/tokenizer assets are handled separately.

### Output cuts off

Use `longform_chunking=True` for long text. The node now avoids the misleading `chunk 1/1` path when chunking is off, but a single unchunked pass can still end early if the text needs more audio tokens than `max_new_tokens` allows. Raise `max_new_tokens` or keep `words_per_chunk` around 35-55 for the 2048 default.

### Voice clone sounds weak

Provide a clean reference clip and a correct `reference_text`. Whisper can help, but a manually corrected transcript is better.

If a strong emotion tag at the beginning changes the cloned voice, place it after the first word and attach the next word directly to the tag:

```text
This <|emotion:sadness|>is a short test sentence.
```

### SFX does not trigger

Make sure the SFX tag is immediately followed by written sound text:

```text
<|sfx:laughter|>Haha
```

### Inline controls are ignored after chunking

Use `longform_chunking=True`. Style and delivery-prosody tags carry through chunks, but emotion tags intentionally remain local to avoid cloned-speaker drift. Add an emotion tag again inside any later chunk where you want that emotion applied.
