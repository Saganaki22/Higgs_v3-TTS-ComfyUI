<img width="1006" height="78" alt="logo" src="https://github.com/user-attachments/assets/02fde121-2666-4119-83dd-0931e40610b8" />

# Higgs_v3-TTS-ComfyUI

**[English](./README.md)** | **中文**

**版本：v0.1.71**

[bosonai/higgs-audio-v3-tts-4b](https://huggingface.co/bosonai/higgs-audio-v3-tts-4b) 的 ComfyUI 原生节点：多语言对话式 TTS、零样本语音克隆、内联情感/风格/韵律/音效标签、长文本分块、多说话人对话、Whisper 参考音频转写，以及 ComfyUI/AIMDO DynamicVRAM 支持。

[![ComfyUI](https://img.shields.io/badge/ComfyUI-Custom%20Node-orange)](https://github.com/comfyanonymous/ComfyUI)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-bosonai%2Fhiggs--audio--v3--tts--4b-blue)](https://huggingface.co/bosonai/higgs-audio-v3-tts-4b)

> 许可说明：Higgs Audio v3 TTS 由 Boson AI 发布，仅供研究和非商业用途。未经许可请勿使用语音克隆功能。

<img width="1077" height="1115" alt="Screenshot 2026-06-05 041705" src="https://github.com/user-attachments/assets/64d17c30-80c5-42b1-8d5a-2d98ebbc45ee" />


## 功能特性

- **原生进程内推理** — 在 ComfyUI 内使用本地 Transformers Qwen3 主干网络及 Higgs 音频 token 嵌入/头部逻辑。
- **ComfyUI AUDIO 输入/输出** — 参考音频和生成的音频使用标准 ComfyUI `AUDIO` 类型。
- **语音克隆** — 参考音频加可选转写文本。正确的转写文本能显著提升克隆质量。
- **多说话人对话** — 使用 `[Speaker_1]:`、`[Speaker_2]:` 等标签配合各自的参考音频。
- **内联控制** — 情感、风格、韵律、停顿和音效可直接在提示文本中输入。
- **长文本分块** — 在句子/停顿边界处拆分长文本，避免切断 `<|...|>` 标签。
- **AIMDO DynamicVRAM 支持** — Higgs 模型/codec 权重先加载到 CPU，启用 DynamicVRAM 时使用 ComfyUI/AIMDO 分页。
- **托管模型文件夹** — 模型文件存放在 `ComfyUI/models/higgsv3tts/` 下。
- **无保持加载开关，无卸载节点** — 加载器内部处理模型切换的清理工作。

## 安装

### 手动安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Saganaki22/Higgs_v3-TTS-ComfyUI.git
cd Higgs_v3-TTS-ComfyUI
python install.py
```

本地 Windows 环境使用：

```powershell
...\venv\Scripts\python.exe ...\ComfyUI\custom_nodes\Higgs_v3-TTS-ComfyUI\install.py
```

安装或更新后请重启 ComfyUI。

`install.py` 不会修改 `torch`、`torchaudio` 或 `transformers`。本节点包与当前 ComfyUI 环境中已有的 Qwen3 和 Higgs Audio V2 分词器模块兼容。

## Transformers 兼容性

本节点包面向 Transformers **5.3.0 到 5.5.0**，推荐使用 **5.5.0**。

实现方式不是依赖远程代码的 `AutoModel` 路径，而是在本地原生构建 Qwen3 主干网络、Higgs 音频 code 嵌入和输出头，直接映射 `model.safetensors` 权重。同时会规范化随包的 Higgs Audio V2 tokenizer 配置，移除旧版 Transformers 5.3.0 不接受的新元数据键，因此 5.3.0 到 5.5.0 都能工作。

## 模型文件

将大检查点放置在：

```text
ComfyUI/models/higgsv3tts/higgs-audio-v3-tts-4b/model.safetensors
```

也支持根文件夹布局：

```text
ComfyUI/models/higgsv3tts/model.safetensors
```

节点包包含/下载的小型 Hugging Face 资产文件位于：

```text
ComfyUI/custom_nodes/Higgs_v3-TTS-ComfyUI/assets/higgs-audio-v3-tts-4b/
```

加载时，`config.json`、`tokenizer.json`、`tokenizer_config.json` 和 `model.safetensors.index.json` 等小文件会被复制到 `model.safetensors` 旁边。

如果启用 `download_if_missing` 且 `model.safetensors` 不存在，加载器会从 Hugging Face 下载大文件到：

```text
ComfyUI/models/higgsv3tts/higgs-audio-v3-tts-4b/
```

检查点磁盘大小约 **9.31 GB**。CUDA `bf16` 下 Higgs 模型/codec 路径大约需要 **11 GB VRAM**，另外还需要给 ComfyUI 和其他已加载模型留出余量。AIMDO DynamicVRAM 可以通过分页可转换权重降低实时 VRAM 压力，但在低显存工作流充分测试之前，仍建议以 11 GB+ 作为目标。

## 节点

<details>
<summary><strong>1. Higgs v3 Load Model（加载模型）</strong> - 加载原生 Higgs v3 模型包</summary>

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model` | COMBO | `Higgs Audio v3 TTS 4B - bosonai (auto-download)` | 从 `ComfyUI/models/higgsv3tts/` 选择托管模型。 |
| `dtype` | COMBO | `auto` | `auto`、`bf16`。`auto` 在支持 bf16 的 CUDA 上使用 bf16，否则使用 fp32。`fp16` 已隐藏，因为它可能产生非有限音频样本。 |
| `device` | COMBO | `auto` | `auto`、`cuda`、`cpu`。`auto` 跟随 ComfyUI 当前 torch 设备。 |
| `attention` | COMBO | `auto` | `auto`、`sdpa`、`flash_attention`、`sageattention`。 |
| `download_if_missing` | BOOLEAN | `True` | 缺失时自动下载小资产文件和大模型文件。 |

**输出：** `higgs_model`（`HIGGSV3TTS_MODEL`）

</details>

<details>
<summary><strong>2. Higgs v3 Generate（生成）</strong> - 无参考音频的文本转语音</summary>

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `higgs_model` | HIGGSV3TTS_MODEL | 必填 | Load Model 的输出。 |
| `text` | STRING | 示例文本 | 要合成的文本。允许使用内联控制标签。 |
| `max_new_tokens` | INT | `2048` | 每次生成的最大音频 token 步数。`2048` 大约对应 25-30 秒音频。 |
| `temperature` | FLOAT | `1.0` | 采样多样性。`0` 为贪心；`0.8-1.1` 通常较自然。 |
| `top_p` | FLOAT | `0.95` | 核采样。`1.0` 禁用。 |
| `top_k` | INT | `50` | Top-K 码本采样。`0` 禁用。 |
| `seed` | INT | `0` | `0` 使用当前随机状态；正值种子会在每个长文本块中保持不变并重复使用。 |
| `longform_chunking` | BOOLEAN | `True` | 在句子/停顿边界安全拆分长文本。关闭时节点只进行一次直接生成。 |
| `words_per_chunk` | INT | `45` | 目标块大小。35-55 左右更适合 2048 token 默认值；CJK 文本按字符式分割。 |
| `pause_between_chunks` | FLOAT | `0.15` | 块之间插入的静音时长（秒）。 |

**输出：** `audio`（`AUDIO`）

</details>

<details>
<summary><strong>3. Higgs v3 Voice Clone（语音克隆）</strong> - 使用参考音频的文本转语音</summary>

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `higgs_model` | HIGGSV3TTS_MODEL | 必填 | Load Model 的输出。 |
| `text` | STRING | 示例文本 | 使用参考音色合成的文本。 |
| `reference_audio` | AUDIO | 必填 | 干净的说话人参考音频。 |
| `reference_text` | STRING | 空 | 参考音频的转写文本。强烈建议提供。 |
| 生成控制 | 同 Generate | | 与 Generate 相同的控制和长文本分块行为。 |

参考音频清理为内部处理：启用裁剪，静音阈值 `-42 dB`，最大参考长度 `100s`。

启用长文本分块时，Voice Clone 的每个块都会使用相同的原始 `reference_audio` 和 `reference_text`。关闭分块时，Clone 节点只进行一次直接生成，不调用分块器。

**输出：** `audio`（`AUDIO`）

</details>

<details>
<summary><strong>4. Higgs v3 Multi-Speaker（多说话人）</strong> - 多个克隆音色的对话</summary>

使用标签脚本：

```text
[Speaker_1]: 你好啊。
[Speaker_2]: 嗨。<|sfx:laughter|>哈哈，我听到了。
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `higgs_model` | HIGGSV3TTS_MODEL | 必填 | Load Model 的输出。 |
| `text` | STRING | 示例脚本 | 使用 `[Speaker_N]:` 标签的多说话人脚本。 |
| `num_speakers` | DYNAMIC | `2` | 使用的说话人数量，2 到 6。 |
| `speaker_N_audio` | AUDIO | 活跃说话人必填 | `[Speaker_N]:` 的参考音频。 |
| `speaker_N_reference_text` | STRING | 空 | 该说话人参考音频的转写文本。 |
| `pause_between_speakers` | FLOAT | `0.3` | 说话人轮次间插入的静音时长（秒）。 |
| 生成控制 | 同 Generate | | 应用于每个说话人轮次/块。 |

说话人输入按顺序配对：`speaker_1_audio`、`speaker_1_reference_text`，然后 `speaker_2_audio`、`speaker_2_reference_text`，最多 6 个。

**输出：** `audio`（`AUDIO`）

</details>

<details>
<summary><strong>5. Higgs v3 Whisper Transcribe（Whisper 转写）</strong> - 参考音频转文本</summary>

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `audio` | AUDIO | 必填 | 要转写的参考音频。 |
| `model` | COMBO | `whisper-large-v3-turbo (auto-download)` | Whisper 模型，存放在 `ComfyUI/models/audio_encoders/` 下。 |
| `dtype` | COMBO | `auto` | `auto`、`bf16`、`fp32`。 |
| `language` | COMBO | `auto` | 可选语言提示。 |
| `task` | COMBO | `transcribe` | `transcribe` 保持源语言；`translate` 输出英文。 |
| `chunk_length_s` | INT | `30` | Whisper 块长度。`0` 由 Transformers 自动选择。 |
| `download_if_missing` | BOOLEAN | `True` | 缺失时下载所选 Whisper 模型。 |

**输出：** `transcript`（`STRING`）

</details>

## 支持的语言

上游 Higgs Audio v3 TTS 模型在 100 种语言上报告了个位数的 WER/CER。Boson 将其分为两个等级。

### 精细、生产级质量

WER/CER 低于 5，共 83 种语言：

南非语、阿拉伯语、亚美尼亚语、阿萨姆语、阿斯图里亚斯语、阿塞拜疆语、巴什基尔语、巴斯克语、白俄罗斯语、孟加拉语、波斯尼亚语、保加利亚语、加泰罗尼亚语、宿务语、库尔德语（中）、中文、克罗地亚语、捷克语、丹麦语、荷兰语、东马里语、英语、世界语、爱沙尼亚语、芬兰语、法语、加利西亚语、格鲁吉亚语、德语、希腊语、古吉拉特语、海地克里奥尔语、豪萨语、希伯来语、印地语、匈牙利语、印尼语、意大利语、爪哇语、卡纳达语、哈萨克语、基尼亚卢旺达语、吉尔吉斯语、拉脱维亚语、林加拉语、立陶宛语、卢奥语、马其顿语、马来语、马拉雅拉姆语、马耳他语、毛利语、马拉地语、蒙古语、尼泊尔语、挪威语、奥克语、波斯语、波兰语、葡萄牙语、罗马尼亚语、俄语、塞佩迪语、塞尔维亚语、绍纳语、斯洛伐克语、斯洛文尼亚语、西班牙语、斯瓦希里语、瑞典语、他加禄语、塔吉克语、泰米尔语、泰卢固语、土耳其语、乌克兰语、乌尔都语、维吾尔语、乌兹别克语、越南语、科萨语、祖鲁语、韩语。

### 可用，质量稍低

WER/CER 在 5 到 10 之间，共 17 种语言：

阿尔巴尼亚语、奇切瓦语/尼亚贾语、东旁遮普语、干达语、冰岛语、爱尔兰语、卡比尔语、佛得角克里奥尔语、坎巴语、拉丁语、卢森堡语、奥罗莫语、普什图语、信德语、索马里语、翁本杜语、威尔士语。

## 内联控制标签

直接在 `text` 中输入的内联标签是情感、风格、语速、音高、表现力、停顿和音效的控制方式。节点不再提供单独的演绎下拉控件。

在需要特定时刻变化时使用内联标签：

```text
<|emotion:anger|>我告诉过你等着。
<|prosody:pause|>
<|emotion:relief|>好吧，我们可以解决这个问题。
<|sfx:sigh|>啊，重新来过吧。
```

长文本分块会保留标签并避免在 `<|...|>` 内部切割。风格以及语速、音高、表现力等演绎韵律标签会传递到后续块。情感标签仅作用于用户写入该标签的块，避免强烈情感被自动插入每个后续块的开头。

### 情感

`elation`（兴奋）、`amusement`（好笑）、`enthusiasm`（热情）、`determination`（坚定）、`pride`（自豪）、`contentment`（满足）、`affection`（喜爱）、`relief`（释然）、`contemplation`（沉思）、`confusion`（困惑）、`surprise`（惊讶）、`awe`（敬畏）、`longing`（渴望）、`arousal`（激动）、`anger`（愤怒）、`fear`（恐惧）、`disgust`（厌恶）、`bitterness`（苦涩）、`sadness`（悲伤）、`shame`（羞愧）、`helplessness`（无助）

示例：

```text
<|emotion:amusement|>等等，那真的好好笑。
```

#### 使用强烈情感时保持克隆音色

当强烈情感标签作为第一个 token 时，它可能压过参考说话人的条件信息，导致克隆音色发生漂移。已观察到 `<|emotion:sadness|>` 可能出现这种情况，而 `<|emotion:amusement|>` 等较温和的标签放在开头时可能仍能正确保持说话人。

对于较强的情感，先让 Higgs 用至少一个词建立克隆说话人的音色，再放置标签：

```text
This <|emotion:sadness|>is a short test sentence to test the text to speech.
```

标签后应直接连接下一个词，不要添加空格：

```text
推荐：<|emotion:sadness|>is
避免：<|emotion:sadness|> is
```

这是已知的模型限制，并不是 Voice Clone 节点随机选择了其他说话人。参考音频和 `reference_text` 仍然会正常传递给模型。

### 风格

`singing`（唱歌）、`shouting`（喊叫）、`whispering`（耳语）

示例：

```text
<|style:whispering|>小声点。
```

### 音效

音效是位置性的。将标签精确放在声音应该出现的位置，并与书面声音文字配对。

- `<|sfx:cough|>` - 咳嗽。建议文字：`咳咳`。
- `<|sfx:laughter|>` - 笑。建议文字：`哈哈`、`嘿嘿`。
- `<|sfx:crying|>` - 哭。建议文字：`呜呜`。
- `<|sfx:screaming|>` - 尖叫。建议文字：`啊啊`。
- `<|sfx:burping|>` - 打嗝。建议文字：`嗝`。
- `<|sfx:humming|>` - 哼。建议文字：`嗯嗯`、`唔`。
- `<|sfx:sigh|>` - 叹气。建议文字：`啊`、`唉`。
- `<|sfx:sniff|>` - 吸鼻子。建议文字：`嘶`。
- `<|sfx:sneeze|>` - 打喷嚏。建议文字：`阿嚏`。

示例：

```text
太完美了。<|sfx:laughter|>哈哈，绝对完美。
```

### 韵律

- `<|prosody:speed_very_slow|>` - 约 0.65 倍速。
- `<|prosody:speed_slow|>` - 约 0.85 倍速。
- `<|prosody:speed_fast|>` - 约 1.2 倍速。
- `<|prosody:speed_very_fast|>` - 约 1.4 倍速。
- `<|prosody:pitch_low|>` - 降低音高。
- `<|prosody:pitch_high|>` - 提高音高。
- `<|prosody:pause|>` - 短暂停顿，约 400-700 毫秒。
- `<|prosody:long_pause|>` - 较长停顿，约 700-1500 毫秒。
- `<|prosody:expressive_high|>` - 更有表现力的演绎。
- `<|prosody:expressive_low|>` - 更平淡的演绎。

#### 验证语速控制

进行可控对比时，请使用准确的 `reference_text`、干净的单说话人参考音频、固定种子 `12345`、`temperature=0.8`、`top_p=1.0`、`top_k=50`、`max_new_tokens=1024`，并关闭长文本分块。使用完全相同的设置生成以下两个提示：

```text
<|prosody:speed_very_slow|>This is a short test sentence to test the text to speech of Higgs audio version 3 text to speech voice clone node by Saganaki 22

<|prosody:speed_very_fast|>This is a short test sentence to test the text to speech of Higgs audio version 3 text to speech voice clone node by Saganaki 22
```

在一次已验证的测试中，`speed_very_slow` 生成了约 9 秒音频，`speed_very_fast` 生成了约 7 秒音频。实际时长取决于参考音色和采样结果，因此应使用相同的固定种子进行对比，而不是期待固定的精确时长。

## 长文本分块

Higgs v3 上下文长度有限，长文本也可能达到 `max_new_tokens` 上限。分块适用于长篇叙述和对话。

音频 codec 大约是每秒 75 个音频 token，因此 `max_new_tokens=2048` 不足以在单次生成中说完很长的文本。关闭分块时，如果文本所需音频 token 超过上限，模型可能在说完整段前停止。长文本建议开启分块，或为单次生成提高 `max_new_tokens`。

分块器会：

- 在句子结尾和 `<|prosody:pause|>` / `<|prosody:long_pause|>` 处拆分；
- 避免切断 `<|...|>` 控制标签；
- 避免以裸 SFX/控制标签结束一个块；
- 将活跃的风格和演绎韵律标签传递到后续块；
- 让情感标签仅作用于它所在的块；
- 在每个块中重复使用相同的正值种子；
- 在块之间插入 `pause_between_chunks` 秒的静音。

音色一致性：

- Generate 在无外部参考音频时，将第 1 块作为内部音色参考用于后续块。
- Voice Clone 对每个块使用相同的用户参考音频和参考文本。
- Multi-Speaker 对每个说话人轮次使用各自的参考音频和参考文本。

如需精细的表演控制，可手动编写短轮次或使用 Multi-Speaker 标签作为自然的分块边界。

## 控制台进度

生成期间，节点会在 ComfyUI 终端中记录进度：

- 长文本块数；
- 当前块编号和预览文本；
- 多说话人轮次编号和说话人 ID；
- 原地更新的 `tqdm` 音频 token 进度条，显示百分比、已用时间、预计剩余时间和 token 速度：

```text
Higgs v3 audio tokens: 64%|████████████████████████▋             | 1310/2048 [00:39<00:21, 34.52tok/s]
```

如果模型在达到 `max_new_tokens` 前生成自然停止 token，完成后的进度条会改用实际生成的 token 数，并显示为 100%。

ComfyUI 节点进度条会根据原生音频 token 进度持续更新。对于长文本和多说话人生成，token 进度会映射到全部块和轮次，进度条会平滑前进，不会在片段之间重置。

## 注意力后端

| 选项 | 行为 |
|------|------|
| `auto` | 使用 PyTorch SDPA。 |
| `sdpa` | 显式 PyTorch 缩放点积注意力。 |
| `flash_attention` | 安装 `flash_attn` 后使用 Transformers FlashAttention 2 路径。 |
| `sageattention` | 使用 SDPA 配置加运行时 SageAttention 补丁，适用于 CUDA BF16 张量。对于此逐 token 生成路径可能比 SDPA/FlashAttention 更慢，建议在自己的 GPU 上测试。 |

如果可选注意力包未安装，选择它将引发清晰的错误。

## 显存行为

Higgs v3 会先将权重加载到 CPU，将 Qwen/Higgs/codec torch 模块转换为 Comfy 可 cast 的模块，并通过 ComfyUI 模型管理注册模型和 codec。启用 AIMDO DynamicVRAM 时，可转换权重会使用 VBAR 后端，并在 forward 期间分页进入 VRAM；未启用 AIMDO 时，ComfyUI 会回退到普通静态模型加载。Whisper 也会注册到 ComfyUI 模型管理。

没有专用卸载节点，也没有保持加载开关。更改模型、dtype、设备或注意力设置时，会先硬卸载之前的活跃 Higgs 模型包再加载新的：注销 Comfy model patcher、清空 AIMDO 状态、将权重移到 meta、断开 bundle 引用、运行 Python GC，并请求 Comfy/PyTorch 清空加速器缓存。

ComfyUI 的 offload 与硬卸载不同。offload 是把权重从 VRAM 移到 CPU RAM，以便同一个已加载节点下次运行时不用重新读取检查点。因此在仅 offload 时看到 CPU RAM 占用是预期行为，直到活跃 bundle 被硬卸载。

## 故障排除

### 下载提示缺少网络

如果日志提到 `hf-mirror.com` 或 Hugging Face 元数据/HEAD 失败，请更新到本节点包最新版本后重试。下载强制通过 `https://huggingface.co`。

大文件仅下载为：

```text
model.safetensors
```

小型配置/分词器资产单独处理。

### 输出截断

长文本请使用 `longform_chunking=True`。关闭分块时，节点现在会走一次直接生成，不再出现误导性的 `chunk 1/1` 路径；但如果文本需要的音频 token 超过 `max_new_tokens`，单次生成仍可能提前结束。可以提高 `max_new_tokens`，或在 2048 默认值下把 `words_per_chunk` 保持在 35-55 左右。

### 语音克隆效果不佳

提供干净的参考音频片段和正确的 `reference_text`。Whisper 可以辅助，但人工校正的转写效果更好。

如果放在开头的强烈情感标签改变了克隆音色，请将标签移到第一个词之后，并让下一个词直接紧跟标签：

```text
This <|emotion:sadness|>is a short test sentence.
```

### 音效未触发

确保音效标签后紧跟书面声音文字：

```text
<|sfx:laughter|>哈哈
```

### 分块后内联控制被忽略

使用 `longform_chunking=True`。风格和演绎韵律标签会传递到后续块，但为了减少克隆说话人漂移，情感标签会有意保留在其所在块中。如需在后续块继续使用某种情感，请在该块中再次写入情感标签。
