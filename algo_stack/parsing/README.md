# Multimodal Document Parsing — 文 / 图 / 音 / 视

> Open-source, NumPy-first document parsing components for **text, image,
> audio, and video**. Unified API with pluggable backends (ffmpeg, Whisper,
> Tesseract, …).

## Install / import

```python
from algo_stack.parsing import parse, DocumentParser, Modality
```

## Public API

| Name | Description |
| ---- | ----------- |
| `parse(source, **kwargs)` | One-shot parse path or bytes → `ParsedDocument` |
| `DocumentParser(...)` | Configurable parser (force modality, truncate text, …) |
| `ParsedDocument` | Result dataclass: `text`, `metadata`, `arrays`, `blocks` |
| `Modality` | Enum: `TEXT`, `IMAGE`, `AUDIO`, `VIDEO` |

### Modality parsers (sub-packages)

| 模态 | Module | Built-in capability |
| ---- | ------ | ------------------- |
| 文 | `algo_stack.parsing.text` | `.txt/.md/.json/.csv/.html` |
| 图 | `algo_stack.parsing.image` | PNG → RGB `ndarray`; JPEG/GIF metadata |
| 音 | `algo_stack.parsing.audio` | WAV → waveform `ndarray` |
| 视 | `algo_stack.parsing.video` | Container sniff + **ffprobe/ffmpeg** when installed |

## Quick start

```python
from algo_stack.parsing import parse

doc = parse("notes.md")
print(doc.text)

img = parse("scan.png")
print(img.arrays["rgb"].shape)   # (H, W, 3) float64 in [0,1]

wav = parse("voice.wav")
print(wav.metadata["duration_sec"], wav.arrays["waveform"].shape)

vid = parse("clip.mp4")
print(vid.metadata.get("duration_sec"), vid.metadata.get("ffprobe_available"))
```

## Run the bundled demo

```bash
python -m algo_stack.parsing.example
```

## Where to go next

- Architecture & design → [`PRINCIPLE.md`](PRINCIPLE.md)
- Plug in OCR / ASR / PDF / ffmpeg → [`EXTENSION.md`](EXTENSION.md)
- Per-modality docs → `text/`, `image/`, `audio/`, `video/` subfolders
