# Document Parsing — Extension Guide

## 1. Extension surface

- ``registry.register_parser(custom)`` — replace or add a ``Modality`` handler.
- ``registry.sniff_modality`` — add magic bytes / extensions.
- Each parser's ``parse_bytes`` — subclass and override.

## 2. Recommended open-source backends

| 模态 | Task | Suggested OSS | Integration point |
| ---- | ---- | ------------- | ----------------- |
| 文 | PDF / DOCX | **PyMuPDF**, **python-docx**, **Unstructured** | New ``pdf/`` parser; register ``Modality.TEXT`` or new ``Modality.DOCUMENT`` |
| 图 | JPEG decode | **Pillow**, **opencv-python** | Subclass ``ImageParser._jpeg_decode`` |
| 图 | OCR | **Tesseract** + pytesseract, **PaddleOCR** | Set ``ParsedDocument.text`` from OCR output |
| 音 | MP3/FLAC decode | **pydub**, **soundfile**, **librosa** | Extend ``AudioParser`` |
| 音 | ASR / 语音转文字 | **Whisper**, **faster-whisper**, **Vosk** | Post-process ``waveform`` → ``doc.text`` |
| 视 | Demux / decode | **ffmpeg** (already probed), **PyAV** | ``VideoParser._extract_frames_*`` |
| 视 | Scene / caption | **CLIP**, **Video-LLaMA** hooks | Consume ``arrays["frames"]`` |

## 3. Example: Whisper transcript hook

```python
from algo_stack.parsing import parse, ParsedDocument

def attach_whisper(doc: ParsedDocument, model) -> ParsedDocument:
    if doc.modality.value != "audio" or "waveform" not in doc.arrays:
        return doc
    sr = doc.metadata["sample_rate"]
    doc.text = model.transcribe(doc.arrays["waveform"], sr)
    return doc
```

## 4. Example: register custom parser

```python
from algo_stack.parsing.registry import register_parser, Modality
from algo_stack.parsing._types import ParsedDocument

class PdfParser:
    modality = Modality.TEXT
    def parse_bytes(self, data: bytes, source: str) -> ParsedDocument:
        ...

register_parser(PdfParser())
```

## 5. Invariants

- ``parse_bytes`` must not mutate input ``data``.
- ``arrays`` values are ``np.ndarray`` (float64 for pixels/waveforms unless documented).
- ``source`` is preserved verbatim for traceability.

## 6. Pitfalls

- Do not assume ffmpeg is installed — always check ``shutil.which``.
- Truncating text without setting ``metadata["text_truncated"]`` breaks downstream audit trails.
- PNG filter types 3/4 (Paeth) are not implemented in the minimal decoder — use Pillow for arbitrary PNGs.
