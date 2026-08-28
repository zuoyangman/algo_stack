# Document Parsing — Principle

## 1. Problem

Multimodal ML pipelines need a **uniform ingestion layer** that turns files on
disk into:

- **Text** for NLP / retrieval
- **Image tensors** for vision models
- **Waveforms** for speech / audio models
- **Video metadata + frames** for video understanding

## 2. Architecture

```
        ┌────────────── DocumentParser.parse ──────────────┐
        │  sniff_modality(ext + magic bytes + MIME)        │
        └────────────┬──────────┬──────────┬───────────────┘
                     │          │          │
              TextParser  ImageParser  AudioParser  VideoParser
                     │          │          │          │
                     └──────────┴──────────┴──────────┘
                              ParsedDocument
                    (text, metadata, arrays, blocks)
```

Each parser implements ``parse_bytes(data, source) -> ParsedDocument``.

## 3. Design choices

- **Stdlib + NumPy core** — no mandatory Pillow/ffmpeg; optional tools
  degrade gracefully with metadata + extension hooks.
- **Registry pattern** — ``register_parser()`` allows swapping backends
  (e.g. Pillow JPEG decode, Whisper transcript).
- **ParsedDocument.blocks** — preserves structure for composite docs (future:
  PDF pages, video shots).

## 4. Complexity

| Parser | Typical cost |
| ------ | ------------ |
| Text | O(n) bytes |
| PNG | O(n) decompress + O(H×W) filter reconstruction |
| WAV | O(samples) |
| Video (ffmpeg) | dominated by decode; bounded by ``max_frames`` |

## 5. Limitations (by design)

- JPEG **decode** not built-in (metadata only); see EXTENSION.md.
- Video **frame extract** requires ``ffmpeg`` on PATH.
- No PDF/DOCX in v1 — extension points documented.
