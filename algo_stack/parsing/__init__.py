"""Multimodal document parsing — 文 / 图 / 音 / 视.

Unified entry point::

    from algo_stack.parsing import DocumentParser, parse

    doc = parse("report.md")          # text
    doc = parse("photo.png")          # image → ndarray + metadata
    doc = parse("clip.wav")           # audio → waveform
    doc = parse("demo.mp4")           # video → metadata (+ frames if ffmpeg present)
"""

from algo_stack.parsing._types import ContentBlock, Modality, ParsedDocument
from algo_stack.parsing.document_parser import DocumentParser, parse

__all__ = [
    "ContentBlock",
    "Modality",
    "ParsedDocument",
    "DocumentParser",
    "parse",
]
