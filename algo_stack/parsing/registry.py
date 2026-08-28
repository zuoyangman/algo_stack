"""Parser registration and modality sniffing."""

from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Callable, Protocol

from algo_stack.parsing._types import Modality, ParsedDocument

ParserFn = Callable[[bytes, str], ParsedDocument]


class BaseParser(Protocol):
    modality: Modality

    def parse_bytes(self, data: bytes, source: str) -> ParsedDocument: ...


# Extension → modality for routing when MIME is ambiguous.
_EXTENSION_MAP: dict[str, Modality] = {
    # 文 text
    ".txt": Modality.TEXT,
    ".md": Modality.TEXT,
    ".markdown": Modality.TEXT,
    ".json": Modality.TEXT,
    ".csv": Modality.TEXT,
    ".html": Modality.TEXT,
    ".htm": Modality.TEXT,
    ".xml": Modality.TEXT,
    # 图 image
    ".png": Modality.IMAGE,
    ".jpg": Modality.IMAGE,
    ".jpeg": Modality.IMAGE,
    ".gif": Modality.IMAGE,
    ".bmp": Modality.IMAGE,
    ".webp": Modality.IMAGE,
    # 音 audio
    ".wav": Modality.AUDIO,
    ".wave": Modality.AUDIO,
    ".flac": Modality.AUDIO,
    ".mp3": Modality.AUDIO,
    ".ogg": Modality.AUDIO,
    ".m4a": Modality.AUDIO,
    # 视 video
    ".mp4": Modality.VIDEO,
    ".mkv": Modality.VIDEO,
    ".webm": Modality.VIDEO,
    ".avi": Modality.VIDEO,
    ".mov": Modality.VIDEO,
}

# Magic-byte sniffing (first bytes).
_SIGNATURES: list[tuple[bytes, Modality, str]] = [
    (b"\x89PNG\r\n\x1a\n", Modality.IMAGE, "png"),
    (b"\xff\xd8\xff", Modality.IMAGE, "jpeg"),
    (b"GIF87a", Modality.IMAGE, "gif"),
    (b"GIF89a", Modality.IMAGE, "gif"),
    (b"RIFF", Modality.AUDIO, "riff"),  # WAV/WebP/AVI share RIFF — refined below
    (b"ID3", Modality.AUDIO, "mp3"),
    (b"\x1aE\xdf\xa3", Modality.VIDEO, "mkv/webm"),
    (b"\x00\x00\x00", Modality.VIDEO, "mp4/mov"),  # ftyp often at offset 4
]

_REGISTRY: dict[Modality, BaseParser] = {}


def register_parser(parser: BaseParser) -> BaseParser:
    """Register a parser instance for its ``modality`` (decorator-friendly)."""

    _REGISTRY[parser.modality] = parser
    return parser


def get_parser(modality: Modality) -> BaseParser:
    if modality not in _REGISTRY:
        raise ValueError(
            f"No parser registered for modality {modality!r}. "
            f"Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[modality]


def sniff_modality(data: bytes, source: str = "") -> Modality:
    """Guess modality from filename extension and/or magic bytes."""

    ext = Path(source).suffix.lower()
    if ext in _EXTENSION_MAP:
        mod = _EXTENSION_MAP[ext]
        if mod != Modality.AUDIO or not data.startswith(b"RIFF"):
            if ext not in (".wav", ".wave") or data[8:12] == b"WAVE":
                return mod

    if data.startswith(b"\x89PNG"):
        return Modality.IMAGE
    if data.startswith(b"\xff\xd8\xff"):
        return Modality.IMAGE
    if data.startswith(b"GIF8"):
        return Modality.IMAGE
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return Modality.AUDIO
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"AVI ":
        return Modality.VIDEO
    if len(data) >= 8 and data[4:8] == b"ftyp":
        return Modality.VIDEO
    if data.startswith(b"ID3") or (len(data) >= 2 and data[0] == 0xFF and data[1] & 0xE0 == 0xE0):
        return Modality.AUDIO

    # Mostly printable → text
    if data and len(data) < 10_000_000:
        sample = data[:4096]
        try:
            sample.decode("utf-8")
            non_print = sum(1 for b in sample if b < 9 and b not in (0,))
            if non_print / max(len(sample), 1) < 0.05:
                return Modality.TEXT
        except UnicodeDecodeError:
            pass

    mime, _ = mimetypes.guess_type(source)
    if mime:
        if mime.startswith("text/") or mime in ("application/json", "application/xml"):
            return Modality.TEXT
        if mime.startswith("image/"):
            return Modality.IMAGE
        if mime.startswith("audio/"):
            return Modality.AUDIO
        if mime.startswith("video/"):
            return Modality.VIDEO

    return Modality.UNKNOWN
