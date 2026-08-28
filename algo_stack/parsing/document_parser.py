"""Unified facade: route files/bytes to modality-specific parsers."""

from __future__ import annotations

from pathlib import Path

from algo_stack.parsing._types import Modality, ParsedDocument
from algo_stack.parsing.registry import get_parser, register_parser, sniff_modality

# Side-effect: register built-in parsers on import.
from algo_stack.parsing.audio import AudioParser  # noqa: E402
from algo_stack.parsing.image import ImageParser  # noqa: E402
from algo_stack.parsing.text import TextParser  # noqa: E402
from algo_stack.parsing.video import VideoParser  # noqa: E402

register_parser(TextParser())
register_parser(ImageParser())
register_parser(AudioParser())
register_parser(VideoParser())


class DocumentParser:
    """Open-source multimodal document parser (文 / 图 / 音 / 视).

    Parameters
    ----------
    force_modality : Modality | None
        Skip auto-detection and always use this parser.
    encoding : str
        Text decoding fallback when parsing text modalities.
    max_text_chars : int | None
        Truncate extracted text to this many characters (``None`` = no limit).
    video_max_frames : int
        Upper bound on frames extracted when ffmpeg is available.
    """

    def __init__(
        self,
        *,
        force_modality: Modality | None = None,
        encoding: str = "utf-8",
        max_text_chars: int | None = None,
        video_max_frames: int = 8,
    ) -> None:
        self.force_modality = force_modality
        self.encoding = encoding
        self.max_text_chars = max_text_chars
        self.video_max_frames = video_max_frames

    def parse(self, source: str | Path | bytes, *, hint: str | None = None) -> ParsedDocument:
        """Parse a filesystem path or raw ``bytes``.

        ``hint`` is used as a synthetic filename when ``source`` is ``bytes``
        (e.g. ``hint="clip.wav"``).
        """

        if isinstance(source, (str, Path)):
            path = Path(source)
            data = path.read_bytes()
            name = str(path)
        else:
            data = bytes(source)
            name = hint or "<bytes>"

        modality = self.force_modality or sniff_modality(data, name)
        if modality == Modality.UNKNOWN:
            raise ValueError(
                f"Could not detect modality for {name!r}. "
                "Pass force_modality= or a filename hint with a known extension."
            )

        parser = get_parser(modality)
        doc = parser.parse_bytes(data, name)

        if self.max_text_chars is not None and doc.text and len(doc.text) > self.max_text_chars:
            doc.text = doc.text[: self.max_text_chars]
            doc.metadata["text_truncated"] = True

        return doc

    def parse_many(self, sources: list[str | Path]) -> list[ParsedDocument]:
        return [self.parse(s) for s in sources]


def parse(
    source: str | Path | bytes,
    *,
    hint: str | None = None,
    **kwargs,
) -> ParsedDocument:
    """Functional shortcut: ``parse("file.png")`` or ``parse(data, hint="x.png")``."""

    return DocumentParser(**kwargs).parse(source, hint=hint)
