"""Shared types for the multimodal parsing pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class Modality(str, Enum):
    """Content modality — 文 / 图 / 音 / 视."""

    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    UNKNOWN = "unknown"


@dataclass
class ContentBlock:
    """One logical chunk inside a parsed document (paragraph, frame, clip, …)."""

    modality: Modality
    text: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    arrays: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class ParsedDocument:
    """Result of parsing a file or byte stream.

    Attributes
    ----------
    source : str
        File path or ``"<bytes>"`` when parsed from memory.
    modality : Modality
        Primary modality detected for this source.
    text : str | None
        Flattened textual content (body, transcript, OCR placeholder, …).
    metadata : dict
        Format-specific metadata (page count, duration, MIME, …).
    arrays : dict[str, ndarray]
        Numeric payloads, e.g. ``{"rgb": H×W×3}``, ``{"waveform": T}``,
        ``{"frames": N×H×W×3}``.
    blocks : list[ContentBlock]
        Structured sub-units (multi-page text, video shots, …).
    """

    source: str
    modality: Modality
    text: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    arrays: dict[str, np.ndarray] = field(default_factory=dict)
    blocks: list[ContentBlock] = field(default_factory=list)

    def summary(self) -> str:
        parts = [f"modality={self.modality.value}", f"source={self.source!r}"]
        if self.text:
            preview = self.text[:80].replace("\n", " ")
            parts.append(f"text_len={len(self.text)} preview={preview!r}…")
        if self.arrays:
            parts.append("arrays=" + ", ".join(f"{k}{v.shape}" for k, v in self.arrays.items()))
        if self.metadata:
            keys = ", ".join(sorted(self.metadata.keys())[:8])
            parts.append(f"metadata_keys=[{keys}]")
        return "ParsedDocument(" + "; ".join(parts) + ")"
