"""Tests for multimodal document parsing (文/图/音/视)."""

from __future__ import annotations

import io
import json
import struct
import tempfile
import wave
from pathlib import Path

import numpy as np
import pytest

from algo_stack.parsing import DocumentParser, Modality, parse
from algo_stack.parsing.registry import sniff_modality
from algo_stack.parsing.text import TextParser
from algo_stack.parsing.image import ImageParser
from algo_stack.parsing.audio import AudioParser


def _minimal_png(h: int = 3, w: int = 3) -> bytes:
    import zlib

    def chunk(tag: bytes, payload: bytes) -> bytes:
        crc = zlib.crc32(tag + payload) & 0xFFFFFFFF
        return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", crc)

    rows = []
    for _ in range(h):
        row = bytes([0]) + bytes([255, 0, 0]) * w
        rows.append(row)
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(b"".join(rows)))
        + chunk(b"IEND", b"")
    )


def _minimal_wav() -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(8000)
        wf.writeframes(np.zeros(800, dtype=np.int16).tobytes())
    return buf.getvalue()


class TestSniffModality:
    def test_extensions(self):
        assert sniff_modality(b"", "a.md") == Modality.TEXT
        assert sniff_modality(b"", "a.png") == Modality.IMAGE
        assert sniff_modality(b"", "a.wav") == Modality.AUDIO
        assert sniff_modality(b"", "a.mp4") == Modality.VIDEO

    def test_magic(self):
        assert sniff_modality(_minimal_png()[:16], "x") == Modality.IMAGE
        assert sniff_modality(_minimal_wav()[:16], "x") == Modality.AUDIO


class TestTextParser:
    def test_markdown(self):
        doc = TextParser().parse_bytes(b"# Title\n\nBody", "t.md")
        assert doc.modality == Modality.TEXT
        assert "Title" in doc.text
        assert len(doc.blocks) >= 1

    def test_json(self):
        payload = json.dumps({"a": 1}).encode()
        doc = TextParser().parse_bytes(payload, "t.json")
        assert '"a"' in doc.text
        assert doc.metadata["format"] == "json"

    def test_html_strip(self):
        doc = TextParser().parse_bytes(b"<p>Hello</p><div>World</div>", "t.html")
        assert "Hello" in doc.text and "World" in doc.text


class TestImageParser:
    def test_png_rgb_shape(self):
        png = _minimal_png(4, 5)
        doc = ImageParser().parse_bytes(png, "x.png")
        assert doc.arrays["rgb"].shape == (4, 5, 3)
        assert doc.metadata["format"] == "png"


class TestAudioParser:
    def test_wav_waveform(self):
        doc = AudioParser().parse_bytes(_minimal_wav(), "x.wav")
        assert doc.arrays["waveform"].shape[0] == 800
        assert doc.metadata["sample_rate"] == 8000


class TestDocumentParser:
    def test_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "note.txt"
            p.write_text("hello multimodal", encoding="utf-8")
            doc = parse(p)
            assert doc.text == "hello multimodal"

    def test_bytes_hint(self):
        doc = parse(_minimal_png(), hint="inline.png")
        assert doc.modality == Modality.IMAGE

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            DocumentParser().parse(b"\x00\x01\x02\x03\xff\xfe", hint=".bin")
