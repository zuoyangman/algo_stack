"""Runnable demo: ``python -m algo_stack.parsing.example``."""

from __future__ import annotations

import io
import struct
import tempfile
import wave
from pathlib import Path

import numpy as np

from algo_stack.parsing import DocumentParser, parse


def _write_text(path: Path) -> None:
    path.write_text("# Demo\n\nMultimodal parsing — 文图音视.\n", encoding="utf-8")


def _write_png(path: Path, h: int = 4, w: int = 4) -> None:
    """Write a minimal PNG (filter type 0, RGB)."""

    import zlib

    def chunk(tag: bytes, payload: bytes) -> bytes:
        crc = zlib.crc32(tag + payload) & 0xFFFFFFFF
        return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", crc)

    raw_rows = []
    for y in range(h):
        row = bytes([0])  # filter none
        for x in range(w):
            row += bytes([x * 40, y * 40, 128])
        raw_rows.append(row)
    compressed = zlib.compress(b"".join(raw_rows))
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", compressed) + chunk(b"IEND", b"")
    path.write_bytes(png)


def _write_wav(path: Path, sr: int = 8000, duration: float = 0.25) -> None:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    tone = (0.3 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(tone.tobytes())


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        text_p = root / "demo.md"
        png_p = root / "demo.png"
        wav_p = root / "demo.wav"
        _write_text(text_p)
        _write_png(png_p)
        _write_wav(wav_p)

        parser = DocumentParser()
        for p in (text_p, png_p, wav_p):
            doc = parser.parse(p)
            print(doc.summary())

        # bytes + hint
        doc = parse(png_p.read_bytes(), hint="inline.png")
        print("from bytes:", doc.arrays["rgb"].shape)


if __name__ == "__main__":
    main()
