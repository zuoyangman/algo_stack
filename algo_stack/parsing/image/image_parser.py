"""Image parser — 图 (PNG decode + JPEG/GIF metadata)."""

from __future__ import annotations

import struct
import zlib

import numpy as np

from algo_stack.parsing._types import ContentBlock, Modality, ParsedDocument


class ImageParser:
    """Decode PNG to RGB ``ndarray``; extract metadata for JPEG/GIF."""

    modality = Modality.IMAGE

    def parse_bytes(self, data: bytes, source: str) -> ParsedDocument:
        if data.startswith(b"\x89PNG\r\n\x1a\n"):
            rgb, meta = self._parse_png(data)
            text = None
        elif data.startswith(b"\xff\xd8\xff"):
            rgb, meta = None, self._jpeg_metadata(data)
            text = f"[JPEG image {meta.get('width')}×{meta.get('height')}; decode requires Pillow extension]"
        elif data.startswith(b"GIF8"):
            meta = self._gif_metadata(data)
            rgb, text = None, f"[GIF {meta.get('width')}×{meta.get('height')}]"
        else:
            raise ValueError(f"Unsupported or unknown image format for {source!r}.")

        arrays = {"rgb": rgb} if rgb is not None else {}
        block = ContentBlock(modality=Modality.IMAGE, text=text, metadata=meta, arrays=arrays)
        return ParsedDocument(
            source=source,
            modality=Modality.IMAGE,
            text=text,
            metadata={"format": meta.get("format"), **meta},
            arrays=arrays,
            blocks=[block],
        )

    def _parse_png(self, data: bytes) -> tuple[np.ndarray, dict]:
        """Minimal PNG reader (RGBA/RGB, 8-bit, no interlace)."""

        pos = 8
        width = height = bit_depth = color_type = None
        idat = bytearray()
        while pos + 8 <= len(data):
            length = struct.unpack(">I", data[pos : pos + 4])[0]
            ctype = data[pos + 4 : pos + 8]
            chunk = data[pos + 8 : pos + 8 + length]
            pos += 12 + length
            if ctype == b"IHDR":
                width, height, bit_depth, color_type = struct.unpack(">IIBB", chunk[:10])[:4]
            elif ctype == b"IDAT":
                idat.extend(chunk)
            elif ctype == b"IEND":
                break

        if width is None or height is None:
            raise ValueError("PNG missing IHDR chunk.")
        if bit_depth != 8 or color_type not in (2, 6):
            raise ValueError(f"PNG color type {color_type} / bit depth {bit_depth} not supported.")

        raw = zlib.decompress(bytes(idat))
        channels = 4 if color_type == 6 else 3
        stride = 1 + width * channels
        rows = []
        prev = np.zeros(width * channels, dtype=np.uint8)
        i = 0
        for _ in range(height):
            filt = raw[i]
            i += 1
            row = np.frombuffer(raw[i : i + width * channels], dtype=np.uint8).copy()
            i += width * channels
            if filt == 0:
                cur = row
            elif filt == 1:
                cur = (row + np.roll(row, channels)) % 256
                cur[:channels] = row[:channels]
            elif filt == 2:
                cur = (row + prev) % 256
            else:
                cur = row
            rows.append(cur)
            prev = cur

        flat = np.concatenate(rows)
        if channels == 4:
            rgba = flat.reshape(height, width, 4)
            rgb = rgba[..., :3]
        else:
            rgb = flat.reshape(height, width, 3)
        meta = {
            "format": "png",
            "width": int(width),
            "height": int(height),
            "channels": 3,
        }
        return rgb.astype(np.float64) / 255.0, meta

    def _jpeg_metadata(self, data: bytes) -> dict:
        # SOF0 marker scan
        i = 2
        width = height = None
        while i + 4 < len(data):
            if data[i] != 0xFF:
                i += 1
                continue
            marker = data[i + 1]
            if marker in (0xC0, 0xC2) and i + 9 < len(data):
                height = struct.unpack(">H", data[i + 5 : i + 7])[0]
                width = struct.unpack(">H", data[i + 7 : i + 9])[0]
                break
            length = struct.unpack(">H", data[i + 2 : i + 4])[0]
            i += 2 + length
        return {"format": "jpeg", "width": width, "height": height}

    def _gif_metadata(self, data: bytes) -> dict:
        if len(data) < 10:
            return {"format": "gif"}
        width, height = struct.unpack("<HH", data[6:10])
        return {"format": "gif", "width": width, "height": height}
