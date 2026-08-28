"""Video parser — 视 (ffprobe/ffmpeg when available; container sniffing fallback)."""

from __future__ import annotations

import json
import shutil
import subprocess
import struct
from typing import Any

import numpy as np

from algo_stack.parsing._types import ContentBlock, Modality, ParsedDocument


class VideoParser:
    """Extract video metadata and optional frames via ffmpeg/ffprobe."""

    modality = Modality.VIDEO

    def __init__(self, *, max_frames: int = 8) -> None:
        self.max_frames = max_frames

    def parse_bytes(self, data: bytes, source: str) -> ParsedDocument:
        meta = self._sniff_container(data, source)
        text = f"[Video {meta.get('format', 'unknown')}"
        if meta.get("duration_sec"):
            text += f", duration={meta['duration_sec']:.2f}s"
        text += "]"

        arrays: dict[str, np.ndarray] = {}
        blocks: list[ContentBlock] = []

        ffprobe_meta = self._ffprobe(source if not source.startswith("<") else None, data)
        if ffprobe_meta:
            meta.update(ffprobe_meta)
            streams = ffprobe_meta.get("streams", [])
            for s in streams:
                if s.get("codec_type") == "video":
                    meta.setdefault("width", s.get("width"))
                    meta.setdefault("height", s.get("height"))
                    meta.setdefault("fps", self._parse_fps(s.get("r_frame_rate", "")))
                if s.get("codec_type") == "audio":
                    meta["has_audio"] = True

        frames = self._extract_frames_ffmpeg(source if not source.startswith("<") else None, data)
        if frames is not None and frames.size:
            arrays["frames"] = frames
            meta["frame_count"] = int(frames.shape[0])
            for i in range(frames.shape[0]):
                blocks.append(
                    ContentBlock(
                        modality=Modality.VIDEO,
                        metadata={"frame_index": i},
                        arrays={"rgb": frames[i]},
                    )
                )

        if not blocks:
            blocks.append(ContentBlock(modality=Modality.VIDEO, text=text, metadata=meta))

        return ParsedDocument(
            source=source,
            modality=Modality.VIDEO,
            text=text,
            metadata=meta,
            arrays=arrays,
            blocks=blocks,
        )

    def _sniff_container(self, data: bytes, source: str) -> dict[str, Any]:
        ext = source.rsplit(".", 1)[-1].lower() if "." in source else ""
        meta: dict[str, Any] = {"format": ext or "unknown"}
        if len(data) >= 12 and data[4:8] == b"ftyp":
            meta["format"] = "mp4/mov"
            meta["brand"] = data[8:12].decode("ascii", errors="replace")
        elif len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"AVI ":
            meta["format"] = "avi"
        elif data.startswith(b"\x1aE\xdf\xa3"):
            meta["format"] = "mkv/webm"
        return meta

    def _ffprobe(self, path: str | None, data: bytes) -> dict[str, Any] | None:
        if not shutil.which("ffprobe"):
            meta = {"ffprobe_available": False}
            return meta
        try:
            if path is not None:
                cmd = [
                    "ffprobe", "-v", "quiet", "-print_format", "json",
                    "-show_format", "-show_streams", path,
                ]
                out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=30)
            else:
                cmd = [
                    "ffprobe", "-v", "quiet", "-print_format", "json",
                    "-show_format", "-show_streams", "-i", "pipe:0",
                ]
                out = subprocess.check_output(cmd, input=data, stderr=subprocess.DEVNULL, timeout=30)
            info = json.loads(out.decode("utf-8"))
            fmt = info.get("format", {})
            result: dict[str, Any] = {
                "ffprobe_available": True,
                "duration_sec": float(fmt.get("duration", 0) or 0),
                "bit_rate": fmt.get("bit_rate"),
                "streams": info.get("streams", []),
            }
            return result
        except (subprocess.CalledProcessError, json.JSONDecodeError, subprocess.TimeoutExpired):
            return {"ffprobe_available": True, "ffprobe_error": True}

    def _extract_frames_ffmpeg(self, path: str | None, data: bytes) -> np.ndarray | None:
        if not shutil.which("ffmpeg"):
            return None
        n = max(1, self.max_frames)
        try:
            if path is not None:
                cmd = [
                    "ffmpeg", "-v", "error", "-i", path,
                    "-vf", f"select=not(mod(n\\,{max(1, 30 // n)})),scale=64:-1",
                    "-frames:v", str(n),
                    "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
                ]
                raw = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=60)
            else:
                return None  # pipe decode omitted for simplicity
            if not raw:
                return None
            # Infer frame size from ffprobe would be better; assume 64×H
            # Use fixed 64×64 if unknown
            frame_size = len(raw) // n
            if frame_size % (64 * 3) == 0:
                h = frame_size // (64 * 3)
                frames = np.frombuffer(raw, dtype=np.uint8).reshape(n, h, 64, 3)
            else:
                side = int((frame_size // 3) ** 0.5)
                frames = np.frombuffer(raw, dtype=np.uint8).reshape(n, side, side, 3)
            return frames.astype(np.float64) / 255.0
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return None

    @staticmethod
    def _parse_fps(rate: str) -> float | None:
        if not rate or rate == "0/0":
            return None
        if "/" in rate:
            num, den = rate.split("/", 1)
            return float(num) / float(den) if float(den) else None
        return float(rate)
