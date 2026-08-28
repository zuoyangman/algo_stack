"""Audio parser — 音 (WAV decode; metadata for other formats)."""

from __future__ import annotations

import io
import struct
import wave

import numpy as np

from algo_stack.parsing._types import ContentBlock, Modality, ParsedDocument


class AudioParser:
    """Decode PCM WAV; report metadata for MP3/FLAC/OGG (no decode without extensions)."""

    modality = Modality.AUDIO

    def parse_bytes(self, data: bytes, source: str) -> ParsedDocument:
        if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WAVE":
            waveform, meta = self._parse_wav(data)
            text = None
        elif data.startswith(b"ID3") or (len(data) >= 2 and data[0] == 0xFF and (data[1] & 0xE0) == 0xE0):
            waveform, meta = None, {"format": "mp3", "note": "MP3 decode not built-in; use EXTENSION.md hooks"}
            text = "[MP3 audio — attach Whisper/faster-whisper for transcript]"
        elif data.startswith(b"fLaC"):
            meta = {"format": "flac", "note": "FLAC decode requires optional extension"}
            waveform, text = None, "[FLAC audio]"
        elif data.startswith(b"OggS"):
            meta = {"format": "ogg"}
            waveform, text = None, "[OGG audio]"
        else:
            raise ValueError(f"Unsupported audio format for {source!r}.")

        arrays = {"waveform": waveform} if waveform is not None else {}
        if waveform is not None:
            meta["samples"] = int(waveform.shape[0])
            meta["duration_sec"] = meta["samples"] / meta["sample_rate"]

        block = ContentBlock(modality=Modality.AUDIO, text=text, metadata=meta, arrays=arrays)
        return ParsedDocument(
            source=source,
            modality=Modality.AUDIO,
            text=text,
            metadata=meta,
            arrays=arrays,
            blocks=[block],
        )

    def _parse_wav(self, data: bytes) -> tuple[np.ndarray, dict]:
        with wave.open(io.BytesIO(data), "rb") as wf:
            n_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            sample_rate = wf.getframerate()
            n_frames = wf.getnframes()
            pcm = wf.readframes(n_frames)

        if sample_width == 2:
            ints = np.frombuffer(pcm, dtype=np.int16)
        elif sample_width == 1:
            ints = np.frombuffer(pcm, dtype=np.uint8).astype(np.int16) - 128
        else:
            raise ValueError(f"WAV sample width {sample_width} not supported.")

        if n_channels > 1:
            ints = ints.reshape(-1, n_channels).mean(axis=1)

        waveform = ints.astype(np.float64) / (2 ** (8 * sample_width - 1))
        meta = {
            "format": "wav",
            "sample_rate": sample_rate,
            "channels": n_channels,
            "sample_width": sample_width,
        }
        return waveform, meta

    @staticmethod
    def parse_wav_header(data: bytes) -> dict:
        """Lightweight header-only parse (public helper)."""

        if len(data) < 44:
            return {}
        if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
            return {}
        sample_rate = struct.unpack("<I", data[24:28])[0]
        channels = struct.unpack("<H", data[22:24])[0]
        bits = struct.unpack("<H", data[34:36])[0]
        return {"sample_rate": sample_rate, "channels": channels, "bits_per_sample": bits}
