# Audio Parser — 音

Decodes **WAV** PCM to mono waveform; reports metadata for MP3/FLAC/OGG.

```python
from algo_stack.parsing.audio import AudioParser
doc = AudioParser().parse_bytes(wav_bytes, "x.wav")
waveform = doc.arrays["waveform"]
```

See [`../EXTENSION.md`](../EXTENSION.md) for Whisper / librosa integration.
