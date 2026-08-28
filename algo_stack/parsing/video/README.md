# Video Parser — 视

Sniffs MP4/MKV/AVI containers; uses **ffprobe/ffmpeg** when available.

```python
from algo_stack.parsing.video import VideoParser
doc = VideoParser(max_frames=4).parse_bytes(data, "clip.mp4")
```

See [`../EXTENSION.md`](../EXTENSION.md) for PyAV and caption models.
