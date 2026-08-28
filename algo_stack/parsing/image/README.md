# Image Parser — 图

Decodes **PNG** to RGB `float64` arrays; reads **JPEG/GIF** metadata.

```python
from algo_stack.parsing.image import ImageParser
doc = ImageParser().parse_bytes(png_bytes, "x.png")
rgb = doc.arrays["rgb"]  # (H, W, 3)
```

See [`../EXTENSION.md`](../EXTENSION.md) for Pillow OCR hooks.
